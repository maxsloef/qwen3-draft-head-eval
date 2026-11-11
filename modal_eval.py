"""Matched evaluation of the public EAGLE-3 draft heads for Qwen3-8B as predictors of the model's future tokens.

Qwen3-8B answers the 480 Spec-Bench prompts greedily (thinking off and thinking on). Each head then gets the
model's real hidden states and is scored, at every generated position, on predicting the tokens the model
actually produced 1..K steps later (step 1 = the token after next, as in EAGLE drafting).

Launch:  uv run --no-project --python 3.12 --with modal modal run --detach modal_eval.py::main
Fetch:   uv run --no-project --python 3.12 --with modal modal run modal_eval.py::fetch
"""
import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
BASE = "Qwen/Qwen3-8B"
HEADS = {  # name: (repo, weight file, norm_before_residual, trunk-layer offset)
    "whiflash": ("WhiFlash/Qwen3-8B-EAGLE3", "model.safetensors", False, 0),
    "tengyunw": ("Tengyunw/qwen3_8b_eagle3", "pytorch_model.bin", False, 0),
    "angelslim": ("AngelSlim/Qwen3-8B_eagle3", "pytorch_model.bin", False, 0),
    "redhat": ("RedHatAI/Qwen3-8B-speculator.eagle3", "model.safetensors", True, 0),
    "redhat+1": ("RedHatAI/Qwen3-8B-speculator.eagle3", "model.safetensors", True, 1),   # same weights, layers 3/19/34 (smoke run preferred these)
}
LAYERS = (2, 18, 33)            # inputs to trunk layers 2, n//2, n-3 (n = 36)
VARIANTS = [(nbr, off) for nbr in (False, True) for off in (0, 1)]   # convention check: norm_before_residual x layer offset
K = 7
METRICS = ["n", "top1", "top5", "covered", "overlap", "prefix", "base_top1"]

hf = modal.Volume.from_name("owl-hf")
out = modal.Volume.from_name("draft-head-eval", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub")
         .env({"HF_HOME": "/hf"})
         .add_local_file(HERE / "data" / "spec_bench_question.jsonl", "/data/q.jsonl")
         .add_local_python_source("heads", "summarize"))
app = modal.App("draft-head-eval", image=image)


@app.function(volumes={"/out": out}, timeout=1800, cpu=4)
def prep():
    import os
    from huggingface_hub import hf_hub_download
    for repo, fn, _, _ in HEADS.values():
        if not os.path.exists(f"/out/heads/{repo}/{fn}"):
            hf_hub_download(repo, fn, local_dir=f"/out/heads/{repo}", cache_dir="/tmp/hfc")
    out.commit()


@app.function(gpu="L40S", volumes={"/hf": hf, "/out": out}, timeout=3600, cpu=8, memory=64 * 1024)
def run_shard(cond: str, shard: int, n_shards: int, max_new: int = 256, limit: int = 0):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from heads import Head

    torch.set_grad_enabled(False)
    dev = "cuda"
    rows = [json.loads(l) for l in open("/data/q.jsonl")]
    cats = sorted({r["category"] for r in rows})
    rows = rows[shard::n_shards][: limit or None]
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16, device_map=dev).eval()
    E = model.get_input_embeddings().weight
    heads = {}
    for name, (repo, fn, nbr, off) in HEADS.items():
        p = f"/out/heads/{repo}/{fn}"
        sd = load_file(p) if fn.endswith(".safetensors") else torch.load(p, map_location="cpu", weights_only=True)
        if shard == 0:
            print(name, {k: tuple(v.shape) for k, v in sd.items()}, flush=True)
        heads[name] = Head(sd, E, nbr, dev)
        heads[name].off = off
        if heads[name].own_embed and shard == 0:
            print(name, "own embedding equals base:", bool((heads[name].embed == E).all()), flush=True)

    # ---- 1. greedy generation, batched by length ----
    prompts = [tok(tok.apply_chat_template([{"role": "user", "content": r["turns"][0]}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=(cond == "think")), add_special_tokens=False).input_ids for r in rows]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(rows)), key=lambda j: len(prompts[j]))
    seqs = {}
    i = 0
    while i < len(order):
        bs = 1
        while i + bs < len(order) and bs < 48 and (bs + 1) * (len(prompts[order[i + bs]]) + max_new) <= 40000:
            bs += 1
        idx = order[i:i + bs]
        L = max(len(prompts[j]) for j in idx)
        ids = torch.full((bs, L), pad, dtype=torch.long)
        am = torch.zeros((bs, L), dtype=torch.long)
        for b, j in enumerate(idx):
            ids[b, L - len(prompts[j]):] = torch.tensor(prompts[j])
            am[b, L - len(prompts[j]):] = 1
        gen = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=pad)[:, L:].tolist()
        for b, j in enumerate(idx):
            g = []
            for t in gen[b]:
                g.append(t)
                if t in eos:
                    break
            seqs[j] = (prompts[j], g)
        i += bs
        print(f"[{cond}/{shard}] generated {i}/{len(order)}", flush=True)

    # ---- 2. teacher-forced trunk pass + head scoring ----
    grabbed = {}
    for li in sorted({l + o for l in LAYERS for o in (0, 1)}):
        def pre(mod, args, kwargs, li=li):
            grabbed[li] = args[0] if args else kwargs["hidden_states"]
        model.model.layers[li].register_forward_pre_hook(pre, with_kwargs=True)
    stats = {n: torch.zeros(len(cats), K, len(METRICS), dtype=torch.float64) for n in heads}
    var = {n: torch.zeros(len(VARIANTS), 2, dtype=torch.float64) for n in heads}      # [variant] -> (n, step-1 top1)
    seqlen = lambda j: len(seqs[j][0]) + len(seqs[j][1])
    order = sorted(seqs, key=seqlen)
    i = 0
    while i < len(order):
        bs = 1
        while i + bs < len(order) and bs < 32 and (bs + 1) * seqlen(order[i + bs]) ** 2 <= 16e6:
            bs += 1
        idx = order[i:i + bs]
        lens = torch.tensor([seqlen(j) for j in idx], device=dev)
        plen = torch.tensor([len(seqs[j][0]) for j in idx], device=dev)
        cat_id = torch.tensor([cats.index(rows[j]["category"]) for j in idx], device=dev)
        T = int(lens.max())
        ids = torch.full((bs, T), pad, dtype=torch.long, device=dev)
        for b, j in enumerate(idx):
            ids[b, : lens[b]] = torch.tensor(seqs[j][0] + seqs[j][1], device=dev)
        am = (torch.arange(T, device=dev)[None] < lens[:, None]).long()
        final = model.model(input_ids=ids, attention_mask=am).last_hidden_state            # [B,T,d], after the final norm
        feats = {o: torch.cat([grabbed[l + o] for l in LAYERS], -1)[:, : T - 1].float() for o in (0, 1)}
        Tm = T - 1
        pos = torch.arange(Tm, device=dev)[None].expand(bs, Tm)
        # anchor i is scored when its first target (token i+2) is generated text and its last (token i+K+1) exists
        valid = (pos + 2 >= plen[:, None]) & (pos + K + 1 <= lens[:, None] - 1)
        vb, vi = valid.nonzero(as_tuple=True)
        vcat = cat_id[vb]
        tok_at = lambda off: ids.gather(1, (pos + off).clamp(max=T - 1))
        for name, h in heads.items() if len(vb) else []:
            for vn, (nbr, off) in enumerate(VARIANTS):          # off-convention variants, step 1 only
                if (nbr, off) == (h.nbr, h.off):
                    continue
                g = h.step1(tok_at(1), feats[off], nbr=nbr, keep=False)
                hits = sum((h.tids[h.logits(g[vb[c:c + 1024], vi[c:c + 1024]]).argmax(-1)] == ids[vb[c:c + 1024], vi[c:c + 1024] + 2]).sum().item()
                           for c in range(0, len(vb), 1024))
                var[name][vn] += torch.tensor([len(vb), hits], dtype=torch.float64)
            g = h.step1(tok_at(1), feats[h.off])
            alive = torch.ones(len(vb), dtype=torch.bool, device=dev)
            for s in range(1, K + 1):
                if s > 1:
                    g = h.step_next(tok_at(s), g, s)
                true = ids[vb, vi + s + 1]
                m = torch.zeros(len(vb), len(METRICS), dtype=torch.float64, device=dev)
                for c in range(0, len(vb), 256):
                    sl = slice(c, c + 256)
                    lg = h.logits(g[vb[sl], vi[sl]])
                    top5 = h.tids[lg.topk(5, -1).indices]
                    bp = model.lm_head(final[vb[sl], vi[sl] + s]).float().softmax(-1)      # trunk's own next-token distribution
                    hit = top5[:, 0] == true[sl]
                    alive[sl] &= hit
                    m[sl, 0] = 1
                    m[sl, 1] = hit
                    m[sl, 2] = (top5 == true[sl, None]).any(-1)
                    m[sl, 3] = h.t2d[true[sl]]
                    m[sl, 4] = torch.minimum(bp[:, h.tids], lg.softmax(-1)).sum(-1)        # 1 - total variation
                    m[sl, 5] = alive[sl]
                    m[sl, 6] = bp.argmax(-1) == true[sl]
                stats[name][:, s - 1] += torch.zeros(len(cats), len(METRICS), dtype=torch.float64, device=dev).index_add_(0, vcat, m).cpu()
                if s == 1:
                    var[name][VARIANTS.index((h.nbr, h.off))] += torch.tensor([len(vb), m[:, 1].sum().item()], dtype=torch.float64)
            h.clear()
        i += bs
        print(f"[{cond}/{shard}] scored {i}/{len(order)} (T={T}, anchors={len(vb)})", flush=True)
    return {"cond": cond, "shard": shard, "cats": cats,
            "stats": {n: v.tolist() for n, v in stats.items()}, "variants": {n: v.tolist() for n, v in var.items()},
            "gens": [{"question_id": rows[j]["question_id"], "category": rows[j]["category"], "prompt_tokens": len(seqs[j][0]),
                      "gen_tokens": len(seqs[j][1]), "text": tok.decode(seqs[j][1])} for j in sorted(seqs)]}


@app.function(volumes={"/out": out}, timeout=2 * 3600, cpu=2)
def orchestrate(n_shards: int = 4, max_new: int = 256, limit: int = 0, tag: str = "main", conds: str = "nothink,think"):
    import torch
    prep.remote()
    jobs = [(c, s, n_shards, max_new, limit) for c in conds.split(",") for s in range(n_shards)]
    res = {"K": K, "metrics": METRICS, "variants": VARIANTS, "max_new": max_new, "conds": {}}
    gens = []
    for r in run_shard.starmap(jobs):
        c = res["conds"].setdefault(r["cond"], {"stats": {}, "variants": {}})
        res["cats"] = r["cats"]
        for key in ("stats", "variants"):
            for n, v in r[key].items():
                c[key][n] = (torch.tensor(v) + torch.tensor(c[key][n])).tolist() if n in c[key] else v
        gens += [{"cond": r["cond"], **g} for g in r["gens"]]
    pathlib.Path(f"/out/results/{tag}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/out/results/{tag}/results.json").write_text(json.dumps(res))
    pathlib.Path(f"/out/results/{tag}/generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens))
    from summarize import render
    pathlib.Path(f"/out/results/{tag}/summary.txt").write_text(render(res))
    out.commit()
    return tag


@app.local_entrypoint()
def main(n_shards: int = 4, max_new: int = 256, limit: int = 0, tag: str = "main", conds: str = "nothink,think"):
    fetch(orchestrate.remote(n_shards, max_new, limit, tag, conds))


@app.function(volumes={"/out": out}, timeout=600)
def _read(tag: str):
    from summarize import render
    out.reload()
    d = {f: pathlib.Path(f"/out/results/{tag}/{f}").read_text() for f in ("results.json", "generations.jsonl")}
    return {**d, "summary.txt": render(json.loads(d["results.json"]))}


@app.local_entrypoint()
def fetch(tag: str = "main"):
    (HERE / "results" / tag).mkdir(parents=True, exist_ok=True)
    for f, txt in _read.remote(tag).items():
        (HERE / "results" / tag / f).write_text(txt)
        print("wrote", HERE / "results" / tag / f)
