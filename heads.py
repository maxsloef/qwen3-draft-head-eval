"""EAGLE-3 draft head, re-implemented for scoring (not serving).

All four public Qwen3-8B heads share this layout: fc (3*4096 -> 4096) over three trunk layers,
one decoder layer whose attention reads cat(embedding, feature) (8192 wide), a final norm and
a 32K-token output layer with a draft->target id offset table (d2t).

Conventions checked against SafeAILab/EAGLE (cnets.py) and RedHatAI's eagle3.py:
- position i gets the trunk feature at i and the embedding of token i+1, and predicts token i+2;
- later draft steps reuse the head's own pre-norm output as the feature (no fc), with the next token;
- the residual stream is the feature before hidden_norm (SafeAILab) or after it (RedHatAI, norm_before_residual).
"""
import math

import torch
import torch.nn.functional as F

N_HEADS, N_KV, HEAD_DIM, EPS, THETA = 32, 8, 128, 1e-6, 1e6


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):
    """x [B,H,T,d]; pos [T] or [B,T] (absolute positions)."""
    d = x.shape[-1]
    inv = 1.0 / (THETA ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[..., None].float() * inv
    if ang.dim() == 3:
        ang = ang[:, None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Head:
    def __init__(self, sd, base_embed, norm_before_residual, device):
        pref = "midlayer." if any(k.startswith("midlayer.") for k in sd) else "layers.0."
        g = lambda k: sd[k].to(device=device, dtype=torch.float32)
        self.fc = g("fc.weight")
        self.hn, self.iln, self.pln = g(pref + "hidden_norm.weight"), g(pref + "input_layernorm.weight"), g(pref + "post_attention_layernorm.weight")
        self.wq, self.wk, self.wv, self.wo = (g(pref + f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.wg, self.wu, self.wd = (g(pref + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.norm, self.lm = g("norm.weight"), g("lm_head.weight")
        self.own_embed = "embed_tokens.weight" in sd
        self.embed = sd["embed_tokens.weight"].to(device) if self.own_embed else base_embed
        d2t = sd["d2t"].to(device).long()
        self.tids = torch.arange(d2t.numel(), device=device) + d2t      # draft index -> target token id
        self.t2d = sd["t2d"].to(device).bool()                          # target token id -> is it draftable
        self.nbr = norm_before_residual

    def _inputs(self, tok, hid, nbr):
        emb = self.embed[tok].float()
        if nbr:
            hid = rms(hid, self.hn)
            res = hid
        else:
            res = hid
            hid = rms(hid, self.hn)
        x = torch.cat([rms(emb, self.iln), hid], -1)
        B, T, _ = x.shape
        rep = N_HEADS // N_KV
        q = (x @ self.wq.T).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = (x @ self.wk.T).view(B, T, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        v = (x @ self.wv.T).view(B, T, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        return q, k, v, res

    def _fuse(self, feats):
        return feats @ self.fc.T

    def _finish(self, res, a):
        B, _, T, _ = a.shape
        h = res + a.transpose(1, 2).reshape(B, T, -1) @ self.wo.T
        m = rms(h, self.pln)
        return h + (F.silu(m @ self.wg.T) * (m @ self.wu.T)) @ self.wd.T

    def logits(self, g):
        return rms(g, self.norm) @ self.lm.T

    def step1(self, tok_next, feats3, nbr=None, keep=True):
        """tok_next [B,T] = token i+1 at position i; feats3 [B,T,3*4096]. Returns pre-norm output g [B,T,4096]."""
        nbr = self.nbr if nbr is None else nbr
        q, k, v, res = self._inputs(tok_next, self._fuse(feats3), nbr)
        pos = torch.arange(q.shape[2], device=q.device)
        q, k = rope(q, pos), rope(k, pos)
        if keep:
            self.kr, self.vr, self.kc, self.vc = k, v, [], []
        return self._finish(res, F.scaled_dot_product_attention(q, k, v, is_causal=True))

    def step_next(self, tok, g_prev, s):
        """Draft step s>=2 for every anchor at once. tok [B,T] = token i+s; g_prev = this anchor's previous output.
        The query at anchor i sees the real keys at positions <= i plus this anchor's own earlier draft steps."""
        q, k, v, res = self._inputs(tok, g_prev, self.nbr)
        B, H, T, d = q.shape
        pos = torch.arange(T, device=q.device)[None].expand(B, T) + (s - 1)
        q, k = rope(q, pos), rope(k, pos)
        self.kc.append(k)
        self.vc.append(v)
        real = q @ self.kr.transpose(-1, -2) / math.sqrt(d)
        real = real.masked_fill(torch.ones(T, T, device=q.device, dtype=torch.bool).triu(1), float("-inf"))
        chain = torch.stack([(q * kc).sum(-1) for kc in self.kc], -1) / math.sqrt(d)
        w = torch.cat([real, chain], -1).softmax(-1)
        a = w[..., :T] @ self.vr
        for c, vc in enumerate(self.vc):
            a = a + w[..., T + c, None] * vc
        return self._finish(res, a)

    def clear(self):
        self.kr = self.vr = self.kc = self.vc = None


class MTPHead(Head):
    """Qwen3.5-style MTP block retrofitted onto Qwen3-8B (AngelSpec, AngelSlim/Qwen3-8B-MTP-TTT3).

    Conventions read from AngelSpec (models/draft/mtp.py, training/mtp_trainer.py):
    - position i gets the trunk's last-layer hidden state at i and the embedding of token i+1, predicts token i+2;
    - x = fc(cat(norm_e(emb), norm_h(hidden))), then one Qwen3 decoder layer (q/k norm per head before RoPE), then norm;
    - the output layer is the trunk's own lm_head over the full vocabulary;
    - later steps feed the block's own post-norm output back as the hidden state, at position i + step;
    - the token embedding at sequence position 0 is zeroed.
    """

    def __init__(self, sd, base_embed, lm_weight, device):
        g = lambda k: sd["mtp." + k].to(device=device, dtype=torch.float32)
        self.fc, self.en, self.hn, self.norm = g("fc.weight"), g("pre_fc_norm_embedding.weight"), g("pre_fc_norm_hidden.weight"), g("norm.weight")
        self.iln, self.pln = g("layers.0.input_layernorm.weight"), g("layers.0.post_attention_layernorm.weight")
        self.wq, self.wk, self.wv, self.wo = (g(f"layers.0.self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = g("layers.0.self_attn.q_norm.weight"), g("layers.0.self_attn.k_norm.weight")
        self.wg, self.wu, self.wd = (g(f"layers.0.mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.embed, self.lm, self.own_embed, self.nbr = base_embed, lm_weight, False, None
        self.tids = torch.arange(lm_weight.shape[0], device=device)
        self.t2d = torch.ones(lm_weight.shape[0], dtype=torch.bool, device=device)

    def _fuse(self, feats):
        return feats

    def _inputs(self, tok, hid, nbr):
        emb = self.embed[tok].float()
        emb[:, 0] = 0
        x = torch.cat([rms(emb, self.en), rms(hid, self.hn)], -1) @ self.fc.T
        y = rms(x, self.iln)
        B, T, _ = y.shape
        rep = N_HEADS // N_KV
        q = rms((y @ self.wq.T).view(B, T, N_HEADS, HEAD_DIM), self.qn).transpose(1, 2)
        k = rms((y @ self.wk.T).view(B, T, N_KV, HEAD_DIM), self.kn).transpose(1, 2).repeat_interleave(rep, dim=1)
        v = (y @ self.wv.T).view(B, T, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
        return q, k, v, x

    def _finish(self, res, a):
        return rms(super()._finish(res, a), self.norm)       # post-norm output; also the next step's hidden input

    def logits(self, g):
        return (g.to(self.lm.dtype) @ self.lm.T).float()


class DFly:
    """AngelSpec DFly block drafter for Qwen3-8B (AngelSlim/Qwen3-8B-DFly-Block8), re-implemented for scoring.

    Not a single-layer head: five decoder layers whose attention reads keys/values from the trunk's hidden
    states (five trunk layers, fused per draft layer) and from its own block. Conventions read from AngelSpec
    (models/draft/dflash.py, dflare.py, dfly.py, models/dflash.py):
    - a block for anchor position a is [token a, MASK x (block-1)]; it sees trunk context at positions < a
      and attends bidirectionally inside the block; slot k predicts token a+k, all slots in parallel;
    - context for draft layer l = norm(fc(concat of 5 trunk layers) + sum_t softmax(w[l])[t] * trunk layer t);
    - the optional "hidden correction" adds a SwiGLU of (slot output, embedding of the previous token)
      before the trunk's lm_head, which makes each slot depend on the token just before it.
    """

    LAYER_IDS = (1, 9, 17, 25, 33)      # trunk layers whose outputs are read (angelspec/config/dfly_qwen3_8b_draft_config.json)

    def __init__(self, sd, base_embed, lm_weight, device, block=8, mask_id=151669):
        g = lambda k: sd[k].to(device=device, dtype=torch.float32)
        self.n_layers = sd["layer_fusion_weights"].shape[0]
        self.fw, self.cp, self.cn, self.fn = g("layer_fusion_weights"), g("context_proj.weight"), g("context_norm.weight"), g("final_norm.weight")
        self.L = [{k: g(f"layers.{l}.{n}.weight") for k, n in [
            ("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
            ("qn", "self_attn.q_norm"), ("kn", "self_attn.k_norm"), ("g", "mlp.gate_proj"), ("u", "mlp.up_proj"),
            ("d", "mlp.down_proj"), ("iln", "input_layernorm"), ("pln", "post_attention_layernorm")]} for l in range(self.n_layers)]
        self.c = {k: g(f"hidden_correction.{n}.weight") for k, n in [("hn", "hidden_norm"), ("en", "embed_norm"), ("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj")]}
        self.embed = sd["embed_tokens.weight"].to(device) if "embed_tokens.weight" in sd else base_embed
        self.lm, self.block, self.mask_id = lm_weight, block, mask_id
        self.tids = torch.arange(lm_weight.shape[0], device=device)
        self.t2d = torch.ones(lm_weight.shape[0], dtype=torch.bool, device=device)

    def logits(self, g):
        return (g.to(self.lm.dtype) @ self.lm.T).float()

    def predict(self, ids, feats, anchors):
        """ids [T]; feats [T, 5, D] trunk hidden states; anchors [N]. Returns slot outputs [N, block, D]
        without and with the hidden correction (the latter teacher-forced on the true previous token)."""
        T, N, B, dev = ids.shape[0], anchors.shape[0], self.block, ids.device
        rep, sc_ = N_HEADS // N_KV, 1.0 / math.sqrt(HEAD_DIM)
        base = feats.flatten(1) @ self.cp.T
        pos_c = torch.arange(T, device=dev)
        pos_d = anchors[:, None] + torch.arange(B, device=dev)
        draft = torch.full((N, B), self.mask_id, dtype=torch.long, device=dev)
        draft[:, 0] = ids[anchors]
        h = self.embed[draft].float()
        hide = (pos_c[None] >= anchors[:, None])[:, None, None, :]            # context strictly before the anchor
        for l, W in enumerate(self.L):
            ctx = rms(base + torch.einsum("t,std->sd", self.fw[l].softmax(-1), feats), self.cn)
            y = rms(h, W["iln"])
            q = rope(rms((y @ W["q"].T).view(N, B, N_HEADS, HEAD_DIM), W["qn"]).transpose(1, 2), pos_d)
            kd = rope(rms((y @ W["k"].T).view(N, B, N_KV, HEAD_DIM), W["kn"]).transpose(1, 2), pos_d).repeat_interleave(rep, dim=1)
            vd = (y @ W["v"].T).view(N, B, N_KV, HEAD_DIM).transpose(1, 2).repeat_interleave(rep, dim=1)
            kc = rope(rms((ctx @ W["k"].T).view(1, T, N_KV, HEAD_DIM), W["kn"]).transpose(1, 2), pos_c)[0].repeat_interleave(rep, dim=0)
            vc = (ctx @ W["v"].T).view(T, N_KV, HEAD_DIM).transpose(0, 1).repeat_interleave(rep, dim=0)
            s_ctx = (torch.einsum("nhqd,htd->nhqt", q, kc) * sc_).masked_fill(hide, float("-inf"))
            w = torch.cat([s_ctx, q @ kd.transpose(-1, -2) * sc_], -1).softmax(-1)
            a = torch.einsum("nhqt,htd->nhqd", w[..., :T], vc) + w[..., T:] @ vd
            h = h + a.transpose(1, 2).reshape(N, B, -1) @ W["o"].T
            m = rms(h, W["pln"])
            h = h + (F.silu(m @ W["g"].T) * (m @ W["u"].T)) @ W["d"].T
        g = rms(h, self.fn)
        prev = ids[(pos_d - 1).clamp(min=0, max=T - 1)]
        prev[:, 0] = ids[anchors]
        c = torch.cat([rms(g, self.c["hn"]), rms(self.embed[prev].float(), self.c["en"])], -1)
        return g, g + (F.silu(c @ self.c["g"].T) * (c @ self.c["u"].T)) @ self.c["d"].T
