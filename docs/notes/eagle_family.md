# EAGLE family of speculative-decoding draft heads (EAGLE, EAGLE-2, EAGLE-3, HASS, FSPAD, CORAL, and ecosystem follow-ups) — notes as of November 2026

Source-quality legend used below:
- **[paper]** = read from the arXiv HTML/PDF of the paper itself.
- **[code]** = read directly from the official SafeAILab/EAGLE repository files (downloaded and grepped).
- **[card]** = self-reported on a Hugging Face model card (not peer reviewed, often incomplete methodology).
- **[snippet]** = only seen in a search-result summary; NOT verified against the primary page. Treat as a lead.

Speedup/acceptance-length cells are written `speedup / tau` unless stated otherwise. tau = average acceptance length = tokens generated per draft-verify cycle (includes the one token the target always contributes).

---

## 1. Architecture, inputs, losses, and what changed between EAGLE-1, EAGLE-2, EAGLE-3

### Takeaway
All three versions use the same basic draft head: a single transformer decoder layer (plus small FC fusion layers) that reuses the frozen base model's embedding and (in v1/v2) LM head; the base model is never updated. EAGLE-1 trains feature regression (Smooth L1) + 0.1 x token cross-entropy on top-layer features; EAGLE-2 changes only inference (dynamic draft tree, no retraining); EAGLE-3 drops the feature loss entirely, feeds fused low/mid/high-layer features, and trains with 7 unrolled "training-time test" steps on its own outputs, which is what makes more data pay off.

### Cited Findings

**EAGLE-1 (arXiv 2401.15077, ICML 2024)**
- Draft head ("Autoregression Head") = one FC layer that reduces the concatenation [feature ; token embedding] (2 x hidden_dim) back to hidden_dim, followed by one decoder layer. Input is the target model's second-to-top-layer feature sequence plus the embedding of the token sequence *shifted one step ahead* (i.e. the actually sampled next token), which resolves sampling uncertainty. Output is the predicted next feature; the frozen target LM head turns it into a token distribution. [paper] — [EAGLE, Sec. 3.1](https://arxiv.org/html/2401.15077)
- Trainable parameter counts: 0.24B (7B target), 0.37B (13B), 0.56B (33B), 0.99B (70B); Mixtral 8x7B: 0.28B. [paper] — [EAGLE, Sec. 4](https://arxiv.org/html/2401.15077)
- Loss: L = L_reg + w_cls * L_cls, where L_reg is Smooth L1 between predicted and true next feature, L_cls is cross-entropy between the LM-head distributions computed from the true vs predicted feature, and w_cls = 0.1. [paper] — [EAGLE, Sec. 3.2](https://arxiv.org/html/2401.15077)
- Noise augmentation: uniform noise U(-0.1, 0.1) added to the target-LLM features fed as input during training, to mitigate error accumulation when the head later consumes its own predictions. [paper] — [EAGLE, Sec. 3.2](https://arxiv.org/html/2401.15077). The v1 training script confirms: `"p_w": 0.1, "v_w": 1.0, "data_noise": True, "noise": "uniform", "std": 0.2` (the uniform-noise class scales `(rand-0.5)*std`, i.e. +/-0.1 at seq len 512). [code] — [v1 eagle/train/main.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/v1/eagle/train/main.py)
- Base model is frozen; only the head is trained. Draft structure is a *static* tree; the draft phase takes 5 forward passes of the head; root has k=4 children (top-k for greedy, k samples for non-greedy). The total node count of the default static tree is not stated in the paper text. [paper] — [EAGLE, Sec. 3.1/4](https://arxiv.org/html/2401.15077)
- Input ablation (Fig. 8, Vicuna 7B) says feeding the shifted token alongside the feature gives the largest gain of the design choices; numeric values are only in the figure. [paper] — [EAGLE, Sec. 4.3](https://arxiv.org/html/2401.15077)
- Tree vs chain draft (Table 5, MT-bench, T=0), tau chain -> tree: Vicuna 7B 3.20 -> 3.94; 13B 3.23 -> 3.98; 33B 2.97 -> 3.68; LLaMA2-Chat 7B 3.00 -> 3.62; 13B 3.18 -> 3.90; 70B 3.12 -> 3.81. So the tree is worth ~+0.6 to +0.75 tokens per cycle. [paper] — [EAGLE, Table 5](https://arxiv.org/html/2401.15077)

**EAGLE-2 (arXiv 2406.16858, EMNLP 2024)**
- No retraining: "EAGLE-2 does not require training any extra models"; it reuses EAGLE-1's head unchanged. The change is the draft tree: instead of a fixed shape, a context-aware dynamic tree built from the draft model's confidence, which is well calibrated (Fig. 6, Vicuna 7B: draft confidence < 0.05 -> ~0.04 acceptance rate; > 0.95 -> ~0.98). [paper] — [EAGLE-2, Sec. 1, 3, 4](https://arxiv.org/html/2406.16858)
- Expansion phase: at each depth, expand the top-k nodes by "value" = product of confidences along the path from root. Rerank phase: pick the top-m nodes by value across the whole tree, flatten them with an ancestor-only attention mask, and verify in one target forward pass. [paper] — [EAGLE-2, Sec. 4.1-4.2](https://arxiv.org/html/2406.16858)
- Hyperparameters (Appendix A): total draft tokens m = 60 for 7B/8B targets, 50 for 13B, 48 for 70B; draft tree depth 6; top-k = 10 nodes expanded per level. [paper] — [EAGLE-2, App. A](https://arxiv.org/html/2406.16858)
- Ablation (Table 3, Vicuna 7B, T=0, MT-bench / GSM8K): no value + no rerank 2.81x/3.92 and 2.85x/3.93; w/o value 3.21x/4.39 and 2.93x/3.96; w/o rerank 3.48x/4.86 and 3.50x/4.85; full EAGLE-2 3.62x/4.98 and 3.63x/4.97. [paper] — [EAGLE-2, Table 3](https://arxiv.org/html/2406.16858)

**EAGLE-3 (arXiv 2503.01840, NeurIPS 2025)**
- Change 1: feature prediction is abandoned in favour of direct token prediction; the feature loss l_fea is removed, so the head's output no longer has to match the target's top-layer feature. Change 2: instead of only the top-layer feature, low-, mid- and high-level target features (each k-dim) are concatenated (3k) and reduced to k by an FC layer ("fused feature" g). Change 3: "training-time test" (TTT) — during training the head is unrolled for several steps with its own outputs a fed back as inputs, with attention masks adjusted, simulating multi-step drafting. Still a single decoder layer; still uses EAGLE-2's dynamic tree. [paper] — [EAGLE-3, Sec. 1, 3.1, 3.2](https://arxiv.org/html/2503.01840)
- The official training code makes the recipe concrete [code] — [traineagle3/cnets.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/cnets.py), [traineagle3/main.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/main.py), [traineagle3/config.json](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/config.json):
  - `self.length = 7` (7 TTT unroll steps per batch).
  - `self.fc = nn.Linear(hidden_size*3, hidden_size, bias=False)` (the 3-layer feature fusion); one decoder layer `midlayer` whose attention q/k/v projections take `hidden_size*2` inputs (feature and token embedding concatenated inside the layer).
  - Target model is loaded inside the training module, `.eval()`, all params `requires_grad=False`; the token embedding is copied from the target and frozen.
  - The head has its OWN trainable LM head over a reduced draft vocabulary: `self.lm_head = nn.Linear(hidden_size, draft_vocab_size)` with `"draft_vocab_size": 32000` in the Llama-3.1-8B config (target vocab 128256). The 32k tokens are chosen by scanning token frequency in the training data (`scandata`), with t2d/d2t maps.
  - Loss per step = soft cross-entropy between the target's next-token distribution (restricted to the draft vocab, softmaxed) and the head's log-softmax: `plogp = target_p * out_logp; loss = -sum(position_mask * plogp)`. Positions where the target's argmax token falls outside the draft vocab are masked out. No feature/regression term.
  - Per-step losses are combined with geometric weights: `ploss_weight = [0.8 ** i ...]` (step i weighted 0.8^i).
- Ablation (LLaMA-3.1 8B, MT-bench / GSM8K): EAGLE-2 3.16x/4.05 and 3.39x/4.24; + remove feature constraint (with TTT) 3.82x/5.37 and 3.77x/5.22; + fused features (= EAGLE-3) 4.40x/6.13 and 4.48x/6.23. [paper] — [EAGLE-3, ablation table](https://arxiv.org/html/2503.01840)
- With the higher acceptance rate the authors increase draft tree depth from 6 to 8 at inference. [paper] — [EAGLE-3 PDF, Appendix "Implementation details"](https://arxiv.org/pdf/2503.01840v3). A maintainer confirmed in an issue that the repo default depth 5 is meant for EAGLE-2 and EAGLE-3 needs `--depth 7` or `8`. — [EAGLE issue #209](https://github.com/SafeAILab/EAGLE/issues/209)
- Acceptance rate by number of self-predicted inputs (Fig. 7, LLaMA-3.1 8B, MT-bench): EAGLE's rate "drops significantly" as more self-predicted inputs are used, EAGLE-3's "remains almost unchanged". Only shown as a figure, no numbers in text. [paper] — [EAGLE-3, Sec. 4.2](https://arxiv.org/html/2503.01840)

**Parameter counts for Qwen3-8B EAGLE-3 heads (community checkpoints)**
- WhiFlash/Qwen3-8B-EAGLE3 (Samsung AI Center Cambridge): 1 decoder layer, hidden 4096, 32 heads / 8 KV heads, target vocab 151,936, draft vocab 32,000; safetensors metadata reports 0.4B params. [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3)
- RedHatAI/Qwen3-8B-speculator.eagle3: safetensors metadata reports 1B params. [card] — [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)
- Rule of thumb from a production vendor: single decoder layer, ~1-5% of target parameters. — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/); SpecForge paper says drafts are ~3-5% of target size — [SpecForge, arXiv 2603.18567](https://arxiv.org/html/2603.18567)

### Inferences
- The 0.4B vs 1B discrepancy between the two Qwen3-8B checkpoints is most plausibly a packaging difference: a full copy of the frozen 151,936 x 4096 embedding table is ~0.62B parameters; decoder layer + 3k->k FC + 32k-vocab LM head is roughly 0.2B + 0.05B + 0.13B ≈ 0.4B. So the *trainable* part for Qwen3-8B is ≈ 0.4B, and whether the checkpoint also ships the frozen embedding accounts for the rest. (Arithmetic is mine; neither card states this.)
- EAGLE-3 is not "one layer taking the last hidden state": it needs three intermediate hidden states (low/mid/high) from the base model at every verify step. Anyone specifying a retrofit as "last hidden state + embedding" is describing EAGLE-1/2's input; EAGLE-3's ablation attributes 3.82x -> 4.40x (MT-bench) to the fused multi-layer input.
- EAGLE-3's reduced 32k draft vocabulary is a real design constraint for multilingual use: tokens outside the 32k set can never be drafted (see caveats section).

### Gaps
- EAGLE-3 paper gives no explicit loss formula or per-step weights; the 0.8^i weights and TTT length 7 come from the code only. The paper (not the code) does not state a parameter count for the EAGLE-3 heads.
- Exact layer indices for "low/mid/high" features were not extracted from the official code in this pass.
- The node count of EAGLE-1's static tree was not found in the paper text read.

---

## 2. Training data: datasets, size, epochs, generated vs fixed, EAGLE-3 data scaling

### Takeaway
EAGLE-1/2 train on a fixed ShareGPT set of ~68k dialogues for 20 epochs (max length 2048); regenerating the answers with the target model helped only slightly (2.78x -> 2.88x). EAGLE-3 uses ShareGPT (~68k) + UltraChat-200K (~464k entries), i.e. "approximately 8x more data", with responses regenerated by the target model, and is the first version whose speedup keeps rising with data; by 2026 practitioners use 200k-1.4M target-regenerated conversations (~1B tokens for a Qwen3-8B head).

### Cited Findings

**EAGLE-1 / EAGLE-2**
- Dataset: ShareGPT, 68,000 dialogue iterations (intro: "no more than 70k dialogues"), fixed text (not generated by the target LLM). LR 3e-5, AdamW betas (0.9, 0.95), gradient clipping 0.5. [paper] — [EAGLE, Sec. 4](https://arxiv.org/html/2401.15077)
- Epochs and batch size are not in the paper; the official v1 script has `"num_epochs": 20`, `--bs` default 4, `"max_len": 2048`, `"num_warmup_steps": 2000`, `"total_steps": 800000`, and holds out the last 5% of the data as test. [code] — [v1 eagle/train/main.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/v1/eagle/train/main.py)
- Fixed vs target-generated data (Table 6, LLaMA2-Chat 7B, MT-bench, T=0): fixed dataset 2.78x / tau 3.62; data generated by the target LLM 2.88x / tau 3.75. [paper] — [EAGLE, Table 6](https://arxiv.org/html/2401.15077)
- v1 pipeline is two-stage: `python -m eagle.ge_data.allocation --outdir [path]` first runs the base model over the data and writes hidden states to disk; then `accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir [data] --cpdir [ckpt]`. [code] — [v1 README](https://raw.githubusercontent.com/SafeAILab/EAGLE/v1/README.md)
- EAGLE-2 trains nothing new; its LLaMA3-Instruct heads were EAGLE-1-style heads trained on ShareGPT. Qwen2 heads were trained on ShareGPT "which has removed non-English data"; Qwen2 targets need bf16 rather than fp16 "to avoid numerical overflow". — [EAGLE-2, Sec. 5](https://arxiv.org/html/2406.16858); [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
- CORAL (which follows the EAGLE-1 data recipe) states it explicitly: ShareGPT, 68,000 examples, 20 epochs, batch size 2 per GPU. [paper] — [CORAL, arXiv 2502.16880](https://arxiv.org/html/2502.16880v1)

**EAGLE-3**
- Data: ShareGPT (~68K entries) and UltraChat-200K (~464K entries); "We call the target model to generate responses rather than using a fixed dataset." For DeepSeek-R1-Distill-LLaMA 8B, OpenThoughts-114k-math was added. LR 5e-5, AdamW (0.9, 0.95), grad clip 0.5. [paper] — [EAGLE-3, Sec. 4 "Implementation"](https://arxiv.org/html/2503.01840)
- Scale: EAGLE-3 is "trained with approximately 8x more data than EAGLE". [paper] — [EAGLE-3, Sec. 1](https://arxiv.org/html/2503.01840)
- Scaling curve (Fig. 1, MT-bench, LLaMA-Instruct 3.1 8B, x-axis = data scale relative to ShareGPT at 1, 2, 4, 8): the EAGLE-3 curve is increasing while EAGLE-2's is flat; the paper says an increasing curve "was never observed in the previous works". The plot's y-axes span speedup 3.6-4.4 and acceptance length 4.0-6.0; exact per-point values are not printed. The 8x endpoint must correspond to the Table-1 EAGLE-3 result for this model (MT-bench 4.40x / 6.13). [paper] — [EAGLE-3 PDF, Fig. 1](https://arxiv.org/pdf/2503.01840v3)
- Official training config: `"num_epochs": 40`, `"max_len": 2048` (longer samples are filtered out), micro-batch 1 per GPU with `gradient_accumulation_steps: 2`, WarmupDecayLR with `warmup_max_lr 5e-5`, `warmup_num_steps 12000`, `total_num_steps 800000`, DeepSpeed ZeRO-2, fp16, weight decay 0. Hidden states are computed online by the co-located frozen target (no disk cache step in v3). [code] — [traineagle3/main.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/main.py), [traineagle3/ds_config.json](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/ds_config.json)

**Community / industrial EAGLE-3 data recipes (2025-2026), incl. Qwen3-8B**
- WhiFlash/Qwen3-8B-EAGLE3: UltraChat-200k, ~464K examples, assistant responses generated by Qwen3-8B; 800K training steps, batch size 8, LR 5e-5, warmup 1.5%, betas 0.9/0.95, weight decay 0, TTT length 7; "followed the EAGLE-3 training protocol as closely as possible". [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3)
- Tengyunw/qwen3_8b_eagle3: UltraChat-200k prompts only, responses synthesized by the target; "about 600K samples and 1B tokens". [card] — [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3)
- RedHatAI/Qwen3-8B-speculator.eagle3: trained with the `speculators` library on Aeala/ShareGPT_Vicuna_unfiltered + HuggingFaceH4/ultrachat_200k, thinking mode disabled. [card] — [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)
- Baseten guidance: generic heads need 200k-300k samples for small models (~20B or less) and ~500k for large models; specialized heads (JSON / structured output) ~100k; target 1k-2k total tokens per sample; outputs MUST be regenerated by the target model. TTT length 7-9; LR 1e-4 for ~3-7B targets, 5e-5 for ~7-20B, 2e-5 for 20B+; per-step training accuracy should plateau at 70-80%. — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/)
- SpecForge / SpecBundle (arXiv 2603.18567): trained on Open-PerfectBlend, ~1.4M conversations, responses regenerated by the target at temperature 0.8; 2 epochs, LR 1e-4, cosine annealing, sequence length 4096. Moving from ShareGPT (120K) + UltraChat (200K) to PerfectBlend (1.4M) raised Math500 acceptance length from 2.82 to 3.48 for Llama-3.1-8B. Regenerating responses raised acceptance length on nearly all benchmarks (FinanceQA the exception), ~5.3% average throughput gain. [paper] — [SpecForge](https://arxiv.org/html/2603.18567)
- SpecForge TTT-length sweep (1 to 17; official is 7): MT-Bench peaks around TTT≈3, while Math500, GSM8K and HumanEval peak around 13. Dense draft heads beat all MoE draft variants tested. [paper] — [SpecForge](https://arxiv.org/html/2603.18567)
- SpecForge v0.3 blog: "target-generated training responses were the largest lever on final acceptance" and are recommended for every strategy. — [LMSYS blog, 2026-08-04](https://lmsys.org/blog/2026-08-04-specforge-v0-3)
- TorchSpec Kimi-K2.5 EAGLE-3 head: 600k samples, 6 billion tokens (long-context, up to 100k-token sequences). — [PyTorch blog, 2026-03-19](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- Under-training data point: a user trained a Qwen2.5-7B EAGLE-3 head with SpecForge on ~50,000 ShareGPT conversations for 5 epochs and got acceptance length 1.9 on GSM8K (~2 in SGLang); SpecForge suggested 200K or 500K samples. Another commenter's QwQ head peaked at 1.86. — [EAGLE issue #276](https://github.com/SafeAILab/EAGLE/issues/276)
- Sanity-check floor from the vLLM `speculators` tutorial (default target Qwen3-8B): 5K ShareGPT samples x 5 epochs gives acceptance length only ~1.5 (P-EAGLE: 1.53, per-position acceptance 40.84% / 10.84% / 1.58% / 0.15%; DFlash 1.47); the page calls these sanity checks, not production results. — [speculators training tutorial](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/train/)

### Inferences
- Token count for the original recipe is not reported, but Tengyunw's "600K samples ≈ 1B tokens" (~1.7k tokens/sample) and Baseten's "1k-2k tokens per sample" imply the full EAGLE-3 set (68k + 464k ≈ 532k entries, max_len 2048) is on the order of 0.5-1B tokens, and EAGLE-1's ShareGPT-68k is on the order of 0.07-0.14B tokens. These are estimates, not reported figures.
- WhiFlash's schedule (800K steps x batch 8 = 6.4M sample-visits over 464K examples) works out to ~14 epochs; the official config's 800K-step LR schedule with effective batch 2 x n_GPUs likewise implies roughly 10-25 passes over ~532k samples on 4-8 GPUs (the `num_epochs: 40` loop bound would exceed the 800K-step schedule, so the step budget is probably the binding one). SpecForge gets competitive heads with 2 epochs over 1.4M conversations. So a realistic budget is ~3-7M sample-visits, i.e. several billion token-positions of base-model forward passes, each also unrolled 7x through the one-layer head.
- The EAGLE-1 result that target-generated data gives only +0.13 tau conflicts in emphasis with 2026 practitioner claims that regeneration is the "largest lever"; a plausible reconciliation is that EAGLE-3-style heads (which match the target's token distribution directly and scale with data) benefit more than the feature-regression EAGLE-1 head, and that Vicuna/LLaMA2 were themselves close to ShareGPT's distribution.
- For Qwen3-8B specifically, a decision is needed on thinking mode: RedHat trained with thinking disabled; EAGLE-3's authors added a reasoning dataset (OpenThoughts-114k-math) for the R1-distill model. A head trained only on non-thinking outputs should be expected to do worse inside `<think>` spans (inference from the R1 precedent; not directly measured in any source found).

### Gaps
- No source reports total token counts for the ShareGPT-68k or EAGLE-3 training sets.
- EAGLE-3's Fig. 1 per-point values (1x, 2x, 4x) are not printed anywhere I could find; only axis ranges are recoverable.
- No source gives a data-scaling curve beyond 8x ShareGPT for an 8B target with acceptance length reported at each point (SpecForge reports a single 320K -> 1.4M comparison on Math500 only).

---

## 3. Compute: GPUs, wall-clock, hidden-state caching cost and storage

### Takeaway
EAGLE-1 heads were trained in "1-2 days" on RTX 3090-class nodes (7B/13B/33B) or 4x A100-40G (70B); the EAGLE-3 paper reports NO training cost. For EAGLE-3 on an ~8B model the only concrete public data points are "4x H100, 800K steps" (no hours given) and CORAL's "about 2 days for a 7B draft" on 4 GPUs for a 3-step unrolled EAGLE-style recipe; offline hidden-state caching is storage-prohibitive at EAGLE-3 scale (~80-270 MB per sample, i.e. multiple TB to >10 TB), so online/co-located generation is the norm.

### Cited Findings

**EAGLE-1**
- LLaMA2-Chat 70B head: trainable "within 1-2 days" on 4x A100 (40G); 7B/13B/33B heads: 1-2 days on an RTX 3090 node. [paper] — [EAGLE, Sec. 1 & 4](https://arxiv.org/html/2401.15077)
- README: "trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs. So even the GPU poor can afford it." [code] — [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
- FSPAD (EAGLE-2-style training, 68k ShareGPT): four A100 80G; the 70B draft trains "within two days". [paper] — [FSPAD, arXiv 2408.15562](https://arxiv.org/html/2408.15562v1)

**EAGLE-2**
- Zero training cost on top of EAGLE-1 (reuses the same head). [paper] — [EAGLE-2](https://arxiv.org/html/2406.16858)

**EAGLE-3**
- The paper states no GPU type, GPU count, or wall-clock for training. The 405B and 671B models were not tested "due to the GPU constraint". [paper] — [EAGLE-3](https://arxiv.org/html/2503.01840)
- Official script trains with DeepSpeed across GPUs (ZeRO-2, micro-batch 1/GPU, grad-accum 2, 800K scheduled steps) with the frozen fp16 target model loaded in the same process — so each training GPU must hold the full base model plus 7 unrolled steps of head activations. [code] — [traineagle3/ds_config.json](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/ds_config.json), [cnets.py](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/cnets.py)
- The authors now "strongly recommend using SpecForge for out-of-the-box training of EAGLE-3 with SGLang" (README update of 2025-07-23). — [EAGLE README](https://github.com/SafeAILab/EAGLE)
- WhiFlash/Qwen3-8B-EAGLE3: 4x NVIDIA H100 80GB, 800K steps at batch size 8; wall-clock/GPU-hours not stated. [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3)
- SpecForge paper experiments: eight NVIDIA H200, sequence length 4096; GPU-hours not reported. Its Table 1 gives training throughput: baseline (official-style ZeRO-3) 2,025.7 tokens/s vs SpecForge TP=8 20,227.2 tokens/s for Qwen3-235B-A22B (9.99x); speedups over baseline for smaller targets are 2.01x (Llama3.1-8B), 1.39x (Llama3.3-70B), 4.31x (Qwen3-30B-A3B). [paper] — [SpecForge](https://arxiv.org/html/2603.18567)
- TorchSpec: a Kimi K2.5 (~1T-param MoE) EAGLE-3 head took 1,500 H200 GPU-hours for 600k samples / 6B tokens (8x H200 inference + 8x H200 training). TTT "requires high memory usage as it retains intermediate activations for multiple speculative steps". — [PyTorch blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)

**Multi-step (HASS / CORAL) overhead on top of the EAGLE-1 recipe**
- HASS: Align-3 needs "66.34% more" training time than EAGLE-2-style training averaged over four models (max +91.47%, LLaMA3-Instruct 8B) and "approximately 3x computational cost of EAGLE-2"; fits on a single H800 at batch size 2 even at Align-5. Absolute hours not reported. [paper] — [HASS, App. A.8](https://arxiv.org/html/2408.15766)
- CORAL: ~2 days to train a 7B draft under default settings (68k ShareGPT, 20 epochs, batch 2/GPU, 3 steps) on 4x NVIDIA H20-96G; a Llama3-70B draft on A100-40G "may run out of memory". [paper] — [CORAL](https://arxiv.org/html/2502.16880v1)

**Hidden-state caching / storage**
- vLLM `speculators` docs: offline storage per sample = avg_seq_len x num_layers x hidden_size x dtype_bytes; Llama-3.1-8B example 8192 x 4 x 4096 x 2 bytes ≈ 268 MB/sample -> ~260 GB per 1K samples, 2.6 TB per 10K, 13 TB per 50K. Online mode needs GPUs for both a vLLM server and training but no disk cache (example layout: 4-GPU node, 2 GPUs serving, 2 training). — [speculators training tutorial](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/train/)
- NVIDIA NeMo docs: offline path ≈ 100+ MB per sample for an 8B target at seq length 2048 (an earlier version of the page says ~80 MB), "multiple TB" for a large corpus. [snippet] — [NeMo AutoModel precompute_eagle3](https://docs.nvidia.com/nemo/automodel/nightly/nemo-automodel/nemo_automodel/components/speculative/precompute_eagle3)
- AMD's SpecForge tutorial: offline training on UltraChat + ShareGPT would need about 12 TB of storage. [snippet] — [AMD ROCm SpecForge notebook](https://rocm.docs.amd.com/projects/ai-developer-hub/en/v14.0/notebooks/pretrain/SpecForge_SGlang.html)
- TorchSpec, extreme case: Kimi K2.5 (hidden 7168) at 131,072-token context = ~7.0 GB per sample (3 aux layers 5.25 GB + last hidden 1.75 GB, bf16); 10k samples = 70 TB. Target logits need not be stored (recomputable from last hidden state via lm_head). — [PyTorch blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- SpecForge v0.3 disaggregated training testbed for a Qwen3-8B draft (Domino, not EAGLE-3): 8x H20, 3 SGLang capture servers + 5 trainer workers, 3K-token context, 1.10x throughput vs co-located. — [LMSYS blog](https://lmsys.org/blog/2026-08-04-specforge-v0-3)

### Inferences
- Budget sketch for a Qwen3-8B EAGLE-3 head (my estimate, not a reported number): the forward cost is dominated by running the frozen 8B base over every training token once per visit; the head adds 7 one-layer passes (≈ 7/36 of a base forward for a 36-layer model, plus backward). With ~0.5-1B tokens of data and ~3-14 passes, that is ~2-14B token-positions. The two closest public analogues are (a) 4x H100 for an 800K-step, batch-8 run (WhiFlash, hours unknown) and (b) ~2 days on 4x H20 for a 3-step unrolled EAGLE-style head on 68k x 20 epochs (CORAL). A plausible planning range is therefore roughly 100-400 GPU-hours on H100-class cards for the full EAGLE-3 protocol on an 8B target, with a few-dozen-GPU-hour pilot possible at SpecForge-style 2 epochs. This range is an inference and should be validated with a throughput measurement before committing.
- Offline caching at full EAGLE-3 scale is impractical for an 8B model: using the speculators formula with 2048-token samples, 3 fused layers (+ last hidden for logits), hidden 4096, bf16 gives ~50-67 MB/sample, so ~532k samples ≈ 27-36 TB (consistent in order of magnitude with AMD's "~12 TB" and NVIDIA's "80-100+ MB/sample"). Co-located/online hidden-state generation (official script, SpecForge online mode, speculators online mode) is the practical choice; the price is that the 8B base (~16 GB in bf16) sits on every training GPU.
- For EAGLE-1-style training (single top-layer feature, 68k samples, ≤2048 tokens), the cache is 1 layer x 4096 x 2 bytes ≈ 8 KB/token, i.e. tens to low hundreds of GB — which is why v1's two-stage `ge_data` pipeline was feasible on a 3090 box. (Estimate; the paper/README give no storage figure.)

### Gaps
- No official wall-clock or GPU-hour number exists for training any EAGLE-3 head (paper, README, or model cards for 8B-class targets). Flagged as the single biggest budgeting gap.
- EAGLE-1 does not break "1-2 days" down by model size, nor report the `ge_data` hidden-state generation time or disk footprint.
- NVIDIA and AMD storage figures are from search snippets; I did not open those pages.

---

## 4. Results: speedup, acceptance length, acceptance rates

### Takeaway
On the authors' own hardware/harness, 7-8B targets get ~2.8-3.3x (EAGLE-1), ~3.2-3.6x (EAGLE-2) and ~4.4x (EAGLE-3) at T=0 with tau ≈ 3.9 / 4.1-5.0 / 6.1-6.2, falling to ~3.1x / tau 4.2 for EAGLE-3 at T=1 on MT-bench. Independent and production measurements are much lower: Spec-Bench gives EAGLE-2 2.2-2.4x on Vicuna-7B and EAGLE-3 3.0x on Vicuna-13B, and community Qwen3-8B EAGLE-3 heads report 1.7-2.4x at batch size 1 with acceptance lengths from ~2.0 (chain of 2 in vLLM) to 4.8 (60-token tree in SGLang).

### Cited Findings

**EAGLE-1 (static tree) — authors' numbers**
- Table 1, speedup / tau at T=0 (HumanEval, GSM8K, Alpaca): Vicuna 7B 3.33x/4.29, 3.01x/4.00, 2.79x/3.86; Vicuna 13B 3.58x/4.39, 3.08x/3.97, 3.03x/3.95; Vicuna 33B 3.67x/4.28, 3.25x/3.94, 2.97x/3.61; LLaMA2-Chat 7B 3.17x/4.24, 2.91x/3.82, 2.78x/3.71; 13B 3.76x/4.52, 3.20x/4.03, 3.01x/3.83; 70B 3.52x/4.42, 3.03x/3.93, 2.97x/3.77. [paper] — [EAGLE, Table 1](https://arxiv.org/html/2401.15077)
- Same at T=1: Vicuna 7B 2.39x/3.43, 2.34x/3.29, 2.21x/3.30; Vicuna 13B 2.65x/3.63, 2.57x/3.60, 2.45x/3.57; Vicuna 33B 2.76x/3.62, 2.77x/3.60, 2.52x/3.32; LLaMA2-Chat 7B 2.61x/3.79, 2.40x/3.52, 2.29x/3.33; 13B 2.89x/3.78, 2.82x/3.67, 2.66x/3.55; 70B 2.92x/3.76, 2.74x/3.58, 2.65x/3.47. [paper] — [EAGLE, Table 1](https://arxiv.org/html/2401.15077)
- Table 2, MT-bench tau and chain acceptance rates n-alpha (n = number of the head's own predicted features in the input; 0-alpha = all-true features), T=0: Vicuna 7B tau 3.94, 0.79/0.74/0.72/0.73/0.67; Vicuna 13B 3.98, 0.79/0.74/0.72/0.74/0.70; Vicuna 33B 3.68, 0.74/0.69/0.67/0.67/0.66; LLaMA2-Chat 7B 3.62, 0.76/0.69/0.67/0.68/0.68; 13B 3.90, 0.77/0.69/0.69/0.70/0.71; 70B 3.81, 0.75/0.69/0.65/0.64/0.64. T=1: Vicuna 7B 3.17, 0.71/0.68/0.66/0.66/0.65; 13B 3.20, 0.73/0.68/0.68/0.67/0.69; 33B 3.22, 0.71/0.67/0.64/0.64/0.64; LLaMA2-Chat 7B 3.30, 0.71/0.66/0.66/0.66/0.64; 13B 3.45, 0.73/0.69/0.66/0.67/0.67; 70B 3.46, 0.73/0.67/0.64/0.66/0.65. [paper] — [EAGLE, Table 2](https://arxiv.org/html/2401.15077)
- MT-bench speedups are given only in figures; text gives 2.1x-3.8x overall and 2.7x-3.5x for LLaMA2-Chat 70B. Mixtral 8x7B (Table 3, MT-bench, T=0): 1.50x, tau 3.25. [paper] — [EAGLE](https://arxiv.org/html/2401.15077)
- gpt-fast (Table 4, LLaMA2-Chat 7B, MT-bench, T=0, single RTX 3090), tokens/s fp16 / int4: HF vanilla 24.5 / N/A; gpt-fast 55.1 / 106.9; EAGLE + gpt-fast 100.2 / 160.4 (i.e. 1.82x at fp16 but only 1.50x on the int4-quantized model). [paper] — [EAGLE, Table 4](https://arxiv.org/html/2401.15077)

**EAGLE-2 — authors' numbers (Table 1; MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM, Natural Questions, mean)**
- T=0: Vicuna 13B 4.26x/4.83, 4.96x/5.41, 4.22x/4.79, 4.25x/4.89, 3.40x/4.21, 3.13x/3.74, mean 4.04x/4.65. LLaMA2-Chat 13B 4.21x/4.75, 5.00x/5.52, 4.31x/4.90, 4.13x/4.61, 3.45x/4.24, 3.51x/4.04, mean 4.10x/4.68. Vicuna 7B 3.62x/4.98, 3.95x/5.33, 3.63x/4.97, 3.46x/4.86, 2.94x/4.12, 2.76x/3.82, mean 3.39x/4.68. LLaMA2-Chat 7B 3.43x/4.70, 4.03x/5.39, 3.52x/4.77, 3.45x/4.66, 3.01x/4.12, 3.15x/4.19, mean 3.43x/4.64. [paper] — [EAGLE-2, Table 1](https://arxiv.org/html/2406.16858)
- T=1: Vicuna 13B mean 3.65x/4.26 (MT-bench 3.80x/4.40); LLaMA2-Chat 13B mean 3.88x/4.51 (3.92x/4.51); Vicuna 7B mean 2.94x/4.20 (3.05x/4.28); LLaMA2-Chat 7B mean 3.15x/4.41 (3.19x/4.41). [paper] — [EAGLE-2, Table 1](https://arxiv.org/html/2406.16858)
- Table 2 (MT-bench, T=0): LLaMA2-Chat 70B 3.51x/4.48; LLaMA3-Instruct 70B 3.29x/4.16; LLaMA3-Instruct 8B 3.46x/4.53. [paper] — [EAGLE-2, Table 2](https://arxiv.org/html/2406.16858)
- Per-position acceptance (Fig. 5) is only qualitative: highest at position 1, lowest at position 6, high variance at the same position (the motivation for dynamic trees). Evaluation hardware is not specified beyond "same devices". [paper] — [EAGLE-2](https://arxiv.org/html/2406.16858)

**EAGLE-3 — authors' numbers (Table 1; MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM, mean)**
- T=0, EAGLE-3: Vicuna 13B 5.58x/6.65, 6.47x/7.54, 5.32x/6.29, 5.16x/6.17, 5.01x/6.47, mean 5.51x/6.62. LLaMA-3.1 8B 4.40x/6.13, 4.85x/6.74, 4.48x/6.23, 4.82x/6.70, 3.65x/5.34, mean 4.44x/6.23. LLaMA-3.3 70B 4.11x/5.63, 4.79x/6.52, 4.34x/6.15, 4.30x/6.09, 3.27x/5.02, mean 4.12x/5.88. DeepSeek-R1-Distill-LLaMA 8B 4.05x/5.58, 4.59x/6.38, 5.01x/6.93, 3.65x/5.37, 3.52x/4.92, mean 4.16x/5.84. [paper] — [EAGLE-3, Table 1](https://arxiv.org/html/2503.01840)
- T=0, EAGLE-2 baselines in the same table: Vicuna 13B mean 4.22x/4.83; LLaMA-3.1 8B 3.16x/4.05 (MT-bench), mean 3.23x/4.11; LLaMA-3.3 70B mean 2.85x/3.78; DeepSeek-R1-Distill 8B mean 3.26x/3.92. [paper] — [EAGLE-3, Table 1](https://arxiv.org/html/2503.01840)
- T=1, EAGLE-3: Vicuna 13B 4.57x/5.42, 5.15x/6.22, 4.71x/5.58, 4.49x/5.39, 4.33x/5.72, mean 4.65x/5.67. LLaMA-3.1 8B 3.07x/4.24, 4.13x/5.82, 3.32x/4.59, 3.90x/5.56, 2.99x/4.39, mean 3.45x/4.92. LLaMA-3.3 70B mean 3.95x/5.66. DeepSeek-R1-Distill 8B 3.20x/4.49, 3.77x/5.28, 4.38x/6.10, 3.16x/4.30, 3.08x/4.27, mean 3.52x/4.89. EAGLE-2 at T=1: LLaMA-3.1 8B mean 2.80x/3.62; R1-Distill 8B mean 2.77x/3.54. [paper] — [EAGLE-3, Table 1](https://arxiv.org/html/2503.01840)
- Headline: "approximately 3.0x-6.5x", "20%-40% improvement over EAGLE-2"; README: 5.6x vs vanilla and 1.8x vs EAGLE-1 on 13B. Hardware for Table 1 is not stated in the paper. — [EAGLE-3](https://arxiv.org/html/2503.01840); [EAGLE README](https://github.com/SafeAILab/EAGLE)
- In a production engine at batch size 1 (Table 4; SGLang, 1x H100, LLaMA-3.1 8B, MT-bench): no speculation 158.34 tok/s; EAGLE-2 244.10 tok/s (1.54x); EAGLE-3 373.25 tok/s (2.36x). [paper] — [EAGLE-3, Table 4](https://arxiv.org/html/2503.01840)

**Independent: Spec-Bench leaderboard (greedy, fp16, batch size 1, mean of 3 runs; tasks MT-bench / translation / summarization / QA / math / RAG; then mean accepted tokens; overall)**
- RTX 3090, Vicuna-7B: EAGLE-2 2.56x/1.78x/2.09x/2.07x/2.66x/1.86x, 4.35 tokens, overall 2.19x; EAGLE 2.31x/1.72x/2.00x/1.91x/2.38x/1.75x, 3.57, 2.03x; Hydra 1.91x; SpS 1.79x; Medusa 1.44x. — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- A100, Vicuna-7B: EAGLE-2 2.84x/1.88x/2.34x/2.15x/2.79x/2.13x, 4.34, overall 2.36x; EAGLE 2.45x/1.77x/2.08x/1.93x/2.44x/1.87x, 3.58, 2.10x. — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- A100, Vicuna-13B: EAGLE-3 3.48x/2.36x/3.14x/2.94x/3.42x/2.78x, 5.71 tokens, overall 3.02x; EAGLE-2 2.95x/1.96x/2.43x/2.20x/2.95x/2.25x, 4.43, 2.46x; EAGLE 2.16x overall (3.64 tokens). — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- A100, Vicuna-33B: EAGLE-2 overall 2.59x (4.05 tokens); EAGLE 2.43x (3.39 tokens). — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)

**EAGLE-2 as re-measured by follow-up papers (lower than the EAGLE papers)**
- HASS (H800, bs 1; mean over MT-bench/HumanEval/GSM8K; T=0 / T=1): LLaMA2-Chat 7B EAGLE 2.01x / 1.76x, EAGLE-2 2.81x / 2.60x; LLaMA3-Instruct 8B EAGLE 1.71x / 1.44x, EAGLE-2 2.83x / 2.47x (tau 4.52 / 4.31); LLaMA3-Instruct 70B EAGLE-2 3.37x / 3.28x. [paper] — [HASS, Tables 1-2](https://arxiv.org/html/2408.15766)
- CORAL (single A6000, bs 1): Llama3-8B EAGLE-2 MT-bench 2.28x/tau 4.16, HumanEval 2.61x/4.78, GSM8K 2.32x/4.21 at T=0; Qwen2.5-7B EAGLE-2 2.13x/3.91, 2.53x/4.62, 2.33x/4.23 at T=0 and 1.86x/3.45, 2.35x/4.36, 2.21x/4.07 at T=1. [paper] — [CORAL, main table](https://arxiv.org/html/2502.16880v1)
- FSPAD (A100 80G): EAGLE-2 on LLaMA3-8B MT-bench 3.36x/tau 4.21 at T=0, 2.72x/3.81 at T=1; translation (WMT14) 2.97x/3.87. [paper] — [FSPAD, Table 1](https://arxiv.org/html/2408.15562v1)

**Qwen results (closest to the Qwen3-8B target)**
- WhiFlash/Qwen3-8B-EAGLE3 on Spec-Bench (480 examples, single H100, SGLang, 8 decoding steps, top-k 10, 60 draft tokens, mean of 5 seeds), acceptance length / tok/s: conversation 5.11 / 258.9; translation 3.60 / 159.1; summarization 4.21 / 203.9; QA 4.80 / 244.1; math 5.94 / 297.8; RAG 5.04 / 246.9; mean 4.78 / 235.1 = 2.42x. Temperature and thinking mode not stated. [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3)
- Tengyunw/qwen3_8b_eagle3 (SGLang, 6 steps, top-k 10, 32 draft tokens, bf16): H200 187 -> 365 tok/s (~1.95x); RTX 5090 90 -> 220 tok/s (~2.4x). Acceptance length, temperature, prompts not reported. [card] — [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3)
- AngelSlim/Qwen3-8B_eagle3 (vLLM v0.11.2, single H20, num_speculative_tokens=2 chain, bs 1, output_len 1024), vanilla -> EAGLE-3 tok/s (accept length): GSM8K 149.63 -> 257.32 (2.00); Alpaca 149.93 -> 266.69 (2.02); HumanEval 153.85 -> 244.89 (1.97); MT-bench 153.81 -> 258.2 (1.97); mean 151.81 -> 257.52 (1.99) ≈ 1.70x. Other sizes (mean): Qwen3-1.7B 1.69x (AL 2.17), 4B 1.66x (2.08), 14B 1.56x (1.79), 32B 1.71x (1.91), 30B-A3B 1.37x (2.04). Card's general claim: accept length 1.8-3.5, max speedup 1.4-1.9x. [card; ratios computed from the card's table] — [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)
- RedHatAI/Qwen3-8B-speculator.eagle3 (vLLM 0.11.0, 1x A100, temperature 0.6, top_p 0.95, top_k 20; chain draft of k tokens), mean acceptance length for k = 1..7: coding (HumanEval) 1.72, 2.17, 2.39, 2.59, 2.60, 2.59, 2.69; math (GSM8K) 1.73, 2.20, 2.48, 2.63, 2.72, 2.79, 2.81; summarization (CNN/DM) 1.62, 1.96, 2.13, 2.24, 2.25, 2.29, 2.30. Latency/speedup results are only in images. [card] — [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)
- SpecBundle EAGLE-3 heads (SGLang, H200; speedup vs no speculation; evaluation temperature not stated): Qwen3-30B-A3B MTBench 1.55x, GPQA 1.66x, FinanceQA 1.35x, LiveCodeBench 2.29x, HumanEval 2.25x, GSM8K 1.40x, Math500 1.48x. Qwen3-235B-A22B: 1.54x, 1.47x, 1.65x, 1.93x, 2.29x, 1.62x, 2.38x (vs 1.21x-1.61x for the pre-existing public draft). Max across all models 4.48x (Llama-4-Scout, LiveCodeBench). [paper] — [SpecForge, Table 2](https://arxiv.org/html/2603.18567)
- Baseten production experience (Qwen3-4B target): 1.5-2.5x latency improvement attributable to the EAGLE head; "EAGLE papers report up to 4-6x", part of which the post attributes to serving-framework differences. — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/)
- Official repo status: all Qwen3 EAGLE-3 checkpoints in the README are marked non-official (Qwen3-8B: Tengyunw/qwen3_8b_eagle3, AngelSlim/Qwen3-8B_eagle3, Zjcxy-SmartAI/Eagle3-Qwen3-8B-zh); official EAGLE-3 Qwen3 support is an unchecked TODO; "unofficial checkpoints may perform differently". Official EAGLE-1/2 heads exist for Qwen2-7B-Instruct and Qwen2-72B-Instruct. — [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)

### Inferences
- From RedHat's k=1 numbers, the first-draft-token acceptance rate of a production Qwen3-8B EAGLE-3 head at T=0.6 with top-p/top-k is ≈ 0.62-0.73 (AL(k=1) - 1), and the marginal gain per extra chain position decays fast (≈ +0.45, +0.25, +0.17, +0.05 ...), saturating at tau ≈ 2.3-2.8 for a chain. The paper-style tau of 5-6 requires the 60-token dynamic tree, greedy decoding, and depth 7-8.
- EAGLE-1's n-alpha table implies per-step chain acceptance of ~0.65-0.79 at T=0 with only mild decay over 5 positions; EAGLE-3's contribution (Fig. 7) is to flatten this decay, not mainly to raise the first-token rate.
- Reported speedup depends heavily on the baseline: authors' harness (HF-transformers-style vanilla decoding, bs 1) gives EAGLE-3 4.40x on Llama-3.1-8B MT-bench, while the same paper's SGLang/H100 measurement gives 2.36x for the same model/benchmark. For budgeting a Qwen3-8B retrofit served through vLLM/SGLang, ~1.7-2.4x at batch size 1 is the evidence-based expectation, not 4-5x.
- The ratio speedup/tau is a rough efficiency gauge: EAGLE-3 on Llama-3.1 8B gets 4.40/6.13 ≈ 0.72 in the authors' harness, but WhiFlash's Qwen3-8B gets 2.42/4.78 ≈ 0.51 in SGLang on H100 — draft overhead and a faster baseline eat about half the acceptance length.

### Gaps
- No peer-reviewed or neutral third-party benchmark of an EAGLE-3 head on Qwen3-8B exists that I could find; all Qwen3-8B numbers are self-reported model cards with incomplete methodology (temperature and thinking mode often unstated).
- EAGLE-3 per-position acceptance rates (Fig. 7) and EAGLE-2 per-position rates (Fig. 5) are figure-only.
- Hardware for the main Table 1 in EAGLE-2 and EAGLE-3 is not stated (a maintainer mentions reproducing 4.00x-4.20x for Llama-3.1-8B on an "A800" in issue #209).
- Spec-Bench has no EAGLE-3 entries other than Vicuna-13B on A100, and no temperature > 0 leaderboard.

---

## 5. Follow-up papers that change the training recipe

### Takeaway
HASS, FSPAD and CORAL all patch EAGLE-1/2's train/inference mismatch and each buys roughly +0.3 to +0.6 tau and +3-20% speed over EAGLE-2 at ~1.7-3x the training compute; EAGLE-3 subsumes their central idea (multi-step unrolled training) and adds data scaling, so none of them is the better starting point in 2026. Post-EAGLE-3 work mostly targets the draft LM-head/vocabulary cost, training infrastructure (SpecForge, TorchSpec, speculators), or parallel drafting, with modest gains (≈8-30%) over EAGLE-3.

### Cited Findings

**HASS (arXiv 2408.15766, ICLR 2025)**
- Changes: (1) Harmonized objective distillation — adds a Top-K loss L_TopK = -Σ_{x in top-K of target} q(x) log p(x), K = 10, weight w = 1.0, to EAGLE's loss (feature regression retained); (2) Harmonized context alignment — n-step training in which step j feeds the head's own features from earlier passes back in (default Align-3). No inference overhead; same head architecture. [paper] — [HASS, Sec. 3, 4.1](https://arxiv.org/html/2408.15766)
- Data: fixed ShareGPT 68k. Cost: +66.34% training time on average (max +91.47%), ~3x compute vs EAGLE-2. Hardware: H800, bs 1 for inference. [paper] — [HASS, App. A.8](https://arxiv.org/html/2408.15766)
- Results (mean over MT-bench/HumanEval/GSM8K; tau and speedup; T=0 | T=1): LLaMA2-Chat 7B EAGLE-2 4.61, 2.81x | 4.40, 2.60x vs HASS 5.15, 3.24x | 4.92, 2.90x. LLaMA2-Chat 13B 5.16, 3.30x | 5.01, 3.21x vs 5.58, 3.65x | 5.42, 3.48x. LLaMA3-Instruct 8B 4.52, 2.83x | 4.31, 2.47x vs 5.08, 3.09x | 4.80, 2.81x. LLaMA3-Instruct 70B 4.50, 3.37x | 4.43, 3.28x vs 5.21, 4.05x | 5.16, 3.85x. Headline: 2.81x-4.05x, 8%-20% over EAGLE-2. [paper] — [HASS, Tables 1-2](https://arxiv.org/html/2408.15766)
- Ablations (LLaMA2-Chat 7B, mean tau T=0 / T=1): EAGLE-2 + Top-K only 4.78 / 4.57; Align-2 5.11 / 4.86; Align-3 5.15 / 4.92; Align-4 5.16 / 4.91; Align-5 5.11 / 4.86 (saturates at 3-4, attributed to limited head capacity). K sweep: K=1 5.08, K=5 5.16, K=10 5.15, K=100 5.16 (insensitive). Top-K weight: w=0 5.09, 0.1 5.13, 0.5 5.17, 1.0 5.15, 2.0 5.15. Training on model-generated data helps 7B but is mixed on 13B. HASS's first-step acceptance rate is slightly lower than EAGLE-2's on two models, higher at later steps. [paper] — [HASS, Tables 4, 7, App. A.4, Fig. 5-6](https://arxiv.org/html/2408.15766)

**FSPAD (arXiv 2408.15562)**
- Changes: (1) Feature Sampler replaces EAGLE's FC connector: Up and Gate projectors map features and embeddings into an intermediate space, SiLU on the embedding branch, product with the feature branch, Down projector back to hidden size; (2) Partial Alignment Distillation: the decoder layer's MLP output is doubled to emit two vectors — one mapped to logits, one passed on as the next-step feature — which "weakens the connection between features and logits". Loss L = w*L_t + L_f with w = 0.1 (Smooth L1 features + CE tokens). [paper] — [FSPAD, Sec. 3](https://arxiv.org/html/2408.15562v1)
- Bigger head: 0.42B (Vicuna 7B, LLaMA3-8B), 1.09B (Vicuna 33B), 2.05B (LLaMA3-70B) — vs EAGLE's 0.24B / 0.56B / 0.99B (the intro's "0.18B to 1.10B" conflicts with the 2.05B figure). Training: ShareGPT 68k fixed, LR 5e-5, 4x A100 80G, 70B head "within two days". [paper] — [FSPAD, Sec. 4](https://arxiv.org/html/2408.15562v1)
- Results vs EAGLE-2 (tau / speedup, A100 80G). T=0: Vicuna 7B MT-bench 4.74/3.56x -> 5.12/3.71x; GSM8K 5.03/3.69x -> 5.32/3.82x; translation 3.92/2.55x -> 4.05/2.58x. LLaMA3-8B MT-bench 4.21/3.36x -> 4.55/3.43x; GSM8K 4.52/3.51x -> 4.82/3.57x. LLaMA3-70B MT-bench 4.11/3.09x -> 4.36/3.14x. T=1: Vicuna 7B MT-bench 4.20/3.15x -> 4.51/3.26x; LLaMA3-8B MT-bench 3.81/2.72x -> 4.14/2.90x. So tau +0.3-0.5 but speedup only +1-7% (the larger head costs more per draft step). Ablation (Vicuna 7B, MT-bench): w/o FS 4.88/3.65x; w/o PAD 5.10/3.69x; full 5.12/3.71x. [paper] — [FSPAD, Tables 1, 3](https://arxiv.org/html/2408.15562v1)

**CORAL (arXiv 2502.16880, ACL 2025)**
- Changes: (1) Cross-Step Representation Alignment (CSRA): on top of HASS-style 3-step training, an InfoNCE contrastive loss (temperature 0.07) treating the head's output features at the same position across training steps as positives; total L = w_reg*L_reg + w_cls*L_cls + w_CSRA*L_CSRA with w_reg = 0.5 and w_CSRA = 0.05-0.2 depending on the model (0.2 Qwen2.5-7B, 0.15 Llama3-8B, 0.1 Llama3-70B/Qwen2.5-14B/Llama2-7B, 0.05 Llama2-13B). (2) LM-head router: split the LM head into N groups and train (second stage, head frozen) a small router to activate only the top groups; best on Llama3-8B is N=16, top-2. [paper] — [CORAL, Sec. 3](https://arxiv.org/html/2502.16880v1)
- Latency motivation: Llama3-8B draft step 2.58 ms vs target 26.06 ms (9.9%); Qwen2.5-7B 2.69 ms vs 24.58 ms (10.9%); Llama2-7B 1.36 ms vs 23.65 ms (5.8%). On Llama3-8B the draft's LM head takes 1.51 ms and the rest of the draft 1.07 ms — the LM head over a 128k-152k vocabulary is ~59% of draft latency. [paper] — [CORAL, Table 1, App. D](https://arxiv.org/html/2502.16880v1)
- Training: ShareGPT 68k, 20 epochs, batch 2/GPU, ~2 days for a 7B draft on 4x H20-96G. Inference: single A6000 48G, bs 1. [paper] — [CORAL, Sec. 4](https://arxiv.org/html/2502.16880v1)
- Results, Llama3-8B (tau / speedup; EAGLE-2 | HASS | CORAL | CORAL + router): MT-bench T=0 4.16/2.28x | 4.48/2.45x | 4.57/2.50x | 4.26/2.63x; HumanEval T=0 4.78/2.61x | 5.31/2.89x | 5.43/2.95x | 5.22/3.21x; GSM8K T=0 4.21/2.32x | 4.56/2.51x | 4.70/2.58x | 4.42/2.76x; MT-bench T=1 3.84/2.08x | 4.12/2.21x | 4.15/2.24x | 3.92/2.39x. [paper] — [CORAL, main table](https://arxiv.org/html/2502.16880v1)
- Results, Qwen2.5-7B: MT-bench T=0 3.91/2.13x | 4.15/2.26x | 4.22/2.30x | 4.02/2.50x; HumanEval T=0 4.62/2.53x | 4.96/2.71x | 5.09/2.78x | 4.86/3.05x; GSM8K T=0 4.23/2.33x | 4.53/2.49x | 4.67/2.57x | 4.38/2.76x; MT-bench T=1 3.45/1.86x | 3.65/1.96x | 3.83/2.05x | 3.62/2.21x; HumanEval T=1 4.36/2.35x | 4.74/2.55x | 4.86/2.62x | 4.57/2.81x; GSM8K T=1 4.07/2.21x | 4.35/2.35x | 4.50/2.44x | 4.16/2.58x. Average speedup T=0, EAGLE-2 / HASS / CORAL: Llama2-7B 3.07x / 3.31x / 3.32x; Llama2-13B 3.43x / 3.68x / 3.72x; Llama3-70B 2.90x / 2.99x / 3.08x; Qwen2.5-14B 2.65x / 2.90x / 2.96x (router: 3.11x). Headline 2.50x-4.07x. [paper] — [CORAL](https://arxiv.org/html/2502.16880v1)
- CSRA applied to single-step training hurts (Llama3-8B tau 4.16/4.78/4.21 -> 4.10/4.70/4.10). [paper] — [CORAL, ablation](https://arxiv.org/html/2502.16880v1)

**Post-EAGLE-3 ecosystem (2025-2026)**
- SpecForge (arXiv 2603.18567, March 2026; SGLang team): training framework with full EAGLE-3 support (online/offline, tensor-parallel targets), plus SpecBundle public drafts; SpecBundle beats pre-existing public drafts by up to 1.35x. v0.3.0 (August 2026) adds disaggregated capture/training and non-EAGLE drafters (DFlash, Domino, DSpark, P-EAGLE, MTP); of its 11 SpecBundle checkpoints 7 are EAGLE-3 (GLM-5.1, Kimi-K2.5/2.6/2.7-Code, Qwen3-32B, Qwen3.5-35B-A3B, Step-3.5-Flash). On Qwen3.6-27B (2x A100, concurrency 1) block-parallel drafters report 2.4-5.7x (e.g. Domino-B16 MT-Bench 3.44x, MATH500 5.72x; MTP-S3 2.44x on MT-Bench); no EAGLE-3 column is given in the text for that model. — [SpecForge paper](https://arxiv.org/html/2603.18567); [LMSYS blog 2026-08-04](https://lmsys.org/blog/2026-08-04-specforge-v0-3)
- TorchSpec (PyTorch blog, March 2026): disaggregated EAGLE-3 training streaming hidden states over RDMA/TCP (Mooncake), supports vLLM and SGLang as capture engines, "train with decode" (regenerating responses on the fly). Kimi-K2.5 head: throughput +60% at batch size 1, +30% at batch 8, +26% at batch 16 with lookahead 3. — [PyTorch blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- vLLM `speculators` (Red Hat): trains EAGLE-3, P-EAGLE, DFlash, DSpark, MTP heads with Qwen/Qwen3-8B as the default tutorial target; online and offline modes. — [speculators tutorial](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/train/)
- SpecVocab (arXiv 2602.13836, Feb 2026; Samsung AI Center Cambridge): context-dependent vocabulary subset for the draft LM head; claims higher acceptance length than EAGLE-3 and up to 8.1% higher average throughput. [snippet] — [arXiv 2602.13836](https://www.arxiv.org/pdf/2602.13836)
- HCSpec (ACL 2026): dual-path draft; claims 15-30% higher end-to-end speedup than EAGLE-3 and up to 3.72x over vanilla. [snippet] — [ACL Anthology 2026.acl-long.353](https://aclanthology.org/2026.acl-long.353.pdf)
- "EAGLE 3.1" (AMD ATOM vendor documentation, May 2026): described as adding fc_norm and post-norm hidden-state feedback, with a claim of up to 2x acceptance length on long-context production workloads. [snippet; vendor claim, no paper found] — [AMD ROCm ATOM docs](https://rocm.docs.amd.com/projects/atom/en/main/_sources/eagle3_speculative_decoding.md)
- SAMD[EAGLE2] (suffix-automaton retrieval combined with EAGLE-2, no extra training) tops EAGLE-2 on Spec-Bench: Vicuna-7B A100 overall 2.73x vs 2.36x; summarization 3.19x vs 2.34x. — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)

### Inferences
- HASS's Align-n is effectively the same idea as EAGLE-3's training-time test, but EAGLE-3 goes further (7 steps vs 3, token-only loss, fused features, 8x data). On the common model (LLaMA3/3.1-8B, MT-bench-heavy means) HASS reports tau ≈ 5.08 vs EAGLE-3's 6.23 — measured in different harnesses, so indicative only; no paper found runs HASS/CORAL and EAGLE-3 head-to-head.
- Two follow-up ideas remain orthogonal to EAGLE-3 and are relevant to Qwen3's 151,936-token vocabulary: (a) the LM-head cost (CORAL: ~59% of draft latency on Llama3-8B) — EAGLE-3's fixed 32k draft vocabulary already addresses this, CORAL's router and SpecVocab are dynamic alternatives; (b) retrieval hybrids (SAMD) that help on copy-heavy tasks like summarization/RAG.
- HASS's finding that more than 3-4 unroll steps stops helping (capacity-limited) and SpecForge's finding that the best TTT length is task-dependent (3 for chat, 13 for math/code) both suggest TTT length should be tuned to the deployment workload rather than left at 7.

### Gaps
- I did not find a paper literally named "Falcon" that modifies EAGLE training within this pass; Falcon (semi-autoregressive drafting, AAAI 2025) was not opened and is not covered here.
- SpecVocab, HCSpec and "EAGLE 3.1" numbers are from search snippets only and were not verified against the primary documents.
- No follow-up was found that reports its extra training cost in absolute GPU-hours.

---

## 6. Failure modes, caveats, and reproduction gaps

### Takeaway
The published 4-6x figures are best-case: batch size 1, greedy, a slow vanilla baseline, and in-distribution chat/code/math. Speedup shrinks ~15-30% at temperature 1, collapses toward 1.0-1.4x at batch sizes of 16-64, is roughly halved inside optimized engines (SGLang/vLLM), is weaker on translation/summarization/non-English text and MoE targets, and under-trained or mis-templated heads land at acceptance length ~1.5-2.0.

### Cited Findings

**Temperature / sampling**
- EAGLE-3, LLaMA-3.1 8B: mean 4.44x / tau 6.23 at T=0 vs 3.45x / 4.92 at T=1 (-22% speedup); MT-bench 4.40x -> 3.07x (-30%). Vicuna 13B 5.51x -> 4.65x; LLaMA-3.3 70B 4.12x -> 3.95x (-4%). [paper] — [EAGLE-3, Table 1](https://arxiv.org/html/2503.01840)
- EAGLE-1, Vicuna 7B HumanEval: 3.33x (T=0) -> 2.39x (T=1) (-28%); MT-bench tau 3.94 -> 3.17. [paper] — [EAGLE, Tables 1-2](https://arxiv.org/html/2401.15077)
- Baseten: T=1 reduces speedup by roughly 15-25%; repetition penalties and top-p/top-k also lower acceptance; some serving frameworks disable sampling under speculative decoding (TRT-LLM needs an explicit flag). — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/)
- RedHat's Qwen3-8B head at Qwen's recommended sampling (T=0.6, top_p 0.95, top_k 20): chain acceptance length saturates at 2.3-2.8. [card] — [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)

**Batch size / throughput serving**
- EAGLE-1 (Table 7, MT-bench, T=0, fp16), speedup at bs 1/2/3/4: Vicuna 7B 2.90x / 2.87x / 2.65x / 2.76x; LLaMA2-Chat 70B 3.01x / 2.81x / 2.50x / 2.40x; max-throughput gain ~1.97-1.99x; EAGLE reduces the max batch that fits (Vicuna 7B on a 3090: 8 -> 7; 70B on 4x A100-40G: 5 -> 4). [paper] — [EAGLE, Table 7](https://arxiv.org/html/2401.15077)
- EAGLE-3 in SGLang (H100, LLaMA-3.1 8B, MT-bench, chain of 3, no tree), throughput ratio at bs 2/4/8/16/24/32/48/56/64: EAGLE 1.40 / 1.38 / 1.23 / 1.02 / 0.93 / 0.94 / 0.88 / 0.99 / 0.99; EAGLE-3 1.81 / 1.82 / 1.62 / 1.48 / 1.39 / 1.32 / 1.38 / 1.34 / 1.38. [paper] — [EAGLE-3, Table 3](https://arxiv.org/html/2503.01840)
- EAGLE-3 in vLLM (LLaMA-3.1 8B, MT-bench, chain of 2; text says RTX3090, caption says A100), bs 2/4/8/16/24/32/48/56: EAGLE 1.30 / 1.25 / 1.21 / 1.10 / 1.03 / 0.93 / 0.82 / 0.71; EAGLE-3 1.75 / 1.68 / 1.58 / 1.49 / 1.42 / 1.36 / 1.21 / 1.01. [paper] — [EAGLE-3, Table 5](https://arxiv.org/html/2503.01840)
- TorchSpec Kimi-K2.5 EAGLE-3 head: +60% at bs 1, +30% at bs 8, +26% at bs 16. — [PyTorch blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- SpecForge v0.3 (non-EAGLE block drafters, same trend): Qwen3.6-27B Domino 4.60x at concurrency 1 narrows to 1.48-2.11x at concurrency 32. — [LMSYS blog](https://lmsys.org/blog/2026-08-04-specforge-v0-3)
- A Qwen3-Coder-Next EAGLE-3 head (80B MoE) reports only 1.31x on MT-Bench at B=32 with a wide tree and 1.10x with a narrow tree. [snippet] — [thoughtworks/Qwen3-Coder-Next-Eagle3 card (mirror)](https://hface.pages.dev/thoughtworks/Qwen3-Coder-Next-Eagle3/blob/main/README.md?code=true)
- Baseten: EAGLE helps most for "memory-bound, latency-sensitive, lightly batched workloads". — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/)

**Optimized engines / quantization shrink the ratio**
- EAGLE-3 paper, same model and benchmark: 4.40x in the authors' harness vs 373.25 / 158.34 = 2.36x in SGLang on H100 (EAGLE-2: 1.54x). [paper] — [EAGLE-3, Tables 1 & 4](https://arxiv.org/html/2503.01840)
- EAGLE-1 + gpt-fast on LLaMA2-Chat 7B: 1.82x at fp16 (55.1 -> 100.2 tok/s) but 1.50x at int4 (106.9 -> 160.4 tok/s). The README claims the head works on the int4 model with "No additional training required". [paper/code] — [EAGLE, Table 4](https://arxiv.org/html/2401.15077); [v1 README](https://raw.githubusercontent.com/SafeAILab/EAGLE/v1/README.md)
- Qwen2 targets overflow in fp16; use bf16. — [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
- Open issue #207 reports inconsistent outputs between standalone Qwen and EAGLE at temperature 0 (i.e. a "losslessness" complaint in practice); details not read. [snippet] — [EAGLE issue index](https://issues.ecosyste.ms/hosts/GitHub/repositories/SafeAILab%2FEAGLE/issues)

**Domain / language / task shift**
- Task spread on Spec-Bench (A100, Vicuna-13B, EAGLE-3): MT-bench 3.48x and math 3.42x vs translation 2.36x and RAG 2.78x; for EAGLE-2 on Vicuna-7B translation is 1.88x vs 2.84x on MT-bench. — [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- Qwen3-8B (WhiFlash, Spec-Bench): acceptance length 3.60 on translation vs 5.94 on math; throughput 159 vs 298 tok/s. [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3)
- EAGLE-3 authors: summarization (CNN/DM) is the weakest task for every model (e.g. LLaMA-3.1 8B 3.65x vs 4.85x HumanEval); code gains are attributed partly to fixed templates in code; the R1-distill GSM8K result (5.01x) benefits from training on OpenThoughts-114k-math. [paper] — [EAGLE-3, Sec. 4.1](https://arxiv.org/html/2503.01840)
- Fixed 32k draft vocabulary: "cannot directly propose target tokens outside this vocabulary, which can reduce acceptance length" [card] — [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3). A multilingual Qwen3-8B EAGLE-3 research preview states an English-derived 32k vocabulary covers only about half of Hindi/Gujarati tokens, so acceptance drops sharply. [snippet] — [SwitchXDDD/multilingual-eagle3-qwen3-8b](https://huggingface.co/SwitchXDDD/multilingual-eagle3-qwen3-8b). A Chinese-specific Qwen3-8B head exists (Zjcxy-SmartAI/Eagle3-Qwen3-8B-zh), consistent with language-specific heads being needed. — [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
- SpecForge: public EAGLE-3 checkpoints had "limited math and code data"; the broader 1.4M-conversation mix raised Math500 acceptance 2.82 -> 3.48, but MT-Bench slightly dropped for Llama-3 8B and 70B. [paper] — [SpecForge](https://arxiv.org/html/2603.18567)
- System-prompt/template sensitivity: CORAL reports switching the system prompt between training and inference reduced acceptance length (e.g. -0.31 on MT-bench); Baseten calls wrong chat templates and prompt leakage the most common cause of poor results. — [CORAL](https://arxiv.org/html/2502.16880v1); [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/)
- FSPAD: all EAGLE-style methods score lower on LLaMA3-Instruct than on Vicuna, which the authors attribute to ShareGPT being a poorer match to LLaMA3's (closed) SFT data. [paper] — [FSPAD](https://arxiv.org/html/2408.15562v1)

**Reasoning models, MoE targets**
- EAGLE-3 on DeepSeek-R1-Distill-LLaMA 8B needed an added reasoning dataset; with it, mean 4.16x / 5.84 at T=0 and 3.52x / 4.89 at T=1 (EAGLE-2 on the same model: 3.26x and 2.77x). [paper] — [EAGLE-3, Table 1](https://arxiv.org/html/2503.01840)
- SpecBundle covers instruct models only; reasoning and vision-language targets are future work. [paper] — [SpecForge](https://arxiv.org/html/2603.18567)
- MoE targets do worse: Mixtral 8x7B 1.50x with EAGLE-1 [paper — [EAGLE, Table 3](https://arxiv.org/html/2401.15077)]; a maintainer says MoE models perform worse with EAGLE than dense [— [issue #276](https://github.com/SafeAILab/EAGLE/issues/276)]; Qwen3-30B-A3B 1.37x vs ~1.7x for dense Qwen3 sizes [card — [AngelSlim](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)]. (Qwen3-8B is dense, so this is context rather than a direct risk.)

**Reproduction gaps**
- Issue #209: a user got at most 3.18x with the official EAGLE-3 LLaMA-3.1-8B head vs 4.4x in the paper; cause was draft tree depth (default 5 is for EAGLE-2; EAGLE-3 needs 7-8). The maintainer reported 4.00x-4.20x on an A800 and noted results depend mostly on the GPU, with a smaller CPU effect. — [EAGLE issue #209](https://github.com/SafeAILab/EAGLE/issues/209)
- Issue #276: Qwen2.5-7B EAGLE-3 head on 50k ShareGPT x 5 epochs -> acceptance length 1.9. — [EAGLE issue #276](https://github.com/SafeAILab/EAGLE/issues/276)
- Paper-vs-independent gaps: EAGLE-2 on Vicuna-7B MT-bench is 3.62x in the EAGLE-2 paper vs 2.84x (A100) / 2.56x (3090) on Spec-Bench; EAGLE-3 on Vicuna-13B MT-bench is 5.58x / tau 6.65 in the paper vs 3.48x and 5.71 mean accepted tokens (across tasks) on Spec-Bench A100. EAGLE-2 on LLaMA3-8B MT-bench: 3.46x (EAGLE-2 paper) vs 3.36x (FSPAD, A100) vs 2.28x (CORAL, A6000); mean 2.83x (HASS, H800). — [EAGLE-2](https://arxiv.org/html/2406.16858); [EAGLE-3](https://arxiv.org/html/2503.01840); [Spec-Bench](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md); [FSPAD](https://arxiv.org/html/2408.15562v1); [CORAL](https://arxiv.org/html/2502.16880v1); [HASS](https://arxiv.org/html/2408.15766)
- Internal inconsistencies in the EAGLE-3 paper: batch-64 throughput gain is "40%" in the intro vs "38%" / 1.38x in Sec. 4.3; vLLM hardware is RTX3090 in text vs A100 in the Table 5 caption; one DeepSeek-R1 EAGLE-2 CNN/DM cell reads 3.53x with tau 3.33 at T=0 (higher speedup than the EAGLE-3 cell, likely a typo). [paper] — [EAGLE-3](https://arxiv.org/html/2503.01840)
- Acceptance length does not convert 1:1 into speed: FSPAD gains +0.3-0.5 tau for +1-7% speed; CORAL's router lowers tau (4.90 -> 4.63 avg on Llama3-8B) yet raises speed (2.68x -> 2.87x). [paper] — [FSPAD](https://arxiv.org/html/2408.15562v1); [CORAL](https://arxiv.org/html/2502.16880v1)
- Serving-side knobs: Baseten recommends only 3-4 draft tokens in production (8 "rarely helps"); EAGLE-3's own engine experiments use chains of 2-3 without the tree. — [Baseten blog](https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/); [EAGLE-3, Sec. 4.3-4.4](https://arxiv.org/html/2503.01840)

### Inferences
- The speedup ratio is sensitive to hardware in a predictable direction: the less the baseline is bottlenecked by Python/framework overhead and the faster the target's forward pass, the smaller the ratio (authors' harness 4.4x -> SGLang/H100 2.4x; Qwen3-8B 2.4x on RTX 5090 vs 1.95x on H200 in the same card). A retrofit evaluated only against HF `generate` will overstate deployable gains by roughly 2x.
- For a Qwen3-8B deployment with sampling (Qwen's recommended T=0.6-0.7), thinking mode, and moderate concurrency, the evidence supports planning for ~1.5-2x per-request latency gain at low concurrency, decaying toward ~1.0-1.4x at batch 32-64, rather than the 4.4x headline.
- The tree-vs-chain choice is a deployment fork: the big taus (5-6.6) need a 48-60-token tree verified per step, which multiplies verify-pass tokens and is what breaks down under batching; chain drafting of 2-4 tokens is what vLLM/SGLang deployments actually run and yields tau ≈ 2-2.8.
- Because losslessness is enforced by verification, a weak or out-of-distribution head costs only speed, never output quality — but a head with acceptance length < ~1.5 can make serving slower than no speculation (EAGLE-1 at batch ≥ 24 falls to 0.71-0.99x).

### Gaps
- No systematic study of EAGLE-family acceptance on quantized *targets* (AWQ/GPTQ/FP8) was found beyond the single gpt-fast int4 row in EAGLE-1; TorchSpec only says quantized targets are supported for training.
- No controlled measurement of thinking-mode vs non-thinking acceptance for Qwen3 heads was found.
- No controlled non-English benchmark for an EAGLE-3 head was found; the multilingual evidence is one research-preview model card (snippet) plus the draft-vocabulary caveat.
- Issues #207 (Qwen output mismatch at T=0) and #235 (how acceptance length is computed in EAGLE-3) were seen only as titles.
