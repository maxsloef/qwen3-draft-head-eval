# Qwen3-8B draft head evaluation

Which published EAGLE-3 draft head is the best predictor of Qwen3-8B's own future tokens?

Four independent EAGLE-3 heads for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) are public. Each is one transformer layer that reads the frozen base model's hidden states and drafts the next few tokens. Their model cards report serving speed under different settings, so they cannot be ranked on prediction quality. This repo scores all four on identical data.

## Result

Qwen3-8B answered the 480 [Spec-Bench](https://github.com/hemingkx/Spec-Bench) prompts greedily (up to 256 new tokens), with thinking off and with thinking on. Each head was given the model's real hidden states and scored on predicting the tokens the model actually produced, at every generated position (76,316 positions thinking off, 120,000 thinking on).

| Head | Step-1 top-1, thinking off | Correct run of 7, thinking off | Step-1 top-1, thinking on | Correct run of 7, thinking on |
|---|---|---|---|---|
| [WhiFlash/Qwen3-8B-EAGLE3](https://huggingface.co/WhiFlash/Qwen3-8B-EAGLE3) | **0.750** | **2.47** | 0.653 | 1.63 |
| [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) | 0.716 | 2.04 | **0.711** | **1.96** |
| [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3) | 0.671 | 1.63 | 0.669 | 1.58 |
| [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) | 0.579 | 1.23 | 0.560 | 1.10 |

- "Step 1" is the token after next, which is the first thing an EAGLE-3 head predicts.
- "Correct run of 7" is the mean number of tokens in a row that a 7-token greedy draft gets right.
- WhiFlash is best with thinking off; Tengyunw is best with thinking on and the most robust across both.

Full tables (top-5, distribution overlap, per-category, vocabulary coverage) are in [`results/main/summary.txt`](results/main/summary.txt).

## Tencent AngelSpec drafters (added later, same data)

Two more Qwen3-8B drafters from [AngelSpec](https://github.com/Tencent/AngelSpec) were scored on exactly the same positions. Neither is an EAGLE-3 head.

| Drafter | What it is | Step-1 top-1, thinking off | Correct run of 7, thinking off | Step-1 top-1, thinking on | Correct run of 7, thinking on |
|---|---|---|---|---|---|
| [AngelSlim/Qwen3-8B-DFly-Block8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-Block8) | 5-layer block drafter, about 1.3B parameters, reads 5 trunk layers | **0.821** | **3.46** | **0.769** | **2.56** |
| [AngelSlim/Qwen3-8B-MTP-TTT3](https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3) | Qwen3.5-style MTP block, one layer, reads only the last trunk layer | 0.779 | 2.79 | 0.718 | 1.95 |
| WhiFlash EAGLE-3 (best EAGLE-3, thinking off) | one layer, reads 3 trunk layers | 0.750 | 2.47 | 0.653 | 1.63 |
| Tengyunw EAGLE-3 (best EAGLE-3, thinking on) | one layer, reads 3 trunk layers | 0.716 | 2.04 | 0.711 | 1.96 |

- The MTP block is the best single-layer head at step 1 in both modes, uses the trunk's own unembedding over the full vocabulary, and reads only the post-norm last hidden state. It was trained for 3 draft steps and fades faster than WhiFlash beyond that (0.436 against 0.620 at step 7, thinking off).
- DFly is the best predictor overall, but it is a separate 5-layer network on top of the trunk's hidden states. Its slots are predicted in parallel, then a correction module conditions each slot on the embedding of the previous token. The rows above give it the true previous token, as the chain heads get. Without the correction (pure parallel prediction, which is not how it was trained to be used) it scores 0.590 at step 1 and 0.147 at step 7, thinking off.
- Results: [`results/mtp/summary.txt`](results/mtp/summary.txt), [`results/dfly/summary.txt`](results/dfly/summary.txt). Run with `--only angelspec-mtp` or `--only dfly`.
- The DFly checkpoint's `config.json` on the Hub does not match its weights; the settings used here come from `angelspec/config/dfly_qwen3_8b_draft_config.json` in the AngelSpec repo (trunk layers 1/9/17/25/33, mask token 151669), with block size 8.

## How a head is scored

At position *i* the head receives the trunk's hidden states at *i* (the inputs to layers 2, 18 and 33, fused by a linear layer) and the embedding of token *i*+1, and predicts token *i*+2. For steps 2 to 7 it feeds its own output back in with the next true token and predicts one token further. Metrics per step:

- top-1 and top-5 accuracy against the token the model actually generated;
- overlap with the model's own next-token distribution (1 minus total variation);
- whether all steps so far were correct (gives the correct-run length);
- whether the true token is in the head's 32K output vocabulary.

## Files

| File | What it is |
|---|---|
| `heads.py` | Plain-PyTorch EAGLE-3 head forward pass that loads any of the four checkpoints |
| `modal_eval.py` | Generation, scoring and merging across GPU shards on [Modal](https://modal.com) |
| `summarize.py` | Renders the text tables from the raw counts |
| `data/spec_bench_question.jsonl` | The 480 prompts, from Spec-Bench (Apache-2.0) |
| `results/main/` | Tables, raw counts, and the generations that were scored |
| `results/smoke/` | An 8-prompt sanity run |
| `docs/lit-review.md` | Literature review on retrofitting draft heads and MTP modules onto existing models |
| `docs/notes/` | Source notes behind the review |
| `handoff/index.html` | Source of the hand-off page |

## Rerunning

```
uv run --no-project --python 3.12 --with modal modal run --detach modal_eval.py::main --n-shards 4 --tag main
uv run --no-project --python 3.12 --with modal modal run modal_eval.py::fetch --tag main
```

The job expects a Modal volume named `owl-hf` that holds a Hugging Face cache containing `Qwen/Qwen3-8B` (mounted at `/hf`, loaded with `local_files_only=True`). Change the volume name in `modal_eval.py`, or drop `local_files_only`, to use your own. Head weights are downloaded into a volume named `draft-head-eval`. The full run uses 8 L40S containers.

## Caveats

- `heads.py` is a re-implementation written from the [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) and RedHatAI reference code, not a serving stack. It reproduces two model-card figures: AngelSlim's 1.99 tokens per cycle with a 2-token chain (1.89 here) and RedHatAI's 2.1 to 2.5 with a 3-token chain (2.34 here).
- The RedHatAI head scores about one point higher when fed trunk layers one deeper than documented (the `redhat+1` rows). This does not change its rank.
- Greedy decoding only, one benchmark, and the thinking-on run covers only the first 256 tokens of reasoning.
- Two checkpoints ship as pickle files; they are loaded with `weights_only=True`.
