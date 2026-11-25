# Who Read the Edit?

Code, raw results and paper sources for *Who Read the Edit? Repairing Cached Attention State after Mid-Context Changes*.

When the middle of a live LLM context changes (an old tool result is cleared, a document is deleted, a result arrives late, two branches merge), every cached key and value after the change was computed against content that is no longer there. A prefix cache throws all of it away. The paper asks which of those cached tokens actually carry the error, and recomputes only those. The short answer: the tokens that read the changed content and are then read by the next input. How much each token read of the edited span can be recorded during prefill, and how much the next input reads of each cached token takes one extra pass; ranking tokens by both signals together beats either signal alone in most of our experiments.

This repository has:

- `mutablekv/lm.py`: a forward pass for Llama and Qwen2 models, written on the Hugging Face weights, that recomputes any subset of a sequence while reading every other token from cache at any position;
- `mutablekv/heal.py`, `mutablekv/sketch.py`: partial recomputation and the estimate of attention to inserted content;
- `mutablekv/context.py`, `mutablekv/stream.py`: a small runtime for mutable contexts (segments over immutable cells, content-addressed versions that say which cells are still exact, insert, delete, replace, fork, merge, out-of-order arrival and watermarks);
- `experiments/scripts/`: every experiment, and `experiments/results/`: the raw output of each run;
- `paper/`: the paper's sources and the script that turns the raw results into every number, table and figure in it.

Everything was run on one Apple M5 laptop with 24 GB of memory: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct in 32-bit floats, and Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 in bfloat16.

## Setup

```bash
bash env/setup.sh                       # Python 3.12 in ./.venv, pinned packages, pinned model revisions
.venv/bin/python env/download_data.py   # WikiText-103 and HotpotQA validation sets
```

`env/requirements.lock.txt` lists the exact package versions. The scripts use the Apple GPU (MPS) when it is available and the CPU otherwise, and most take an optional device argument. Only MPS and the CPU were tested.

## Tests

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

The tests check the engine against the reference implementation, the relocation identities in 64-bit floats, that recomputing a block starting at the edit gives exact cells, that the sweep converges to the cold pass, the estimate of attention to inserted content, and the runtime's exactness tracking, fork, merge and watermarks. They take about a minute and a half.

## Reproducing the results

```bash
bash experiments/scripts/run_queue.sh experiments/scripts/reproduce.txt
```

`reproduce.txt` lists every run behind the paper, grouped by section. The runner executes one job at a time and writes a log per job to `experiments/logs/`. Run nothing else on the GPU meanwhile: each script loads a model, and two at once will swap on a 24 GB machine. Every script saves its rows as it goes and resumes from them. The 7B runs wait until about 15 GB of memory is free (`wait_for_memory.py`). To use another interpreter, set `PYTHON=/path/to/python`.

| Script | Question | Paper |
|---|---|---|
| `e0_exactness.py` | Is the engine exact? Does rewriting keys in a 16-bit cache drift? | §2.3, §3, App. A |
| `e14_kernels.py` | What do offset grouping and per-segment exposure sums cost in one attention call? | §3, §5.6, App. A |
| `e1_anatomy.py` | Where does staleness live after an interior edit? Gap vs relocation | §4, App. B |
| `e1p_paired.py` | The same deletions read with the question fresh and cached | §4.2 |
| `e15_gap.py` | How much does a position gap alone change the output? | §4.3 |
| `e2c_ablation.py`, `e2d_reader_first.py` | Main experiment: every policy, three placements of the question, three suffix lengths | §5.2, App. C |
| `e22_lmcache.py` | CacheBlend's selection exactly as LMCache runs it | §5.1 |
| `e16_confirm.py` | Replication on new questions with hypotheses fixed in advance (variants `plain`, `decoy`, `long`) | §5.3, App. D |
| `e9_scale.py`, `e19_family.py` | The main experiment on Qwen2.5-7B and Mistral-7B | §5.3, App. E |
| `e10_long.py` | Repair in 4k to 16k-token contexts | §5.3, App. E |
| `e11_longq.py` | A long new input: can reader attention come from a few of its tokens? | §5.6, App. E |
| `e12_agent.py` | Templated agent traces: clearing tool results, once and twice in a row | §5.3, §6, App. E, G |
| `e21_traces.py` | Real agent transcripts (AgentInstruct): clearing an old tool result | §5.3, §5.4, App. E |
| `e3_needle.py` | Needle erasure, following KVEraser's protocol | §5.3, §5.4, App. E |
| `e7_mechanism.py` | Does a block's score predict what recomputing it buys? Greedy selection as a reference | §5.2, App. E |
| `e8_counterfactual.py` | Replacing a fact the answer depends on | App. E |
| `e2_insert_sketch.py`, `e2e_insert_reader.py` | Inserts, where no exposure was recorded | §5.4, App. E |
| `e6_cost.py`, `e6b_recording.py` | Time of a repair against a re-prefill, and of recording exposure | §5.6 |
| `e23_scatter.py` | Scattered tokens against one contiguous block of the same size | §5.6 |
| `e24_cost_model.py` | Roofline estimate of repair in a fused engine (no model run) | §5.6, App. A |
| `e5_streams.py` | Out-of-order arrival and branch merge through the runtime | §6, App. G |
| `e17_precision.py`, `e18_reference.py`, `e20_reference.py` | Numerical noise: 32-bit vs bfloat16, our engine vs the library | §2.3, App. F |
| `e2_healing.py`, `e2_fill.py` | The first, smaller repair run | App. H |

`experiments/preregistration.md` holds the decision rules of the replication, copied unchanged from the research log entry written before most of its results were examined. One file in `experiments/results/backup/` comes from an earlier version of the engine that rounded attention scores to bfloat16 too early; the paper uses it only to report that version's agreement with the library (Appendix F).

## Building the paper

```bash
bash paper/make_pdf.sh   # needs pandoc and tectonic
```

This runs `make_numbers.py` (raw results to every number and table), `make_figures.py` and `build.py` (sources plus numbers to `paper/draft.md`), then typesets `paper/paper.pdf`. No number in the paper is typed by hand: each is a placeholder filled from the raw results, and the build fails if any value or citation is missing.

## Scope

Fidelity is measured against a re-prefill of the same model. Nothing here measures latency or throughput inside a production serving engine or a fused attention kernel, and no model larger than 7B parameters was run.

## Citation

```bibtex
@misc{whoreadtheedit2026,
  title  = {Who Read the Edit? Repairing Cached Attention State after Mid-Context Changes},
  author = {Ashraf Taha},
  year   = {2026}
}
```
