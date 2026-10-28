# Tool-call hallucination project — code v2

Six Colab notebooks plus one shared code file. Every notebook runs on a **fresh runtime** and passes
results to the next one **only through Google Drive**.

## One-time setup

1. In Google Drive create the folder `MyDrive/hallucination_project_v2/code/`.
   (A new folder on purpose — nothing from the old `hallucination_project/` folder is read.)
2. Upload **`toolhall_core.py`** and **`config.json`** into that `code/` folder.
3. Open the notebooks in Colab from this folder.

## Run order

| # | Notebook | Runtime | Typical time | Produces (on Drive) |
|---|---|---|---|---|
| 1 | `Code_1_data_and_benchmark.ipynb` | CPU | ~10–15 min | `data/data_v2/` — registry, benchmarks, splits, real-eval set |
| 2 | `Code_2_reference_metrics.ipynb` | **GPU** (T4) | ~15 min | `runs/run_A/reference_metrics.json` |
| 3 | `Code_3_core_experiments.ipynb` | CPU | ~20–40 min | `runs/run_A/core_results.json`, `core_records.json` |
| 4 | `Code_4_baselines_and_retrieval.ipynb` | **GPU** | ~1–2 h | `runs/run_A/baselines.json`, `retrieval.json` |
| 5 | `Code_5_real_llm_calls.ipynb` | **GPU** | several h (resumable) | `runs/run_A/real_results.json`, `real_records.json` |
| 5b | `Code_5b_selfcheck_baseline.ipynb` | **GPU** | ~45–60 min (3 models × 3 samples; resumable) | `runs/run_A/selfcheck_results.json` |
| 6 | `Code_6_human_eval_and_figures.ipynb` | CPU | ~2 min | `runs/run_A/human_eval/`, `tables/`, `figures/`, `paper_numbers.json` |

Times are estimates. Code_5 caches every generation in `cache/real_generations/`; after a Colab
disconnect, simply run the notebook again and it continues where it stopped.
Code_4 and Code_5 have `QUICK_TEST = None` near the top: set it to e.g. `20` for a short trial run first.

## What to re-run after a change

| You changed | Re-run |
|---|---|
| `data_version`, benchmark settings, injection | 1 → 6 |
| validator / corrector / matcher code or settings | 3 → 6 (Code_5 re-scores from cache, no new generation) |
| only the human ratings | 6 |

To keep an old set of results, change `run_id` in `config.json` (e.g. `run_B`) before re-running;
results go to a new `runs/run_B/` folder. Every saved file records its `run_id` and `data_version`,
and a notebook **refuses to read** a file from a different run — this is what prevents the
mixed-run problem of the previous version.

## Human evaluation

Code_6 writes the rating sheets to `runs/run_A/human_eval/` together with `ANNOTATION_GUIDELINES.md`.
Two raters fill them in **independently**, upload them back with the same names, and Code_6 is
re-run. There is no simulated fallback: without real ratings no human-evaluation number is produced.

See `CHANGES.md` for everything that changed relative to `Final Code Files/`.
