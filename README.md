# Schema-Aware Detection and Correction of Hallucinated Tool Calls

Code, data and results for the paper *"Schema-Aware Detection and Correction of Hallucinated Tool Calls in Large Language Models"* (Sahil Mehta, Vishwesh Nair, Reyansh Mehta, Sunil Ghane; Sardar Patel Institute of Technology, Mumbai).

When an LLM agent calls a tool, it writes a JSON call such as
`{"function": "getWeather", "parameters": {"city": "Mumbai"}}`, and the host program executes it as written. This repository checks such calls **using only the API schema**, with no reference to the correct call. It detects the error, classifies it and repairs what the schema allows.

## What the system handles

| Error type | Example | Handled from the schema? |
|---|---|---|
| `WRONG_FORMAT` | `{"tool": ..., "tool_input": ...}` instead of `{"function", "parameters"}` | yes |
| `WRONG_FUNCTION` | a function name that is not in the registry | yes |
| `EXTRA_PARAMETER` | a parameter the function does not declare | yes |
| `MISSING_PARAMETER` | a required parameter is absent | yes (filled with a marked placeholder) |
| `WRONG_TYPE` | `"10"` where an integer is expected | yes, with adaptive type trust |
| `WRONG_VALUE` | `"measles"` when the user asked about pneumonia | mostly no; needs context |

## Main results

| | Random split | Function-disjoint split (unseen functions) |
|---|---|---|
| Detection, balanced accuracy | 0.904 | 0.910 |
| False-alarm rate on correct calls | 3.3% | 1.6% |
| Classification, macro-F1 | 0.827 | 0.830 |
| Exact match after correction | 0.766 | 0.775 |

On 810 real calls from five open LLMs, the pipeline never damaged a correct call. Against the human labels it reached a precision of 0.99 and a recall of 0.64, because most real errors are semantic. Combining it with SelfCheckGPT caught 94% of the human-identified errors. Every number in the paper is in [`runs/run_A/paper_numbers.json`](runs/run_A/paper_numbers.json).

## Repository layout

```
code/
  toolhall_core.py        all logic: registry, injection, validator, corrector, metrics
  config.json             every setting and threshold
Code_1 ... Code_6, Code_5b  the notebooks (run in order, see below)
data/benchmark/           output of Code_1
  registry.json           schema registry: 5,155 functions, 9,774 parameters
  benchmark_single.json   2,800 calls (1,000 correct + 300 per error type)
  benchmark_multi.json    1,200 calls with 1-4 errors
  splits.json             random and function-disjoint tune/test splits
  real_eval.json          ToolAlpaca queries for the real-LLM study
runs/run_A/               outputs of Code_2 to Code_6
  paper_numbers.json      every number reported in the paper
  *_results.json          aggregated results; *_records.json per-call records
  human_eval/             guidelines, items, both raters' sheets (two passes), adjudication
  tables/  figures/       tables (CSV, LaTeX) and figures (PDF, PNG)
cache/                    cached LLM generations, SelfCheckGPT samples and web searches
```

The raw ToolBench, API-Bank and ToolAlpaca data are **not** included. `Code_1` downloads them from their original sources.

## Running the code

The notebooks are written for Google Colab. Each one starts from a fresh runtime and reads its inputs from the files written by the notebooks before it.

**Setup:** copy this repository into Google Drive as `MyDrive/hallucination_project/`. If you run it elsewhere, set the environment variable `TOOLHALL_ROOT` to the repository folder. Then install the packages in `requirements.txt`; Colab already has most of them.

| # | Notebook | Runtime | Approx. time | Produces |
|---|---|---|---|---|
| 1 | `Code_1_data_and_benchmark` | CPU | 10-15 min | `data/benchmark/` |
| 2 | `Code_2_reference_metrics` | GPU (T4) | 15 min | `reference_metrics.json` |
| 3 | `Code_3_core_experiments` | CPU | 20-40 min | `core_results.json`, `core_records.json` |
| 4 | `Code_4_baselines_and_retrieval` | GPU | 1-2 h | `baselines.json`, `retrieval.json` |
| 5 | `Code_5_real_llm_calls` | GPU | several h | `real_results.json`, `real_records.json` |
| 5b | `Code_5b_selfcheck_baseline` | GPU | 45-60 min | `selfcheck_results.json` |
| 6 | `Code_6_human_eval_and_figures` | CPU | 2 min | `human_eval/`, `tables/`, `figures/`, `paper_numbers.json` |

**Reproducing without the GPU runs:** all LLM generations and web searches are cached in `cache/`, so Code_4, Code_5 and Code_5b reuse them instead of calling the models again. The finished results are already in `runs/run_A/`; running Code_6 alone rebuilds every table and figure from them.

**Notes**
- Code_4 and Code_5 have `QUICK_TEST = None` near the top. Set it to e.g. `20` for a short trial run.
- Every result file records its `run_id`, `data_version`, configuration hash and package versions, and a notebook refuses to read a file from a different run. To keep the existing results, change `run_id` in `code/config.json` (e.g. to `run_B`) before re-running.
- Everything uses the fixed seed 42.

| If you change | Re-run |
|---|---|
| benchmark or injection settings | Code_1 to Code_6 |
| validator, corrector or matcher settings | Code_3 to Code_6 (Code_5 re-scores from the cache) |
| only the human ratings | Code_6 |

## Human evaluation

200 real calls (plus 20 practice items) were labelled independently by two annotators in two passes: first the original call, then the corrected call. Disagreements were resolved by discussion. The guidelines are in `runs/run_A/human_eval/ANNOTATION_GUIDELINES.md`. There is no simulated fallback: without real ratings, Code_6 produces no human-evaluation numbers.

## Models and data used

- Sentence encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Reference metrics: DeBERTa-v3-small (NLI), RoBERTa-large (BERTScore)
- LLMs: Qwen2.5-0.5B/1.5B/3B-Instruct, TinyLlama-1.1B-Chat, Phi-3.5-mini-instruct
- Datasets: ToolBench (Qin et al., 2024), API-Bank (Li et al., 2023), ToolAlpaca (Tang et al., 2023)

## Citation

If you use this code or benchmark, please cite the paper (details will be added after publication).

## License

The code is released under the MIT License (see `LICENSE`). The data derived from ToolBench, API-Bank and ToolAlpaca remain subject to the licenses of those datasets.
