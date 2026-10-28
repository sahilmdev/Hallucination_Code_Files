# Data and results

The folder layout matches the `hallucination_project_v2` Google Drive folder the notebooks use, so the notebooks run unchanged when this repository is placed there.

The raw ToolBench, API-Bank and ToolAlpaca files are not included; `Code_1_data_and_benchmark.ipynb` downloads them from their original sources.

## `data/data_v2/` (Code_1)
- `registry.json` - schema registry (5,155 functions, 9,774 parameters)
- `benchmark_single.json` - 2,800 single-error calls
- `benchmark_multi.json` - 1,200 calls with 1-4 errors
- `splits.json` - random and function-disjoint tune/test splits
- `real_eval.json` - ToolAlpaca queries for the real-LLM study
- `examples_by_type.json`, `data_summary.json` - examples and dataset statistics

## `runs/run_A/` (Code_2 to Code_6)
- `paper_numbers.json`, `final_tables.json` - every number reported in the paper
- `core_results.json`, `core_records.json` - main experiments, ablations, robustness
- `reference_metrics.json`, `baselines.json`, `retrieval.json` - baselines
- `real_results.json`, `real_records.json`, `selfcheck_results.json` - real LLM calls and SelfCheckGPT
- `human_eval/` - annotation guidelines, items, both passes of both raters (R1, R2) and the adjudication
- `tables/`, `figures/` - generated tables (CSV and LaTeX) and figures (PDF and PNG)

## `cache/`
Cached LLM generations, SelfCheckGPT samples, LLM-baseline outputs and web-search results, so all results can be reproduced without re-running the models or the searches. Sentence embeddings are not included; they are recomputed automatically.
