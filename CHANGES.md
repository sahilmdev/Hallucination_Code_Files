# What changed — `Final Code Files/` (v1) → `latest code files/` (v2)

Mentor points are referred to as #1 … #20 (see the review email).

## 0. Structure

| v1 | v2 |
|---|---|
| `Code_1_finalleeee` data + benchmark | **`Code_1_data_and_benchmark`** |
| `Code_2_finalleeee` EM / F1 / NLI / BERTScore | **`Code_2_reference_metrics`** |
| `Code_3_finalleeee` detection (cell 3) + correction (cell 9) | **`Code_3_core_experiments`** (+ ablation, robustness, multi-error, value settings, runtime) |
| `Code_4_finalleeee` retrieval, baselines, ablation, error analysis, cross-dataset, human eval | split into **`Code_4_baselines_and_retrieval`**, **`Code_5_real_llm_calls`**, **`Code_6_human_eval_and_figures`** |
| validator / corrector / matcher copy-pasted into ~6 cells | **one file, `toolhall_core.py`**, imported by every notebook |
| settings scattered in cells | **`config.json`** |
| files copied to/from Drive by name, no check | every file carries `run_id` + `data_version`; loading a file from another run **raises an error** |
| Drive folder `hallucination_project/` | new folder `hallucination_project_v2/` (old results can never be mixed in) |

## 1. Bugs fixed (these change the numbers)

| # | Where (v1) | Bug | Fix (v2) |
|---|---|---|---|
| B1 | Code_3, Code_4 | `TEST_IDX` computed but never used; detection, correction, ablation, error analysis all ran on all 1,400 examples incl. the tune data used for calibration, while the paper says "test split" | everything is tuned on the tune part and reported on the test part, for two splits |
| B2 | Code_4 baseline table | "Exact Match 0.919" / "NLI 0.44" are *detection* accuracy, "full system 0.597" is *end-to-end correctness* — compared (and McNemar-tested) as if the same metric | detection and correction are separate tables with separate tests |
| B3 | Code_3 `STANDARD_KEY_SETS` | `{action, action_input}` — one of the 4 injected wrong envelopes — was listed as an *accepted* envelope, so it could never be detected | accepted envelopes come from `config.json` (default: only `{function, parameters}`) |
| B4 | all scoring | exact match normalised both calls first, so a call left in a wrong envelope counted as correct (why v1's `no_format_check` ablation showed "no effect") | `exact_strict`: output must be in an accepted envelope AND equal the reference |
| B5 | Code_2–4 `normalize_tool_call` | the OpenAI-style `{"type":"function","function":{name, arguments}}` became a function named `"{'name': ...}"` (Code_1's version handled it, the others did not) | one normaliser in the core, handles nested shapes and string `arguments` |
| B6 | Code_2 NLI, Code_4 retrieval | NLI columns misread: for `nli-deberta-v3-small` the order is contradiction / entailment / neutral; v1 used column 2 as "entailment" (it is neutral) | the entailment column is looked up by name from the model config |
| B7 | Code_4 retrieval | a call counted as "genuinely fixed" if ANY one flagged parameter matched, even if the rest of the call was still wrong | the retrieved value is inserted and the WHOLE call must equal the reference |
| B8 | Code_1 ToolBench registry | every parameter (incl. `optional_parameters`) stored as `required: True` | `required` true only for `required_parameters`; MISSING_PARAMETER injection removes only required ones |
| B9 | Code_1 API-Bank | the `input` field = API JSON object(s) **followed by the dialogue**; `json.loads` failed on it, so API-Bank schemas were missing (→ the 62 "function not in registry" false positives) and the query was only the generic instruction | objects decoded one by one; all go into the registry; the dialogue becomes the query |
| B10 | Code_1 API-Bank calls | regex parser cut dict/list values to `"{"` | parsed with Python `ast`, regex only as fallback |
| B11 | Code_1 | ToolBench queries truncated to 300 characters | full query kept (needed for the query-grounding experiment) |
| B12 | Code_3 type check | raw type strings compared: `'date (yyyy-mm-dd)'`, `'enum'`, `'time (...)'` never matched any value → always WRONG_TYPE | types mapped to 6 canonical types; unknown types are not checked |
| B13 | Code_1 benchmark | the same base calls were reused for several error types | every example uses a different base call; at most 3 calls per function |
| B14 | Code_1 ToolAlpaca | AviationAPI's OpenAPI doc nests 5 paths inside another path, and the dataset's function list omits them → their gold calls looked like "unknown function" | nested paths recovered; every OpenAPI operation added |
| B15 | Code_4 runtime | n/a in v1; found while testing v2: timing after a warm cache under-measures latency ~50× | latency measured with an empty similarity cache |
| B16 | Code_1 API-Bank | the same `apiCode` is defined with different parameter lists (330 functions); v1 kept whichever came last | the most frequent definition is kept; reference calls written against another variant are removed and counted |

## How v2 was tested before hand-over

On a local CPU with the real API-Bank, ToolAlpaca and ToolBench data: Code_1 and Code_3 full runs;
Code_2 with small stand-in models (DeBERTa-xsmall NLI, DistilBERT BERTScore); Code_4 and Code_5 with
`QUICK_TEST` and Qwen2.5-0.5B on CPU (incl. live web retrieval and constrained decoding); Code_6 export,
rating import, κ (checked by hand), adjudication sheet, all tables and figures. Dummy ratings were used
ONLY to test the κ code in a scratch folder; no rating file is shipped. The full-size models and full runs
happen on Colab.

## 2. Data and benchmark (Code_1) — #1, #2, #6, #13

* **Reference values cast losslessly to their declared type** (`cast_reference_values`): ToolBench stores
  example values as strings whatever the declared type (e.g. 11,108 NUMBER parameters with `"2"` vs 1,089 real
  numbers in one shard). The v1 finding "20 % of registry types are wrong" was largely this artifact.
  Switch it off in `config.json` to reproduce v1 behaviour.
* **Reference calls must be structurally consistent with their schema** (no undeclared names, no missing
  required parameter); inconsistent API-Bank / ToolBench answers are removed and counted. Type mismatches are
  kept (they are what adaptive type checking is for).
* **Schema variants**: a function defined with several different parameter lists keeps its most frequent definition.
* **Injector rewritten (in the core)** — every injected error is an error w.r.t. the schema:
  WRONG_FUNCTION name verified absent from the registry; EXTRA name verified absent from the schema;
  MISSING removes only a required, non-empty parameter; WRONG_TYPE only corrupts a parameter whose clean value
  matches its declared type and checks the new type is incompatible; **WRONG_VALUE = donor swap** (a real value of
  the same type from another call using the same parameter name — "Mumbai → Delhi"), old reversal / 2099 /
  negation kept only as fallback and labelled.
* **Bigger benchmark**: 1,000 clean + 300 per type = 2,800 (v1: 500 + 150 = 1,400) — tighter confidence intervals.
* **Two splits**: random stratified 30/70 (as v1) and **function-disjoint** (no function in both parts) (#1).
* **Multi-error benchmark**: 300 calls each with k = 1, 2, 3, 4 different error types, on test-side functions only (#2).
* **Real-LLM evaluation set**: ToolAlpaca eval_real + eval_simulated (21 tools, 214 queries with gold calls);
  OpenAPI schemas parsed into the registry (types, required, enum, min/max, default); tools split into
  *known* (half of their gold calls reserved for calibration) and *unseen*; `getDetails` (clarification) gold
  answers set aside (#9, #13).
* Before/after example per error type and all statistics saved (`examples_by_type.json`, `data_summary.json`).

## 3. Detection / correction method (core) — #4, #5, #6, #11, #12

* **Pipeline order** (#12): raw call → parse → *format check on the raw envelope* → normalise → function →
  extra → missing → type → value constraints → (optional) query grounding → classify → correct.
* **Adaptive type trust** (#11) replaces the skip list: trust per declaration from clean tune calls with back-off
  (function+param → param name → declared type → prior), enforced if trust ≥ τ; τ chosen on the tune part.
  Modes `strict`, `off`, `skiplist` (= v1) and `adaptive` are all reported.
* **Empty value = not set** (`empty_is_unset`): `""`/`null` are not type/enum-checked (ToolAlpaca gold answers use
  `"gender": ""`).
* **Query grounding** (#6): optional check that a value occurs in the user request, plus replacement by the best
  matching request span — the "schema + query" setting.
* **Function matcher** (#5): semantic weight and minimum score configurable, semantic-only / edit-only modes,
  weights selected on the tune part; similarity cache; C-speed edit distance via `rapidfuzz` when installed.
* **Corrector**: number of passes and fixes per pass configurable; every fix logged; each fix can be switched off.
* **Four separate outcome levels** (#4): detected / error type / schema-valid / exact match with the reference.

## 4. Evaluation — #3, #5, #8, #10, #15

* Precision, recall, F1, balanced accuracy, FPR/FNR, 7-class confusion matrix, per-class P/R/F1, macro-F1,
  multi-label metrics for multi-error calls, **bootstrap 95 % CIs**, exact McNemar and paired-bootstrap Δ (#3).
* Ablation with 19 variants incl. no edit distance / no semantic similarity / passes 1–8 (#5).
* Schema-noise robustness 0 / 5 / 10 / 20 / 30 % × 3 seeds (#10).
* Baselines (#8): JSON Schema validation (`jsonschema`), similarity-only correction, LLM verifier, LLM corrector,
  constrained decoding (Code_5), placeholder for a published detector.
* Runtime: per component mean / p50 / p95 / p99 + hardware and package versions in every result file (#15, #20).

## 5. Real LLM calls (Code_5) — #9, #13

* v1 prompt gave only the function name and 150 characters of description; v2 gives the full schemas of all the
  tool's functions, uses each model's chat template, and lets the model choose the function.
* 5 models (Qwen2.5 0.5B/1.5B/3B, TinyLlama 1.1B, Phi-3.5-mini) × free and constrained decoding; known vs unseen
  tools; out-of-registry control reproducing v1's "detection = 1.000" artifact.
* Generations cached per model in JSONL, resumable after a disconnect.

## 5b. Published baseline (Code_5b, added later) — #8

* SelfCheckGPT (Manakul et al., EMNLP 2023) adapted to tool calls: N sampled calls from the same model,
  hallucination score = 1 − mean similarity (0.5 × same function + 0.5 × Jaccard of (parameter, value) pairs).
* Threshold chosen on 30 % of the real queries, results on the other 70 %, same calls as the schema-only system;
  also "ours OR SelfCheck" and a comparison against the human labels (Code_6).
* Separate notebook: Code_1–Code_5 results are untouched; only Code_6 is re-run afterwards.

## 6. Human evaluation (Code_6) — #14

* v1 fell back to **simulated** ratings when no ratings existed. v2 has **no simulation**: without real ratings
  the human-evaluation numbers are simply not produced.
* Sheets for two raters on real calls, two passes (pass 1 hides the system's correction), guideline file,
  practice set, κ per question (binary, per error type, weighted κ for intent), adjudication sheet, and system
  metrics against the adjudicated human labels.

## 7. Removed from v1

* All `week N` naming in files and prints.
* Simulated human ratings (`Using SIMULATED ratings for paper draft`).
* The "oracle fallback" credit in retrieval (replaced by an explicit retrieval-recall upper bound).
* Hard-coded numbers in the figure script — figures and tables are built from the result files (Code_6).
