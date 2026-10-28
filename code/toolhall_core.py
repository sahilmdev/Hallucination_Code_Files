"""
toolhall_core.py -- shared code for the tool-call hallucination project.

Every notebook (Code_1 ... Code_6) imports this file from Google Drive, so
the validator, corrector, matcher, calibration and metrics exist in exactly
ONE place.  Change something here and every notebook picks it up on its next
fresh runtime.

Contents
  1. Constants and the error taxonomy
  2. Config, run-ID bookkeeping and checked JSON I/O
  3. Parsing, envelope (format) check and normalisation
  4. Schema registry (canonical types, strict lookup, noise injection)
  5. Adaptive type trust (replaces the old "skip list")
  6. Query grounding (value support in the user request)
  7. Validator (detection + error-type classification)
  8. Function matcher (semantic + edit distance, configurable weights)
  9. Corrector
 10. Reference-based labels (for scoring and for real LLM data)
 11. Splits (random stratified and function-disjoint)
 12. Metrics: binary, multi-class, multi-label, bootstrap CIs, McNemar
 13. Pipeline runner and per-example outcome record
 14. Timing and hardware information

Ground truth is NEVER passed to the validator, the corrector or the matcher.
It is only used by the scoring functions in sections 10, 12 and 13.
"""

import ast
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

CORE_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# 1. Constants and the error taxonomy
# ---------------------------------------------------------------------------

ERROR_TYPES = [
    'WRONG_FUNCTION', 'EXTRA_PARAMETER', 'WRONG_TYPE',
    'WRONG_VALUE', 'MISSING_PARAMETER', 'WRONG_FORMAT',
]
ALL_LABELS = ['NONE'] + ERROR_TYPES

# Order used to pick ONE "primary" error type when a call has several
# violations.  Envelope and function problems come first because nothing
# below them can be checked reliably until they are fixed.
PRIMARY_ORDER = ['WRONG_FORMAT', 'WRONG_FUNCTION', 'EXTRA_PARAMETER',
                 'MISSING_PARAMETER', 'WRONG_TYPE', 'WRONG_VALUE']

TAXONOMY = {
    "WRONG_FUNCTION": {
        "description": "The call names a function that does not exist in the schema registry.",
        "call_component": "function name",
        "schema_detectable": "yes (name lookup)",
        "example": "'Twitter.Search_new' when only 'Twitter.Search' exists",
    },
    "EXTRA_PARAMETER": {
        "description": "The call passes a parameter the function does not declare.",
        "call_component": "parameter set (surplus)",
        "schema_detectable": "yes (set difference)",
        "example": "adds 'verbose': 10 to a function with no 'verbose' parameter",
    },
    "MISSING_PARAMETER": {
        "description": "A parameter the schema marks as required is absent.",
        "call_component": "parameter set (deficit)",
        "schema_detectable": "yes, if the registry records which parameters are required",
        "example": "omits the required 'query' parameter",
    },
    "WRONG_TYPE": {
        "description": "A declared parameter has a value of an incompatible type.",
        "call_component": "parameter value type",
        "schema_detectable": "yes, if the declared type is reliable",
        "example": "'max_length': '4000' (string) where an integer is declared",
    },
    "WRONG_VALUE": {
        "description": "Right parameter and type, but the value does not serve the request.",
        "call_component": "parameter value content",
        "schema_detectable": "only for enum/min/max constraints; otherwise needs the query or retrieval",
        "example": "'city': 'Delhi' when the user asked for Mumbai",
    },
    "WRONG_FORMAT": {
        "description": "Function and arguments are present but in the wrong JSON envelope.",
        "call_component": "envelope / structure",
        "schema_detectable": "yes (raw envelope check before normalisation)",
        "example": "{'tool': ..., 'tool_input': ...} instead of {'function': ..., 'parameters': ...}",
    },
}

# ---------------------------------------------------------------------------
# 2. Config, run-ID bookkeeping and checked JSON I/O
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "run_id": "run_A",
    "data_version": "benchmark",
    "seed": 42,
    "drive_root": "/content/drive/MyDrive/hallucination_project",

    "benchmark": {
        "n_clean": 1000,
        "n_per_type": 300,
        "max_calls_per_function": 3,
        "multi_error_k": [1, 2, 3, 4],
        "n_per_k": 300,
        "tune_frac": 0.30,
        "wrong_value_mode": "donor_swap",
        "cast_reference_values": True,
    },

    "envelope": {
        "accepted_key_sets": [["function", "parameters"]],
    },

    "validator": {
        "check_format": True, "check_function": True, "check_extra": True,
        "check_missing": True, "check_type": True, "check_value": True,
        "check_grounding": False,
        "unknown_required_policy": "not_required",
        "empty_is_unset": True,
    },

    "type_trust": {
        "mode": "adaptive",
        "tau": 0.5,
        "tau_grid": [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        "min_obs": 3,
        "prior_agree": 1.0,
        "prior_disagree": 0.0,
    },

    "grounding": {
        "min_len": 3,
        "fuzzy_threshold": 0.85,
    },

    "matcher": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "use_semantic": True,
        "use_edit": True,
        "w_semantic": 0.75,
        "min_score": 0.55,
        "topk": 20,
        "w_semantic_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "min_score_grid": [0.45, 0.55, 0.65, 0.75],
    },

    "corrector": {
        "max_iterations": 3,
        "max_iterations_grid": [1, 2, 3, 5, 8],
        "fixes_per_iteration": 1,
        "fix_format": True, "fix_function": True, "fix_extra": True,
        "fix_missing": True, "fix_type": True, "fix_value": True,
        "fix_value_from_query": False,
    },

    "noise": {
        "rates": [0.0, 0.05, 0.10, 0.20, 0.30],
        "seeds": [0, 1, 2],
        "kinds": ["type_flip", "required_flip", "drop_param", "rename_param"],
    },

    "bootstrap": {"n": 1000, "alpha": 0.05},

    "models": {
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "bertscore_model": "roberta-large",
        "llm_baseline": "Qwen/Qwen2.5-1.5B-Instruct",
        "llm_baseline_limit": 350,
        "generation_models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/Phi-3.5-mini-instruct",
        ],
        "judge_model": "Qwen/Qwen2.5-3B-Instruct",
        "max_new_tokens": 256,
    },

    "real_eval": {
        "known_tool_frac": 0.5,
        "known_calibration_frac": 0.5,
    },

    "retrieval": {
        "max_results": 4,
        "nli_entail_threshold": 0.5,
        "query_modes": ["schema_names", "user_query"],
        "sleep_seconds": 0.5,
    },

    "human_eval": {
        "n_items": 200,
        "n_practice": 20,
    },
}


def deep_update(base, override):
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None):
    """Read config.json (if given) on top of DEFAULT_CONFIG."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            cfg = deep_update(cfg, json.load(f))
    return cfg


def config_hash(cfg):
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]


class Paths:
    """All Drive locations in one place.

    drive_root/
      code/             toolhall_core.py, config.json
      data/<data_version>/   Code_1 outputs (benchmark, registry, splits)
      runs/<run_id>/    Code_2..Code_6 outputs
      cache/            embeddings, retrieval cache, LLM generations
    """

    def __init__(self, cfg, root=None):
        self.root = root or cfg['drive_root']
        self.code = os.path.join(self.root, 'code')
        self.data = os.path.join(self.root, 'data', cfg['data_version'])
        self.run = os.path.join(self.root, 'runs', cfg['run_id'])
        self.cache = os.path.join(self.root, 'cache')
        for d in (self.code, self.data, self.run, self.cache):
            os.makedirs(d, exist_ok=True)

    def data_file(self, name):
        return os.path.join(self.data, name)

    def run_file(self, name):
        return os.path.join(self.run, name)

    def cache_file(self, name):
        return os.path.join(self.cache, name)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def save_json(obj, path, cfg, notebook, kind='run'):
    """Save with a _meta block so stale files can never be mixed in.

    kind='data' stamps data_version (Code_1 outputs);
    kind='run'  stamps run_id (Code_2..6 outputs).
    """
    meta = {
        'notebook': notebook,
        'kind': kind,
        'data_version': cfg['data_version'],
        'run_id': cfg['run_id'] if kind == 'run' else None,
        'config_hash': config_hash(cfg),
        'core_version': CORE_VERSION,
        'created_utc': _dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }
    payload = {'_meta': meta, 'data': obj}
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return path


def load_json(path, cfg, kind='run', check=True):
    """Load a file written by save_json and verify it belongs to this run."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing input: {path}\n  -> run the notebook that produces it first "
            f"(same run_id / data_version).")
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    if not (isinstance(payload, dict) and '_meta' in payload and 'data' in payload):
        raise ValueError(f"{path} was not written by save_json (old-format file?). "
                         f"Regenerate it with the current notebooks.")
    meta = payload['_meta']
    if check:
        if meta.get('data_version') != cfg['data_version']:
            raise ValueError(
                f"{os.path.basename(path)} has data_version={meta.get('data_version')!r} "
                f"but config says {cfg['data_version']!r}. Refusing to mix runs.")
        if kind == 'run' and meta.get('run_id') != cfg['run_id']:
            raise ValueError(
                f"{os.path.basename(path)} has run_id={meta.get('run_id')!r} "
                f"but config says {cfg['run_id']!r}. Refusing to mix runs.")
    return payload['data']


def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + '\n')


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # a half-written last line after a disconnect
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 3. Parsing, envelope (format) check and normalisation
# ---------------------------------------------------------------------------

def parse_raw_call(raw):
    """Turn a raw call (dict or string) into a dict, or None if impossible.

    Accepts JSON, Python-literal dicts, JSON inside ``` fences and the first
    {...} object inside free text (for LLM outputs).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', s, re.DOTALL)
    if fence:
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(fence.group(1))
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    obj = _first_balanced_object(s)
    if obj:
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(obj)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None


def _first_balanced_object(s):
    start = s.find('{')
    while start != -1:
        depth, in_str, esc, quote = 0, False, False, ''
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str, quote = True, ch
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        start = s.find('{', start + 1)
    return None


def envelope_keys(raw):
    """Top-level keys of the raw call, BEFORE any normalisation."""
    d = parse_raw_call(raw)
    return set(d.keys()) if isinstance(d, dict) else None


def envelope_ok(raw, accepted_key_sets):
    """True if the raw envelope is exactly one of the accepted shapes."""
    keys = envelope_keys(raw)
    if keys is None:
        return False
    accepted = {frozenset(k) for k in accepted_key_sets}
    return frozenset(keys) in accepted


_FUNC_KEYS = ('function', 'name', 'api_name', 'tool', 'action',
              'function_name', 'tool_name')
_PARAM_KEYS = ('parameters', 'arguments', 'args', 'action_input',
               'function_params', 'tool_input', 'input')


def _as_params(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        d = parse_raw_call(v)
        return d if isinstance(d, dict) else {}
    return {}


def normalize_tool_call(raw):
    """Map any supported envelope to {"function": str, "parameters": dict}.

    Handles the OpenAI-style nested shape
      {"type": "function", "function": {"name": ..., "arguments": "<json>"}}
    which the previous Code_2..Code_4 versions turned into a function name
    like "{'name': ...}".  Returns None only if the call cannot be parsed.
    """
    d = parse_raw_call(raw)
    if not isinstance(d, dict):
        return None
    inner = d.get('function')
    if isinstance(inner, dict):
        func = inner.get('name') or inner.get('function') or 'unknown'
        params = _as_params(inner.get('arguments', inner.get('parameters', {})))
        return {"function": str(func), "parameters": params}
    func = None
    for k in _FUNC_KEYS:
        if d.get(k):
            func = d[k]
            break
    params = {}
    for k in _PARAM_KEYS:
        if k in d and d[k] not in (None, ''):
            params = _as_params(d[k])
            if params or isinstance(d[k], dict):
                break
    if func is None:
        func = 'unknown'
    return {"function": str(func), "parameters": params}


def canonical_json(norm):
    return json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)


def calls_equal(a, b):
    """Same function and parameters after normalisation (envelope ignored)."""
    na, nb = normalize_tool_call(a), normalize_tool_call(b)
    if na is None or nb is None:
        return False
    return canonical_json(na) == canonical_json(nb)


def exact_strict(final_raw, gt_raw, accepted_key_sets=None):
    """The end-to-end success criterion.

    The final output must (a) be in an accepted envelope -- a strict executor
    cannot run {'tool': ..., 'tool_input': ...} -- and (b) equal the
    reference after normalisation.  The previous version normalised both
    sides first, so an uncorrected WRONG_FORMAT call counted as a success.
    """
    accepted = accepted_key_sets or DEFAULT_CONFIG['envelope']['accepted_key_sets']
    if final_raw is None or gt_raw is None:
        return False
    if not envelope_ok(final_raw, accepted):
        return False
    return calls_equal(final_raw, gt_raw)


# ---------------------------------------------------------------------------
# 4. Schema registry
# ---------------------------------------------------------------------------
#
# Registry file format (written by Code_1):
#   { "<function name>": {
#        "description": "...",
#        "source": "toolbench" | "apibank" | "toolalpaca_eval",
#        "parameters": { "<param>": {"type": "<canonical or null>",
#                                    "type_raw": "<as published>",
#                                    "required": true | false | null,
#                                    "description": "...",
#                                    "enum": [...]            (optional),
#                                    "minimum"/"maximum": num (optional) } } } }

CANONICAL_TYPES = ('string', 'integer', 'number', 'boolean', 'array', 'object')


def canonical_type(t):
    """Map published type strings (STRING, NUMBER, 'DATE (YYYY-MM-DD)', str ...)
    onto six canonical types.  Unknown types return None (= not checked).

    The previous version compared raw lowercased strings, so a declared
    'date (yyyy-mm-dd)' or 'enum' never matched any value and every such
    parameter was flagged WRONG_TYPE.
    """
    if t is None:
        return None
    s = str(t).strip().lower()
    if not s:
        return None
    if s in ('string', 'str', 'text', 'char', 'enum', 'uuid', 'email', 'url',
             'password', 'binary', 'byte') or s.startswith(('date', 'time', 'geopoint',
                                                            'enum', 'string')):
        return 'string'
    if s in ('integer', 'int', 'int32', 'int64', 'long', 'short'):
        return 'integer'
    if s in ('number', 'float', 'double', 'decimal', 'numeric'):
        return 'number'
    if s in ('boolean', 'bool'):
        return 'boolean'
    if s in ('array', 'list', 'tuple'):
        return 'array'
    if s in ('object', 'dict', 'json', 'map'):
        return 'object'
    return None


def value_type(v):
    """Canonical type of a Python value."""
    if isinstance(v, bool):
        return 'boolean'
    if isinstance(v, int):
        return 'integer'
    if isinstance(v, float):
        return 'number'
    if isinstance(v, str):
        return 'string'
    if isinstance(v, (list, tuple)):
        return 'array'
    if isinstance(v, dict):
        return 'object'
    if v is None:
        return 'null'
    return type(v).__name__


def types_compatible(declared, actual):
    """integer and number are interchangeable (JSON has one number type)."""
    if declared is None:
        return True
    if declared in ('integer', 'number') and actual in ('integer', 'number'):
        return True
    return declared == actual


class Registry:
    """Read-only view over the schema registry with strict lookup."""

    def __init__(self, raw, unknown_required_policy='not_required'):
        self.raw = raw
        self.unknown_required_policy = unknown_required_policy
        self.names = list(raw.keys())
        self._name_set = set(self.names)

    def __len__(self):
        return len(self.raw)

    def __contains__(self, name):
        return self.lookup(name) is not None

    def resolve(self, name):
        """Strict lookup: exact name, or the part after the first '.'.

        No fuzzy prefix matching, so 'get_weather_v2' is NOT resolved to
        'get_weather' (that would hide WRONG_FUNCTION hallucinations).
        """
        if not name:
            return None
        if name in self._name_set:
            return name
        if '.' in name:
            api = name.split('.', 1)[1]
            if api in self._name_set:
                return api
        return None

    def lookup(self, name):
        r = self.resolve(name)
        return self.raw[r] if r is not None else None

    def params(self, name):
        e = self.lookup(name)
        if not isinstance(e, dict):
            return {}
        p = e.get('parameters', {})
        return p if isinstance(p, dict) else {}

    def description(self, name):
        e = self.lookup(name)
        return (e or {}).get('description', '') if isinstance(e, dict) else ''

    def is_required(self, spec):
        if not isinstance(spec, dict):
            return False
        r = spec.get('required')
        if r is None:
            return self.unknown_required_policy == 'required'
        return bool(r)

    def required_params(self, name):
        return [p for p, s in self.params(name).items() if self.is_required(s)]

    def copy_raw(self):
        return copy.deepcopy(self.raw)

    def hash(self):
        return hashlib.sha1(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]


def registry_without(raw, function_names):
    """Copy of a raw registry with some functions removed (out-of-registry control)."""
    drop = set(function_names)
    return {k: v for k, v in raw.items() if k not in drop}


def corrupt_registry(raw, rate, seed, kinds=("type_flip", "required_flip",
                                             "drop_param", "rename_param"),
                     only_functions=None):
    """Return (corrupted_copy, log).  `rate` = fraction of (function, param)
    declarations that receive exactly one corruption, chosen uniformly from
    `kinds`.  Used for the schema-quality robustness experiment (#10).

    type_flip     : declared type replaced by a different canonical type
    required_flip : required True <-> False
    drop_param    : declaration removed (calls using it now look EXTRA)
    rename_param  : declaration renamed (call's name looks EXTRA, real one MISSING if required)
    """
    rng = random.Random(seed)
    out = copy.deepcopy(raw)
    decls = []
    for fn, entry in out.items():
        if only_functions is not None and fn not in only_functions:
            continue
        for p in list((entry.get('parameters') or {}).keys()):
            decls.append((fn, p))
    n = int(round(rate * len(decls)))
    chosen = rng.sample(decls, n) if n > 0 else []
    log = Counter()
    for fn, p in chosen:
        params = out[fn]['parameters']
        if p not in params:
            continue
        kind = rng.choice(list(kinds))
        spec = params[p]
        if kind == 'type_flip':
            cur = spec.get('type')
            choices = [t for t in CANONICAL_TYPES if t != cur
                       and not (cur in ('integer', 'number') and t in ('integer', 'number'))]
            spec['type'] = rng.choice(choices)
        elif kind == 'required_flip':
            spec['required'] = not bool(spec.get('required'))
        elif kind == 'drop_param':
            del params[p]
        elif kind == 'rename_param':
            new = p + '_' + rng.choice(['id', 'value', 'name', 'v2', 'param'])
            if new not in params:
                params[new] = params.pop(p)
        log[kind] += 1
    return out, {'n_declarations': len(decls), 'n_corrupted': len(chosen),
                 'by_kind': dict(log), 'rate': rate, 'seed': seed}


# ---------------------------------------------------------------------------
# 5. Adaptive type trust
# ---------------------------------------------------------------------------

class TypeTrust:
    """How much each declared type can be trusted, learned from CLEAN tune calls.

    Why conflicts happen: the published schemas are hand-written and often
    declare e.g. NUMBER for a field that every real call passes as a string.
    How a conflict is detected: a clean (known-correct) tune call whose value
    type is incompatible with the declared type.
    When type validation is relaxed: the trust estimate for a declaration is
    below the threshold tau.

    Trust is estimated at the most specific level that has at least
    `min_obs` observations, backing off in this order:
        (function, param)  ->  param name  ->  declared type  ->  prior
    so functions never seen during tuning still get an estimate (needed for
    the function-disjoint split).  Estimate = (agree + a) / (seen + a + b).

    mode:
      'strict'   : always enforce declared types (no calibration)
      'off'      : never check types
      'skiplist' : the previous behaviour -- skip a (function, param) pair if
                   ANY clean tune call contradicts it, otherwise enforce
      'adaptive' : enforce only if trust >= tau
    """

    def __init__(self, mode='adaptive', tau=0.5, min_obs=3, prior_agree=1.0,
                 prior_disagree=0.0):
        self.mode = mode
        self.tau = tau
        self.min_obs = min_obs
        self.a = prior_agree
        self.b = prior_disagree
        self.pair = defaultdict(lambda: [0, 0])     # (fn, p) -> [agree, seen]
        self.pname = defaultdict(lambda: [0, 0])    # p -> [agree, seen]
        self.dtype = defaultdict(lambda: [0, 0])    # declared type -> [agree, seen]
        self.fitted_on = 0

    def fit(self, clean_calls, registry):
        """clean_calls: iterable of normalised calls known to be correct."""
        for call in clean_calls:
            if not call:
                continue
            fn = registry.resolve(call.get('function'))
            if fn is None:
                continue
            schema = registry.params(fn)
            for p, v in (call.get('parameters') or {}).items():
                spec = schema.get(p)
                if not isinstance(spec, dict):
                    continue
                declared = spec.get('type')
                if declared is None:
                    continue
                ok = types_compatible(declared, value_type(v))
                for table, key in ((self.pair, (fn, p)), (self.pname, p),
                                   (self.dtype, declared)):
                    table[key][1] += 1
                    table[key][0] += int(ok)
            self.fitted_on += 1
        return self

    def _est(self, cell):
        agree, seen = cell
        return (agree + self.a) / (seen + self.a + self.b) if (seen + self.a + self.b) else 1.0

    def trust(self, fn, p, declared):
        for table, key in ((self.pair, (fn, p)), (self.pname, p), (self.dtype, declared)):
            if key in table and table[key][1] >= self.min_obs:
                return self._est(table[key])
        return 1.0

    def level(self, fn, p, declared):
        for name, table, key in (('pair', self.pair, (fn, p)), ('param', self.pname, p),
                                 ('type', self.dtype, declared)):
            if key in table and table[key][1] >= self.min_obs:
                return name
        return 'prior'

    def enforce(self, fn, p, declared):
        if declared is None or self.mode == 'off':
            return False
        if self.mode == 'strict':
            return True
        if self.mode == 'skiplist':
            cell = self.pair.get((fn, p))
            return not (cell and cell[0] < cell[1])
        return self.trust(fn, p, declared) >= self.tau

    def summary(self):
        contradicted_pairs = sum(1 for c in self.pair.values() if c[0] < c[1])
        return {
            'mode': self.mode, 'tau': self.tau, 'min_obs': self.min_obs,
            'clean_calls_used': self.fitted_on,
            'pairs_seen': len(self.pair),
            'pairs_contradicted': contradicted_pairs,
            'param_names_seen': len(self.pname),
            'declared_type_agreement': {
                t: {'agree': c[0], 'seen': c[1],
                    'rate': round(c[0] / c[1], 4) if c[1] else None}
                for t, c in sorted(self.dtype.items())},
        }


# ---------------------------------------------------------------------------
# 6. Query grounding (is a value supported by the user's request?)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.@:/]*")


def _norm_text(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] if s1[i - 1] == s2[j - 1] else \
                1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]


try:                                   # C implementation, ~100x faster; same definition
    from rapidfuzz.distance import Levenshtein as _RFLev
except Exception:                      # pragma: no cover
    _RFLev = None


def edit_norm(a, b):
    """Levenshtein distance / length of the longer string (0 = identical)."""
    a, b = str(a), str(b)
    if not a and not b:
        return 0.0
    if _RFLev is not None:
        return _RFLev.distance(a, b) / max(len(a), len(b), 1)
    return edit_distance(a, b) / max(len(a), len(b), 1)


def value_grounded(value, query, min_len=3):
    """True / False / None (None = not checkable: too short, bool, empty, container).

    A value is grounded if its normalised text occurs in the normalised
    query; numbers also match their digit string.
    """
    if value is None or isinstance(value, (bool, dict, list)):
        return None
    if isinstance(value, (int, float)):
        txt = str(value)
        if txt.endswith('.0'):
            txt = txt[:-2]
    else:
        txt = str(value)
    t = _norm_text(txt).strip('\'"')
    if len(t) < min_len:
        return None
    q = _norm_text(query or '')
    if not q:
        return None
    return t in q


def query_candidates(query):
    """Spans in the query that could be parameter values."""
    if not query:
        return []
    c = []
    c += re.findall(r'"([^"]{1,80})"', query)
    c += re.findall(r"'([^']{2,80})'", query)
    c += re.findall(r'\b\d{4}-\d{2}-\d{2}\b', query)
    c += re.findall(r'\b\d+(?:\.\d+)?\b', query)
    c += re.findall(r'\b(?:[A-Z][a-zA-Z0-9\-]+)(?:\s+[A-Z][a-zA-Z0-9\-]+)*\b', query)
    c += re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', query)
    c += re.findall(r'https?://\S+', query)
    seen, out = set(), []
    for x in c:
        x = x.strip().strip('.,;:')
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def _shape(s):
    s = str(s)
    return ''.join('9' if ch.isdigit() else ('a' if ch.isalpha() else ch) for ch in s)


def best_query_value(current, query, declared_type):
    """Pick the query span most likely to be the intended value.

    Heuristic: same canonical type after casting, then most similar
    character shape to the current value, then closest length.
    Returns None if the query offers no candidate of the right type.
    """
    cands = query_candidates(query)
    scored = []
    for c in cands:
        v = c
        if declared_type in ('integer', 'number'):
            try:
                f = float(c.replace(',', ''))
                v = int(f) if f.is_integer() and declared_type == 'integer' else f
            except ValueError:
                continue
        elif declared_type not in (None, 'string'):
            continue
        sim = 1.0 - edit_norm(_shape(current), _shape(c))
        scored.append((sim, -abs(len(str(c)) - len(str(current))), v))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ---------------------------------------------------------------------------
# 7. Validator (detection + error-type classification)
# ---------------------------------------------------------------------------

_SEVERITY = {'WRONG_FORMAT': 'CRITICAL', 'WRONG_FUNCTION': 'CRITICAL',
             'EXTRA_PARAMETER': 'HIGH', 'MISSING_PARAMETER': 'HIGH',
             'WRONG_TYPE': 'HIGH', 'WRONG_VALUE': 'MEDIUM'}


def _viol(vtype, stage, detail, **kw):
    d = {'type': vtype, 'severity': _SEVERITY.get(vtype, 'MEDIUM'),
         'stage': stage, 'detail': detail}
    d.update(kw)
    return d


class Validator:
    """Schema-only validator.  Pipeline (order matters, see #12):

      1. parse       raw text -> dict                   (fail => WRONG_FORMAT/parse)
      2. format      raw envelope keys vs accepted shapes, BEFORE normalising
      3. normalise   any envelope -> {function, parameters}
      4. function    strict registry lookup             (fail => WRONG_FUNCTION, stop)
      5. extra       undeclared parameters
      6. missing     required parameters absent
      7. type        declared vs actual type, gated by TypeTrust
      8. value       enum / minimum / maximum from the schema
      9. grounding   (optional) value not supported by the user query

    Ground truth is never an input.
    """

    def __init__(self, registry, cfg_validator=None, accepted_key_sets=None,
                 type_trust=None, grounding_cfg=None):
        self.reg = registry
        self.cfg = deep_update(DEFAULT_CONFIG['validator'], cfg_validator or {})
        self.accepted = accepted_key_sets or DEFAULT_CONFIG['envelope']['accepted_key_sets']
        self.trust = type_trust or TypeTrust(mode='strict')
        self.gcfg = deep_update(DEFAULT_CONFIG['grounding'], grounding_cfg or {})

    def validate(self, raw, query=None):
        c = self.cfg
        out = []
        parsed = parse_raw_call(raw)
        if parsed is None:
            if c['check_format']:
                out.append(_viol('WRONG_FORMAT', 'parse', 'call could not be parsed as JSON',
                                 subtype='unparseable'))
            return out
        if c['check_format'] and not envelope_ok(parsed, self.accepted):
            out.append(_viol('WRONG_FORMAT', 'format',
                             f"envelope keys {sorted(parsed.keys())} not in accepted shapes",
                             keys=sorted(parsed.keys())))
        norm = normalize_tool_call(parsed)
        func, params = norm['function'], norm['parameters']

        fn = self.reg.resolve(func)
        if fn is None:
            if c['check_function']:
                out.append(_viol('WRONG_FUNCTION', 'function',
                                 f"function '{func}' is not in the registry", got=func))
            return out
        schema = self.reg.params(fn)

        if c['check_extra']:
            for p in params:
                if p not in schema:
                    out.append(_viol('EXTRA_PARAMETER', 'extra',
                                     f"'{p}' is not declared by '{fn}'", param=p,
                                     value=params[p]))
        if c['check_missing']:
            for p, spec in schema.items():
                if p not in params and self.reg.is_required(spec):
                    out.append(_viol('MISSING_PARAMETER', 'missing',
                                     f"required '{p}' of '{fn}' is absent", param=p))
        # An empty string / null for a parameter means "not set" in the source
        # datasets (e.g. ToolAlpaca gold answers use "gender": ""), so such
        # values are not type- or enum-checked.
        def _unset(v):
            return c.get('empty_is_unset', True) and (v is None or v == '')

        if c['check_type']:
            for p, v in params.items():
                spec = schema.get(p)
                if not isinstance(spec, dict) or _unset(v):
                    continue
                declared = spec.get('type')
                if not self.trust.enforce(fn, p, declared):
                    continue
                actual = value_type(v)
                if not types_compatible(declared, actual):
                    out.append(_viol('WRONG_TYPE', 'type',
                                     f"'{p}' declared {declared}, got {actual}",
                                     param=p, expected_type=declared, got_type=actual,
                                     gen_value=v,
                                     trust=round(self.trust.trust(fn, p, declared), 4)))
        if c['check_value']:
            for p, v in params.items():
                spec = schema.get(p)
                if isinstance(spec, dict) and not _unset(v):
                    viol = self._value_constraint(p, v, spec)
                    if viol:
                        out.append(viol)
        if c.get('check_grounding') and query:
            for p, v in params.items():
                if p not in schema:
                    continue
                g = value_grounded(v, query, self.gcfg['min_len'])
                if g is False:
                    out.append(_viol('WRONG_VALUE', 'grounding',
                                     f"value of '{p}' does not occur in the user request",
                                     param=p, got=v, subtype='ungrounded'))
        return out

    @staticmethod
    def _value_constraint(p, v, spec):
        allowed = spec.get('enum')
        if isinstance(allowed, list) and allowed:
            if str(v) not in [str(a) for a in allowed]:
                return _viol('WRONG_VALUE', 'value', f"'{p}'={v!r} not in enum",
                             param=p, got=v, allowed=[str(a) for a in allowed],
                             subtype='enum')
            return None
        mn, mx = spec.get('minimum'), spec.get('maximum')
        if mn is None and mx is None:
            return None
        try:
            num = float(v)
        except (TypeError, ValueError):
            return None
        if mn is not None and num < float(mn):
            return _viol('WRONG_VALUE', 'value', f"'{p}'={v} below minimum {mn}",
                         param=p, got=v, minimum=mn, subtype='range')
        if mx is not None and num > float(mx):
            return _viol('WRONG_VALUE', 'value', f"'{p}'={v} above maximum {mx}",
                         param=p, got=v, maximum=mx, subtype='range')
        return None

    @staticmethod
    def flagged(violations):
        return len(violations) > 0

    @staticmethod
    def predicted_types(violations):
        return sorted({v['type'] for v in violations}, key=PRIMARY_ORDER.index)

    @staticmethod
    def primary_type(violations):
        t = Validator.predicted_types(violations)
        return t[0] if t else 'NONE'


# ---------------------------------------------------------------------------
# 8. Function matcher
# ---------------------------------------------------------------------------

class FunctionMatcher:
    """Repairs a hallucinated function name by searching the registry.

    score(candidate) = w_sem * cosine(embed(name), embed(candidate text))
                     + (1 - w_sem) * (1 - normalised edit distance)
    Accept the best candidate if score >= min_score.

    use_semantic=False -> edit distance only (w_sem forced to 0)
    use_edit=False     -> semantic only    (w_sem forced to 1)
    The weights are selected on the tune split in Code_3 (#5).
    """

    def __init__(self, registry, model_name=DEFAULT_CONFIG['matcher']['model_name'],
                 cache_dir=None, use_semantic=True, use_edit=True, w_semantic=0.75,
                 min_score=0.55, topk=20, verbose=True, shared=None):
        self.reg = registry
        self.names = list(registry.names)
        self.use_semantic = use_semantic
        self.use_edit = use_edit
        self.w_sem = w_semantic
        self.min_score = min_score
        self.topk = topk
        self.model, self.emb = None, None
        # similarity cache shared by every configured() copy: the raw cosine and
        # edit scores of a name do not depend on the weights, so a weight sweep
        # computes them once per name.
        self._sim_cache = {}
        if shared is not None and use_semantic:
            # reuse the model and embeddings of another matcher (same registry)
            self.model, self.emb = shared.model, shared.emb
            if self.emb is None:
                self.use_semantic = False
            return
        if not use_semantic or not self.names:
            self.use_semantic = False
            return
        try:
            from sentence_transformers import SentenceTransformer
            texts = [self._describe(n) for n in self.names]
            key = hashlib.sha1(("\n".join(texts) + model_name).encode('utf-8')).hexdigest()[:12]
            path = os.path.join(cache_dir, f'function_embeddings_{key}.npz') if cache_dir else None
            if path and os.path.exists(path):
                data = np.load(path, allow_pickle=True)
                if list(data['names']) == self.names:
                    self.emb = data['emb']
                    if verbose:
                        print(f"FunctionMatcher: cached embeddings loaded ({len(self.names)})")
            self.model = SentenceTransformer(model_name)
            if self.emb is None:
                if verbose:
                    print(f"FunctionMatcher: embedding {len(self.names)} functions ...")
                self.emb = self.model.encode(texts, batch_size=256, show_progress_bar=verbose,
                                             convert_to_numpy=True, normalize_embeddings=True)
                if path:
                    np.savez_compressed(path, names=np.array(self.names, dtype=object),
                                        emb=self.emb)
        except Exception as e:
            print(f"FunctionMatcher: semantic mode unavailable ({type(e).__name__}: {e}); "
                  f"edit distance only")
            self.use_semantic = False
            self.model, self.emb = None, None

    def _describe(self, name):
        text = name.replace('_', ' ').replace('.', ' ')
        desc = self.reg.description(name)
        if desc and str(desc).strip():
            text += ' : ' + str(desc).strip()[:200]
        plist = list(self.reg.params(name).keys())
        if plist:
            text += ' | parameters: ' + ', '.join(plist[:10])
        return text

    def fresh(self):
        """Copy with an EMPTY similarity cache -- used for honest latency measurement."""
        m = self.configured()
        m._sim_cache = {}
        return m

    def configured(self, **kw):
        """A copy with different weights that shares the embedding index."""
        m = FunctionMatcher.__new__(FunctionMatcher)
        m.__dict__.update(self.__dict__)
        for k, v in kw.items():
            setattr(m, k, v)
        if m.emb is None:
            m.use_semantic = False
        return m

    def _semantic_top(self, name):
        key = ('sem', name, self.topk)
        if key not in self._sim_cache:
            q = self.model.encode([name.replace('_', ' ').replace('.', ' ')],
                                  convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = self.emb @ q
            idx = np.argsort(-sims)[:self.topk]
            self._sim_cache[key] = [(self.names[int(i)], float(sims[int(i)])) for i in idx]
        return self._sim_cache[key]

    def _edit_all(self, name):
        key = ('edit', name)
        if key not in self._sim_cache:
            self._sim_cache[key] = [(c, 1.0 - edit_norm(name, c)) for c in self.names
                                    if abs(len(c) - len(name)) <= 12]
        return self._sim_cache[key]

    def score_candidates(self, name):
        sem_on = self.use_semantic and self.emb is not None
        edit_on = self.use_edit
        if not sem_on and not edit_on:
            return []
        if sem_on:
            w = self.w_sem if edit_on else 1.0
            return [(c, w * sim + (1.0 - w) * ((1.0 - edit_norm(name, c)) if edit_on else 0.0))
                    for c, sim in self._semantic_top(name)]
        return self._edit_all(name)

    def match(self, name):
        if not name or not self.names:
            return None
        cands = self.score_candidates(name)
        if not cands:
            return None
        best, score = max(cands, key=lambda x: x[1])
        return best if score >= self.min_score else None


# ---------------------------------------------------------------------------
# 9. Corrector
# ---------------------------------------------------------------------------

def placeholder_for(ptype):
    return {'string': 'example_value', 'integer': 0, 'number': 0.0,
            'boolean': False, 'array': [], 'object': {}}.get(ptype, 'unknown')


def safe_cast(value, target):
    """Cast to the declared canonical type, or None if impossible.

    '5' -> 5 (int), never 5.0 -- the earlier float cast made exact match fail.
    """
    try:
        if target == 'integer':
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                c = value.replace(',', '').strip()
                f = float(c)
                return int(f) if f.is_integer() else None
            if isinstance(value, float):
                return int(value) if value.is_integer() else None
            if isinstance(value, int):
                return value
            if isinstance(value, list) and len(value) == 1:
                return safe_cast(value[0], target)
            return None
        if target == 'number':
            if isinstance(value, list) and len(value) == 1:
                return safe_cast(value[0], target)
            f = float(str(value).replace(',', '').strip())
            return int(f) if f.is_integer() else f
        if target == 'string':
            if isinstance(value, list) and len(value) == 1:
                return str(value[0])
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)
        if target == 'boolean':
            if isinstance(value, str):
                lv = value.strip().lower()
                if lv in ('true', '1', 'yes', 'on'):
                    return True
                if lv in ('false', '0', 'no', 'off'):
                    return False
                return None
            return bool(value)
        if target == 'array':
            if isinstance(value, str):
                p = parse_json_maybe(value)
                return p if isinstance(p, list) else [value]
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]
        if target == 'object':
            if isinstance(value, str):
                p = parse_json_maybe(value)
                return p if isinstance(p, dict) else None
            return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None
    return None


def parse_json_maybe(s):
    try:
        return json.loads(s)
    except Exception:
        return None


class Corrector:
    """Repairs a flagged call using only the registry (and optionally the query).

    One violation is fixed per iteration (fixes_per_iteration=1, the previous
    behaviour) or all fixable violations per iteration ('all'); the call is
    re-validated after every iteration.  Violations are handled in severity
    order: format -> function -> extra/missing/type -> value.

    Values the corrector has to INVENT (schema placeholders) are recorded in
    meta['uncertain_params'] -- the system knows they are guesses, which is
    what the retrieval step later targets.
    """

    def __init__(self, registry, validator, matcher=None, cfg_corrector=None):
        self.reg = registry
        self.validator = validator
        self.matcher = matcher
        self.cfg = deep_update(DEFAULT_CONFIG['corrector'], cfg_corrector or {})

    def correct(self, raw, query=None, max_iterations=None):
        c = self.cfg
        max_iter = c['max_iterations'] if max_iterations is None else max_iterations
        meta = {'iterations': 0, 'actions': [], 'uncertain_params': [],
                'flagged_for_retrieval': [], 'remaining_violations': [],
                'schema_valid': False}
        norm = normalize_tool_call(raw)
        if norm is None:
            meta['reason'] = 'unparseable_input'
            return None, meta
        # With fix_format on, repair always starts from the normalised
        # {function, parameters} envelope (this IS the format fix).  With it
        # off, a call in a wrong envelope keeps that envelope in the output
        # and therefore cannot count as an exact success.
        env_bad = not envelope_ok(raw, self.validator.accepted)
        meta['envelope_fixed'] = bool(env_bad and c['fix_format'])
        meta['envelope_unfixed'] = bool(env_bad and not c['fix_format'])
        if meta['envelope_fixed']:
            meta['actions'].append({'iteration': 0, 'type': 'WRONG_FORMAT',
                                    'action': 'restructured envelope to {function, parameters}'})
        working = copy.deepcopy(norm)
        for it in range(max_iter):
            viols = self.validator.validate(working, query=query)
            meta['iterations'] = it + 1
            if not viols:
                break
            viols = sorted(viols, key=lambda v: PRIMARY_ORDER.index(v['type']))
            fixed = 0
            for v in viols:
                act = self._fix(working, v, meta, query)
                if act:
                    meta['actions'].append({'iteration': it + 1, 'type': v['type'],
                                            'action': act})
                    fixed += 1
                    if c['fixes_per_iteration'] == 1:
                        break
                    if v['type'] == 'WRONG_FUNCTION':
                        break          # params must be re-checked against the new function
            if fixed == 0:
                break
        final = self.validator.validate(working, query=query)
        meta['remaining_violations'] = [v['type'] for v in final]
        meta['schema_valid'] = len(final) == 0
        for v in final:
            if v['type'] == 'WRONG_VALUE':
                p = v.get('param')
                if p and p not in [f['param'] for f in meta['flagged_for_retrieval']]:
                    meta['flagged_for_retrieval'].append(
                        {'param': p, 'reason': f"unresolved value ({v.get('subtype')})"})
        for p in meta['uncertain_params']:
            if p in working.get('parameters', {}) and \
                    p not in [f['param'] for f in meta['flagged_for_retrieval']]:
                meta['flagged_for_retrieval'].append(
                    {'param': p, 'reason': 'value invented by the corrector (placeholder)'})
        return working, meta

    def _fix(self, working, v, meta, query):
        c = self.cfg
        vt = v['type']
        params = working.setdefault('parameters', {})
        if vt == 'WRONG_FORMAT':
            if not c['fix_format']:
                return None
            # `working` is already the normalised envelope; the violation can
            # only re-appear for a call that is unparseable, which cannot be fixed.
            return None
        if vt == 'WRONG_FUNCTION':
            if not c['fix_function'] or self.matcher is None:
                return None
            got = v.get('got', '')
            cand = self.matcher.match(got)
            if cand and cand != got:
                working['function'] = cand
                return f"function '{got}' -> '{cand}'"
            return None
        if vt == 'EXTRA_PARAMETER':
            if not c['fix_extra']:
                return None
            p = v.get('param')
            if p in params:
                del params[p]
                return f"removed undeclared '{p}'"
            return None
        if vt == 'MISSING_PARAMETER':
            if not c['fix_missing']:
                return None
            p = v.get('param')
            spec = self.reg.params(working.get('function')).get(p)
            if not isinstance(spec, dict):
                return None
            if spec.get('default') is not None:
                params[p] = spec['default']
                return f"inserted '{p}' = schema default"
            if c.get('fix_value_from_query') and query:
                qv = best_query_value('', query, spec.get('type'))
                if qv is not None:
                    params[p] = qv
                    if p not in meta['uncertain_params']:
                        meta['uncertain_params'].append(p)
                    return f"inserted '{p}' = value from query"
            params[p] = placeholder_for(spec.get('type'))
            if p not in meta['uncertain_params']:
                meta['uncertain_params'].append(p)
            return f"inserted '{p}' = {spec.get('type')} placeholder"
        if vt == 'WRONG_TYPE':
            if not c['fix_type']:
                return None
            p = v.get('param')
            if p not in params:
                return None
            cast = safe_cast(params[p], v.get('expected_type'))
            if cast is not None and value_type(cast) != value_type(params[p]):
                params[p] = cast
                return f"cast '{p}' -> {v.get('expected_type')}"
            return None
        if vt == 'WRONG_VALUE':
            if not c['fix_value']:
                return None
            p = v.get('param')
            if p not in params:
                return None
            if v.get('subtype') == 'enum' and v.get('allowed'):
                cur = str(params[p])
                best = min(v['allowed'], key=lambda a: edit_norm(cur, a))
                params[p] = best
                return f"snapped '{p}' to enum value"
            if v.get('subtype') == 'ungrounded' and c.get('fix_value_from_query') and query:
                spec = self.reg.params(working.get('function')).get(p, {})
                qv = best_query_value(params[p], query, spec.get('type'))
                if qv is not None and str(qv) != str(params[p]):
                    params[p] = qv
                    return f"replaced '{p}' with a value from the query"
            return None
        return None


# ---------------------------------------------------------------------------
# 10. Reference-based labels
# ---------------------------------------------------------------------------

def reference_labels(gen_raw, gt_raw, registry=None, accepted_key_sets=None):
    """Error types of `gen_raw` relative to the reference call `gt_raw`.

    Used (a) to sanity-check injected benchmark labels and (b) as automatic
    ground truth for real LLM outputs (later replaced by human labels for the
    annotated subset).  Returns (labels:list, details:dict).

    WRONG_FUNCTION is split into subtypes because real models also pick an
    EXISTING but wrong function, which a schema cannot detect:
       'not_in_registry' | 'wrong_existing_function'
    """
    accepted = accepted_key_sets or DEFAULT_CONFIG['envelope']['accepted_key_sets']
    labels, det = set(), {}
    parsed = parse_raw_call(gen_raw)
    gt = normalize_tool_call(gt_raw)
    if parsed is None:
        return ['WRONG_FORMAT'], {'WRONG_FORMAT': 'unparseable'}
    if not envelope_ok(parsed, accepted):
        labels.add('WRONG_FORMAT')
        det['WRONG_FORMAT'] = 'envelope'
    gen = normalize_tool_call(parsed)
    if gt is None:
        return sorted(labels, key=PRIMARY_ORDER.index), det
    if gen['function'] != gt['function']:
        labels.add('WRONG_FUNCTION')
        if registry is not None and registry.resolve(gen['function']) is None:
            det['WRONG_FUNCTION'] = 'not_in_registry'
        else:
            det['WRONG_FUNCTION'] = 'wrong_existing_function'
        return sorted(labels, key=PRIMARY_ORDER.index), det
    gp, rp = gen['parameters'], gt['parameters']
    schema = registry.params(gt['function']) if registry is not None else {}
    extra = set(gp) - set(rp)
    missing = set(rp) - set(gp)
    for p in extra:
        if registry is not None and p in schema:
            labels.add('WRONG_VALUE')
            det.setdefault('WRONG_VALUE', []).append(f'{p}:declared_optional_not_in_reference')
        else:
            labels.add('EXTRA_PARAMETER')
            det.setdefault('EXTRA_PARAMETER', []).append(p)
    for p in missing:
        labels.add('MISSING_PARAMETER')
        det.setdefault('MISSING_PARAMETER', []).append(p)
    for p in set(gp) & set(rp):
        a, b = gp[p], rp[p]
        ta, tb = value_type(a), value_type(b)
        if not types_compatible(tb, ta):        # integer/number count as the same type
            labels.add('WRONG_TYPE')
            det.setdefault('WRONG_TYPE', []).append(p)
            continue
        if canonical_json(a) != canonical_json(b):
            labels.add('WRONG_VALUE')
            det.setdefault('WRONG_VALUE', []).append(p)
    return sorted(labels, key=PRIMARY_ORDER.index), det


# ---------------------------------------------------------------------------
# 11. Splits
# ---------------------------------------------------------------------------

def label_key(ex):
    """Stratification key: the single label, or 'MULTI_k' for multi-error."""
    labels = ex.get('error_types') or []
    if not labels:
        return 'NONE'
    return labels[0] if len(labels) == 1 else f"MULTI_{len(labels)}"


def split_random(examples, tune_frac=0.30, seed=42):
    """Stratified by label; the same function CAN appear in tune and test."""
    by = defaultdict(list)
    for i, ex in enumerate(examples):
        by[label_key(ex)].append(i)
    tune, test = [], []
    for k in sorted(by):
        idx = sorted(by[k])
        random.Random(seed).shuffle(idx)
        n = max(1, int(round(tune_frac * len(idx))))
        tune += idx[:n]
        test += idx[n:]
    return sorted(tune), sorted(test)


def split_function_disjoint(examples, tune_frac=0.30, seed=42):
    """Whole functions go to tune OR test, never both (#1).

    Functions are shuffled and assigned greedily so that each label reaches
    roughly `tune_frac` of its examples in tune.
    """
    func_of = [ex['function_key'] for ex in examples]
    by_func = defaultdict(list)
    for i, f in enumerate(func_of):
        by_func[f].append(i)
    funcs = sorted(by_func)
    random.Random(seed).shuffle(funcs)
    total = Counter(label_key(ex) for ex in examples)
    tune_count = Counter()
    tune_funcs = set()
    for f in funcs:
        labels_f = Counter(label_key(examples[i]) for i in by_func[f])
        # add the function to tune if it helps a label that is still under target
        need = any(tune_count[k] + n <= tune_frac * total[k] + 0.5 for k, n in labels_f.items())
        over = any(tune_count[k] + n > tune_frac * total[k] * 1.15 + 1 for k, n in labels_f.items())
        if need and not over:
            tune_funcs.add(f)
            tune_count.update(labels_f)
    tune = sorted(i for f in tune_funcs for i in by_func[f])
    test = sorted(i for f in funcs if f not in tune_funcs for i in by_func[f])
    return tune, test, sorted(tune_funcs)


def split_report(examples, tune, test):
    ft = {examples[i]['function_key'] for i in tune}
    fs = {examples[i]['function_key'] for i in test}
    return {
        'n_tune': len(tune), 'n_test': len(test),
        'tune_by_label': dict(sorted(Counter(label_key(examples[i]) for i in tune).items())),
        'test_by_label': dict(sorted(Counter(label_key(examples[i]) for i in test).items())),
        'functions_tune': len(ft), 'functions_test': len(fs),
        'functions_in_both': len(ft & fs),
        'test_examples_with_function_seen_in_tune':
            sum(1 for i in test if examples[i]['function_key'] in ft),
    }


# ---------------------------------------------------------------------------
# 12. Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred):
    """y = 1 means 'hallucinated'."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': (tp + tn) / n if n else 0.0,
        'precision': prec, 'recall': rec, 'f1': f1,
        'balanced_accuracy': (tpr + tnr) / 2,
        'fpr': fp / (fp + tn) if fp + tn else 0.0,
        'fnr': fn / (fn + tp) if fn + tp else 0.0,
        'specificity': tnr,
    }


def multiclass_metrics(y_true, y_pred, labels=ALL_LABELS):
    """Confusion matrix + per-class P/R/F1 + macro-F1 + balanced accuracy."""
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    per = {}
    f1s, recalls = [], []
    for l, i in idx.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = {'precision': p, 'recall': r, 'f1': f, 'support': support}
        if support > 0:
            f1s.append(f)
            recalls.append(r)
    n = cm.sum()
    return {
        'labels': list(labels),
        'confusion_matrix': cm.tolist(),
        'per_class': per,
        'accuracy': float(np.trace(cm) / n) if n else 0.0,
        'macro_f1': float(np.mean(f1s)) if f1s else 0.0,
        'balanced_accuracy': float(np.mean(recalls)) if recalls else 0.0,
    }


def multilabel_metrics(true_sets, pred_sets, labels=ERROR_TYPES):
    """For multi-error calls: per-type P/R/F1, macro-F1, exact-set accuracy, Jaccard."""
    per, f1s = {}, []
    for l in labels:
        tp = sum(1 for t, p in zip(true_sets, pred_sets) if l in t and l in p)
        fp = sum(1 for t, p in zip(true_sets, pred_sets) if l not in t and l in p)
        fn = sum(1 for t, p in zip(true_sets, pred_sets) if l in t and l not in p)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = {'precision': pr, 'recall': rc, 'f1': f, 'support': tp + fn}
        if tp + fn > 0:
            f1s.append(f)
    exact = np.mean([set(t) == set(p) for t, p in zip(true_sets, pred_sets)]) if true_sets else 0.0
    jacc = np.mean([len(set(t) & set(p)) / len(set(t) | set(p)) if (set(t) | set(p)) else 1.0
                    for t, p in zip(true_sets, pred_sets)]) if true_sets else 0.0
    return {'per_type': per, 'macro_f1': float(np.mean(f1s)) if f1s else 0.0,
            'exact_set_accuracy': float(exact), 'mean_jaccard': float(jacc)}


def bootstrap_ci(fn, *arrays, n=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of fn(*arrays) resampling examples jointly."""
    arrays = [np.asarray(a, dtype=object) for a in arrays]
    m = len(arrays[0])
    if m == 0:
        return {'point': None, 'low': None, 'high': None}
    rng = np.random.default_rng(seed)
    point = fn(*[list(a) for a in arrays])
    vals = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        try:
            vals.append(fn(*[list(a[ix]) for a in arrays]))
        except ZeroDivisionError:
            continue
    vals = np.sort(np.asarray(vals, dtype=float))
    lo = float(np.quantile(vals, alpha / 2)) if len(vals) else None
    hi = float(np.quantile(vals, 1 - alpha / 2)) if len(vals) else None
    return {'point': float(point), 'low': lo, 'high': hi}


def paired_bootstrap_diff(a, b, n=1000, alpha=0.05, seed=0):
    """CI of mean(a) - mean(b) for paired 0/1 outcomes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    m = len(a)
    diffs = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        diffs.append(a[ix].mean() - b[ix].mean())
    diffs = np.sort(diffs)
    return {'diff': float(a.mean() - b.mean()),
            'low': float(np.quantile(diffs, alpha / 2)),
            'high': float(np.quantile(diffs, 1 - alpha / 2))}


def mcnemar_exact(a_correct, b_correct):
    """Exact (binomial) McNemar test on paired 0/1 outcomes.

    n01 = a right, b wrong;  n10 = a wrong, b right.
    """
    a = np.asarray(a_correct, dtype=int)
    b = np.asarray(b_correct, dtype=int)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return {'n01': 0, 'n10': 0, 'p_value': 1.0}
    k = min(n01, n10)
    # two-sided exact binomial p-value with p = 0.5, computed in log space
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(0, k + 1)]
    mx = max(logs)
    p = 2 * math.exp(mx) * sum(math.exp(l - mx) for l in logs)
    return {'n01': n01, 'n10': n10, 'p_value': float(min(1.0, p))}


def detection_report(records, n_boot=1000, alpha=0.05, seed=0):
    """Full detection + classification report from outcome records (section 13)."""
    y_true = [r['is_hallucinated'] for r in records]
    y_pred = [r['flagged'] for r in records]
    rep = {'binary': binary_metrics(y_true, y_pred)}
    rep['ci'] = {
        'accuracy': bootstrap_ci(lambda t, p: binary_metrics(t, p)['accuracy'],
                                 y_true, y_pred, n=n_boot, alpha=alpha, seed=seed),
        'f1': bootstrap_ci(lambda t, p: binary_metrics(t, p)['f1'],
                           y_true, y_pred, n=n_boot, alpha=alpha, seed=seed),
        'balanced_accuracy': bootstrap_ci(lambda t, p: binary_metrics(t, p)['balanced_accuracy'],
                                          y_true, y_pred, n=n_boot, alpha=alpha, seed=seed),
    }
    single = [r for r in records if len(r['true_types']) <= 1]
    if single:
        t = [r['true_types'][0] if r['true_types'] else 'NONE' for r in single]
        p = [r['pred_primary'] for r in single]
        mc = multiclass_metrics(t, p)
        mc['ci_macro_f1'] = bootstrap_ci(lambda a, b: multiclass_metrics(a, b)['macro_f1'],
                                         t, p, n=n_boot, alpha=alpha, seed=seed)
        rep['classification'] = mc
    rep['by_true_type'] = {}
    for lab in ALL_LABELS:
        rs = [r for r in records if (r['true_types'][0] if r['true_types'] else 'NONE') == lab
              and len(r['true_types']) <= 1]
        if rs:
            rep['by_true_type'][lab] = {
                'n': len(rs),
                'flag_rate': float(np.mean([r['flagged'] for r in rs])),
                'type_correct_rate': float(np.mean([r['pred_primary'] == lab for r in rs])),
            }
    return rep


def correction_report(records, n_boot=1000, alpha=0.05, seed=0):
    """The four separate levels asked for in #4."""
    hall = [r for r in records if r['is_hallucinated']]
    clean = [r for r in records if not r['is_hallucinated']]

    def rate(rs, key):
        return float(np.mean([bool(r[key]) for r in rs])) if rs else None

    rep = {
        'n': len(records), 'n_hallucinated': len(hall), 'n_clean': len(clean),
        # level 1: detection
        'detected_rate_hallucinated': rate(hall, 'flagged'),
        'false_flag_rate_clean': rate(clean, 'flagged'),
        # level 2: classification (single-error calls, flagged ones)
        'type_correct_rate_detected': (float(np.mean([r['pred_primary'] in r['true_types']
                                                      for r in hall if r['flagged']]))
                                       if any(r['flagged'] for r in hall) else None),
        # level 3: correction -> schema-valid output
        'final_schema_valid_rate_all': rate(records, 'final_schema_valid'),
        'final_schema_valid_rate_hallucinated': rate(hall, 'final_schema_valid'),
        # level 4: intent proxy -> output equals the reference call
        'final_exact_rate_all': rate(records, 'final_exact'),
        'final_exact_rate_hallucinated': rate(hall, 'final_exact'),
        'final_exact_rate_clean': rate(clean, 'final_exact'),
        'schema_valid_but_not_exact_hallucinated':
            (float(np.mean([r['final_schema_valid'] and not r['final_exact'] for r in hall]))
             if hall else None),
        'clean_damaged_rate': (float(np.mean([not r['final_exact'] for r in clean]))
                               if clean else None),
    }
    rep['ci'] = {
        'final_exact_rate_all': bootstrap_ci(lambda x: float(np.mean(x)),
                                             [int(r['final_exact']) for r in records],
                                             n=n_boot, alpha=alpha, seed=seed),
        'final_exact_rate_hallucinated': bootstrap_ci(lambda x: float(np.mean(x)),
                                                      [int(r['final_exact']) for r in hall],
                                                      n=n_boot, alpha=alpha, seed=seed),
    }
    by = {}
    for lab in ALL_LABELS:
        rs = [r for r in records if len(r['true_types']) <= 1 and
              (r['true_types'][0] if r['true_types'] else 'NONE') == lab]
        if rs:
            by[lab] = {'n': len(rs), 'flagged': rate(rs, 'flagged'),
                       'final_schema_valid': rate(rs, 'final_schema_valid'),
                       'final_exact': rate(rs, 'final_exact')}
    rep['by_true_type'] = by
    return rep


def strip_ci(d):
    """Round floats for printing."""
    if isinstance(d, dict):
        return {k: strip_ci(v) for k, v in d.items()}
    if isinstance(d, list):
        return [strip_ci(v) for v in d]
    if isinstance(d, float):
        return round(d, 4)
    return d


def fmt_ci(ci, digits=3):
    if not ci or ci.get('point') is None:
        return 'n/a'
    return f"{ci['point']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


# ---------------------------------------------------------------------------
# 13. Pipeline runner
# ---------------------------------------------------------------------------

class Pipeline:
    """Validator + corrector bundle built from one config."""

    def __init__(self, registry, cfg, type_trust=None, matcher=None,
                 validator_overrides=None, corrector_overrides=None):
        self.reg = registry
        self.cfg = cfg
        vcfg = deep_update(cfg['validator'], validator_overrides or {})
        ccfg = deep_update(cfg['corrector'], corrector_overrides or {})
        self.accepted = cfg['envelope']['accepted_key_sets']
        trust = type_trust or TypeTrust(mode='strict')
        self.validator = Validator(registry, vcfg, self.accepted, trust, cfg['grounding'])
        # "schema-valid" is judged by the schema checks only (never grounding),
        # so the four levels in #4 stay separate.
        self.schema_checker = Validator(registry, {**vcfg, 'check_grounding': False},
                                        self.accepted, trust, cfg['grounding'])
        self.corrector = Corrector(registry, self.validator, matcher, ccfg)
        self.detect_only = False

    def run_one(self, ex, timers=None):
        """Outcome record for one example.  GT is used only AFTER the pipeline.

        Unflagged calls are passed through UNCHANGED (raw envelope kept);
        flagged calls are replaced by the corrector's output (or blocked when
        detect_only=True).
        """
        raw = ex['generated_tool_call']
        query = ex.get('query')
        t0 = time.perf_counter()
        viols = self.validator.validate(raw, query=query)
        t1 = time.perf_counter()
        flagged = len(viols) > 0
        meta = None
        envelope_unfixed = False
        if not flagged:
            final = parse_raw_call(raw)
        elif self.detect_only:
            final = None
        else:
            final, meta = self.corrector.correct(raw, query=query)
            envelope_unfixed = bool(meta.get('envelope_unfixed'))
        t2 = time.perf_counter()
        if final is None:
            schema_valid = False
        else:
            schema_valid = (len(self.schema_checker.validate(final)) == 0) and not envelope_unfixed
        if timers is not None:
            timers['validate'].append(t1 - t0)
            timers['correct'].append(t2 - t1 if flagged else 0.0)
            timers['total'].append(t2 - t0)
        gt = ex.get('ground_truth_tool_call')
        true_types = list(ex.get('error_types') or [])
        return {
            'id': ex.get('id'),
            'function_key': ex.get('function_key'),
            'is_hallucinated': bool(true_types),
            'true_types': true_types,
            'n_errors': len(true_types),
            'flagged': flagged,
            'violations': [{'type': v['type'], 'stage': v['stage'], 'param': v.get('param'),
                            'subtype': v.get('subtype')} for v in viols],
            'pred_types': Validator.predicted_types(viols),
            'pred_primary': Validator.primary_type(viols),
            'final_call': final,
            'final_schema_valid': schema_valid,
            'final_exact': (not envelope_unfixed) and exact_strict(final, gt, self.accepted),
            'final_same_call_ignoring_envelope': final is not None and gt is not None
                                                 and calls_equal(final, gt),
            'iterations': (meta or {}).get('iterations', 0),
            'actions': [a['action'] for a in (meta or {}).get('actions', [])],
            'uncertain_params': (meta or {}).get('uncertain_params', []),
            'flagged_for_retrieval': (meta or {}).get('flagged_for_retrieval', []),
        }

    def run(self, examples, timers=None):
        return [self.run_one(ex, timers) for ex in examples]


def new_timers():
    return defaultdict(list)


def latency_summary(timers):
    out = {}
    for k, v in timers.items():
        a = np.asarray(v, dtype=float) * 1000.0
        if len(a) == 0:
            continue
        out[k] = {'n': int(len(a)), 'mean_ms': float(a.mean()), 'p50_ms': float(np.percentile(a, 50)),
                  'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
                  'max_ms': float(a.max())}
    return out


def failure_category(r):
    """One bucket per example (same success criterion as final_exact)."""
    if r['final_exact']:
        if not r['is_hallucinated'] and not r['flagged']:
            return 'OK_CLEAN_PASSED'
        if not r['is_hallucinated']:
            return 'OK_FLAGGED_BUT_RESTORED'
        if not r['flagged']:
            return 'OK_MISSED_BUT_HARMLESS'
        return 'OK_CORRECTED_EXACT'
    if not r['is_hallucinated']:
        return 'FALSE_POSITIVE_DAMAGED'
    if not r['flagged']:
        return 'MISSED_DETECTION'
    if r['final_call'] is None:
        return 'UNPARSEABLE'
    if r['final_schema_valid']:
        if r['uncertain_params']:
            return 'SCHEMA_VALID_PLACEHOLDER_VALUE'
        return 'SCHEMA_VALID_NOT_EXACT'
    return 'CORRECTION_INCOMPLETE'


# ---------------------------------------------------------------------------
# 14. Timing and hardware information
# ---------------------------------------------------------------------------

def hardware_info():
    info = {'platform': platform.platform(), 'python': platform.python_version(),
            'processor': platform.processor() or platform.machine(),
            'cpu_count': os.cpu_count()}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    info['ram_gb'] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu_model'] = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        info['torch'] = None
    return info


def package_versions(names=('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers',
                            'sentence-transformers', 'datasets', 'bert-score', 'ddgs',
                            'jsonschema', 'matplotlib', 'pandas')):
    out = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        return out
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def print_header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


# ---------------------------------------------------------------------------
# 15. Error injection (benchmark construction, used by Code_1)
# ---------------------------------------------------------------------------

FAKE_PARAMS = ['limit', 'offset', 'max_results', 'page_size', 'timeout', 'verbose',
               'debug', 'format', 'version', 'cache', 'retry', 'async_mode', 'callback',
               'filter', 'sort_by', 'include_metadata', 'raw_response', 'api_version',
               'locale']
FAKE_VALUES = [10, "auto", True, "default", 1, "all"]

ALT_ENVELOPES = {
    'action/action_input': lambda fn, p: {"action": fn, "action_input": p},
    'openai_nested': lambda fn, p: {"type": "function",
                                    "function": {"name": fn, "arguments": json.dumps(p)}},
    'tool/tool_input': lambda fn, p: {"tool": fn, "tool_input": p},
    'function_name/function_params': lambda fn, p: {"function_name": fn, "function_params": p},
}


class Injector:
    """Injects errors into a clean call, using the registry so every injected
    error is an error WITH RESPECT TO THE SCHEMA.

    Each operator returns (new_call, metadata) or (None, reason).  `touched`
    is the set of parameter names already modified (multi-error injection
    never edits the same parameter twice).

    Changes from the previous injector (all listed in CHANGES.md):
      * WRONG_FUNCTION: the corrupted name is checked to be absent from the registry.
      * EXTRA_PARAMETER: the fake name is checked against the SCHEMA, not just the call.
      * MISSING_PARAMETER: only a parameter the schema marks REQUIRED is removed
        (removing an optional one is not an error).
      * WRONG_TYPE: only a parameter whose clean value matches its declared
        type is corrupted, and the new value is checked to be incompatible
        with the declared type.
      * WRONG_VALUE: 'donor_swap' replaces the value with a real value of the
        same type taken from ANOTHER call that uses the same parameter name
        (Mumbai -> Delhi style).  The old string reversal / 2099 dates /
        negated numbers are kept only as a fallback ('perturbation').
    """

    def __init__(self, registry, rng, donor_index=None, wrong_value_mode='donor_swap'):
        self.reg = registry
        self.rng = rng
        self.donors = donor_index or {}
        self.wv_mode = wrong_value_mode

    @staticmethod
    def build_donor_index(calls):
        idx = defaultdict(list)
        for c in calls:
            for p, v in (c.get('parameters') or {}).items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
                    idx[p].append(v)
        return {p: list(dict.fromkeys(vs)) for p, vs in idx.items()}

    # -- operators ---------------------------------------------------------
    def wrong_function(self, call, touched):
        orig = call['function']
        if not orig:
            return None, 'no function'
        variants = [orig + "_v2", orig + "_new",
                    orig.replace("get_", "fetch_") if "get_" in orig else "get_" + orig,
                    orig.replace("search", "find") if "search" in orig else orig + "_search",
                    orig + "_api",
                    orig.rsplit(".", 1)[0] + ".NonExistent" if "." in orig else orig + "_deprecated"]
        self.rng.shuffle(variants)
        for v in variants:
            if v != orig and self.reg.resolve(v) is None:
                out = copy.deepcopy(call)
                out['function'] = v
                return out, {'op': 'WRONG_FUNCTION', 'original': orig, 'corrupted': v}
        return None, 'all variants exist in registry'

    def extra_parameter(self, call, touched):
        schema = self.reg.params(call['function'])
        avail = [p for p in FAKE_PARAMS if p not in call['parameters'] and p not in schema]
        if not avail:
            return None, 'no free fake name'
        p = self.rng.choice(avail)
        v = self.rng.choice(FAKE_VALUES)
        out = copy.deepcopy(call)
        out['parameters'][p] = v
        touched.add(p)
        return out, {'op': 'EXTRA_PARAMETER', 'param': p, 'value': v}

    def missing_parameter(self, call, touched):
        req = [p for p in self.reg.required_params(call['function'])
               if p in call['parameters'] and p not in touched
               and call['parameters'][p] not in ('', None)]
        if not req:
            return None, 'no required parameter present'
        p = self.rng.choice(sorted(req))
        out = copy.deepcopy(call)
        removed = out['parameters'].pop(p)
        touched.add(p)
        return out, {'op': 'MISSING_PARAMETER', 'param': p, 'removed_value': removed}

    @staticmethod
    def _corrupt_type(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        if isinstance(v, str):
            return [v]
        return None

    def wrong_type(self, call, touched):
        schema = self.reg.params(call['function'])
        cands = []
        for p, v in call['parameters'].items():
            if p in touched or v in ('', None):
                continue
            spec = schema.get(p)
            if not isinstance(spec, dict) or spec.get('type') is None:
                continue
            if not types_compatible(spec['type'], value_type(v)):
                continue        # the clean value already contradicts the schema
            nv = self._corrupt_type(v)
            if nv is None or types_compatible(spec['type'], value_type(nv)):
                continue
            cands.append((0 if not isinstance(v, str) else 1, p, nv))
        if not cands:
            return None, 'no type-consistent parameter'
        best_rank = min(c[0] for c in cands)
        pool = sorted([c for c in cands if c[0] == best_rank], key=lambda c: c[1])
        _, p, nv = self.rng.choice(pool)          # prefer non-string values, as before
        out = copy.deepcopy(call)
        ov = out['parameters'][p]
        out['parameters'][p] = nv
        touched.add(p)
        return out, {'op': 'WRONG_TYPE', 'param': p, 'original_value': ov,
                     'original_type': value_type(ov), 'corrupted_value': nv,
                     'corrupted_type': value_type(nv),
                     'declared_type': schema[p].get('type')}

    @staticmethod
    def _perturb(v):
        if isinstance(v, str):
            if re.search(r'\d{4}-\d{2}-\d{2}', v):
                return re.sub(r'\d{4}', '2099', v, count=1), 'date_to_future'
            if v.isdigit() and len(v) <= 6:
                return (str(-int(v)) if int(v) > 0 else str(abs(int(v)) + 99999)), 'numeric_wrong'
            if len(v) > 5 and not v.startswith('http'):
                return v[::-1], 'string_reversed'
            return None, None
        if isinstance(v, bool):
            return None, None
        if isinstance(v, (int, float)):
            return (v + 7 if v >= 0 else v - 7), 'numeric_shift'
        return None, None

    def wrong_value(self, call, touched, query=None):
        params = [p for p, v in call['parameters'].items()
                  if p not in touched and isinstance(v, (str, int, float))
                  and not isinstance(v, bool) and str(v).strip()]
        params = sorted(params)
        self.rng.shuffle(params)
        if self.wv_mode == 'donor_swap':
            for p in params:
                v = call['parameters'][p]
                pool = [d for d in self.donors.get(p, [])
                        if value_type(d) == value_type(v) and str(d) != str(v)]
                if pool:
                    nv = self.rng.choice(pool)
                    out = copy.deepcopy(call)
                    out['parameters'][p] = nv
                    touched.add(p)
                    return out, {'op': 'WRONG_VALUE', 'param': p, 'original': v,
                                 'corrupted': nv, 'method': 'donor_swap',
                                 'gt_value_grounded': value_grounded(v, query)}
        for p in params:
            v = call['parameters'][p]
            nv, how = self._perturb(v)
            if nv is not None and str(nv) != str(v):
                out = copy.deepcopy(call)
                out['parameters'][p] = nv
                touched.add(p)
                return out, {'op': 'WRONG_VALUE', 'param': p, 'original': v,
                             'corrupted': nv, 'method': 'perturbation:' + how,
                             'gt_value_grounded': value_grounded(v, query)}
        return None, 'no value to change'

    def wrong_format(self, call, touched):
        name = self.rng.choice(sorted(ALT_ENVELOPES))
        return ALT_ENVELOPES[name](call['function'], copy.deepcopy(call['parameters'])), \
            {'op': 'WRONG_FORMAT', 'envelope': name}

    # Parameter-level errors first, then the function name, then the envelope,
    # so a later operator never undoes an earlier one.
    APPLY_ORDER = ['WRONG_VALUE', 'WRONG_TYPE', 'MISSING_PARAMETER', 'EXTRA_PARAMETER',
                   'WRONG_FUNCTION', 'WRONG_FORMAT']

    def inject(self, call, types, query=None):
        """Apply the given error types to one clean call.

        Returns (generated_call, metadata_list) or (None, reason)."""
        work = copy.deepcopy(call)
        touched, metas = set(), []
        for t in sorted(types, key=self.APPLY_ORDER.index):
            if t == 'WRONG_FORMAT':
                out, meta = self.wrong_format(work, touched)
            elif t == 'WRONG_VALUE':
                out, meta = self.wrong_value(work, touched, query)
            else:
                op = {'WRONG_FUNCTION': self.wrong_function,
                      'EXTRA_PARAMETER': self.extra_parameter,
                      'WRONG_TYPE': self.wrong_type,
                      'MISSING_PARAMETER': self.missing_parameter}[t]
                out, meta = op(work, touched)
            if out is None:
                return None, f"{t}: {meta}"
            work = out
            metas.append(meta)
        if calls_equal(work, call) and 'WRONG_FORMAT' not in types:
            return None, 'injection left the call unchanged'
        return work, metas


def cast_reference_value(v, declared):
    """Losslessly cast a published example value to its declared type.

    ToolBench stores example values as strings whatever the declared type
    (11,108 NUMBER parameters have defaults like "2" vs 1,089 real numbers in
    one shard), so reference calls built from them contradict their own
    schema.  Only exact numeric / boolean strings are cast; everything else
    is returned unchanged.  Returns (value, was_cast).
    """
    if not isinstance(v, str):
        return v, False
    s = v.strip()
    if declared in ('integer', 'number') and re.fullmatch(r'-?\d+', s):
        if len(s.lstrip('-')) > 15:
            return v, False            # IDs too long to be numbers safely
        return int(s), True
    if declared == 'number' and re.fullmatch(r'-?\d+\.\d+', s):
        f = float(s)
        if repr(f) == s or str(f) == s:
            return f, True
        return v, False
    if declared == 'boolean' and s.lower() in ('true', 'false'):
        return s.lower() == 'true', True
    return v, False


# ---------------------------------------------------------------------------
# 16. LLM helpers (Code_4 and Code_5).  torch / transformers imported lazily.
# ---------------------------------------------------------------------------

def schema_for_prompt(registry, fn):
    """Compact schema of one function for an LLM prompt."""
    params = {}
    for p, s in registry.params(fn).items():
        d = {'type': s.get('type') or 'any', 'required': bool(s.get('required'))}
        if s.get('enum'):
            d['enum'] = s['enum'][:20]
        if s.get('description'):
            d['description'] = str(s['description'])[:120]
        params[p] = d
    return {'name': fn, 'description': str(registry.description(fn))[:200], 'parameters': params}


def load_causal_lm(name, device=None, trust_remote_code=False):
    """All default models (Qwen2.5, TinyLlama, Phi-3.5) are supported natively by
    transformers, so the repositories' own Python code is NOT executed by default."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=trust_remote_code, low_cpu_mem_usage=True).to(device)
    model.eval()
    return tok, model, device


def chat_text(tok, messages):
    """Apply the model's chat template; fall back to a plain prompt if it has none."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        out = ''
        for m in messages:
            out += f"{m['role'].upper()}: {m['content']}\n"
        return out + 'ASSISTANT: '


def generate_batch(tok, model, device, prompts, max_new_tokens=256, batch_size=8):
    """Greedy generation for a list of prompt strings. Returns (texts, seconds_per_prompt)."""
    import torch
    outs, secs = [], []
    for i in range(0, len(prompts), batch_size):
        b = prompts[i:i + batch_size]
        enc = tok(b, return_tensors='pt', padding=True, truncation=True, max_length=3072,
                  add_special_tokens=False).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        dt = (time.perf_counter() - t0) / len(b)
        for row in gen:
            outs.append(tok.decode(row[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip())
            secs.append(dt)
    return outs, secs


def _cut_json_value(text):
    """Return the first complete JSON value at the start of `text` (scalar, list or object)."""
    s = text.lstrip()
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
                if depth == 0:
                    return s[:i + 1]
            continue
        if ch == '"':
            in_str = True
        elif ch in '[{':
            depth += 1
        elif ch in ']}':
            if depth == 0:
                return s[:i].strip()
            depth -= 1
            if depth == 0:
                return s[:i + 1]
        elif ch in ',\n' and depth == 0:
            return s[:i].strip()
    return s.strip()


def constrained_generate(tok, model, device, prompt_text, functions, registry, max_value_tokens=40):
    """Schema-constrained decoding (lightweight, no external library).

    1. The function name is CHOSEN, not generated: each allowed name is scored
       by the model's log-likelihood after the prefix '{"function": "' and the
       best one is taken.
    2. The output structure is forced: '{"function": ..., "parameters": {' and
       every parameter KEY comes from the schema (required ones first).
    3. Each VALUE is generated freely (greedy) after '"<key>": ' and cut at the
       first complete JSON value; 'null' means "leave the optional parameter out".
       The value is then cast to the declared type when possible.
    Values are therefore the only part the model controls -- which is exactly
    what constrained decoding can and cannot guarantee.
    """
    import torch
    prefix = prompt_text + '{"function": "'
    best, best_lp = None, -1e18
    for fn in functions:
        ids_p = tok(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        ids_f = tok(fn + '"', return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        ids = torch.cat([ids_p, ids_f], dim=1)
        with torch.no_grad():
            logits = model(ids).logits[0, ids_p.shape[1] - 1:-1]
        lp = torch.log_softmax(logits.float(), -1).gather(1, ids_f[0].unsqueeze(1)).sum().item()
        lp /= max(1, ids_f.shape[1]) ** 0.5          # mild length normalisation
        if lp > best_lp:
            best, best_lp = fn, lp
    schema = registry.params(best)
    keys = sorted(schema, key=lambda p: (not registry.is_required(schema[p]), p))
    params, partial = {}, prompt_text + json.dumps({'function': best})[:-1] + ', "parameters": {'
    for p in keys:
        stem = partial + ('' if not params else ', ') + json.dumps(p) + ': '
        enc = tok(stem, return_tensors='pt', add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_value_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        txt = tok.decode(g[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        raw_v = _cut_json_value(txt)
        try:
            v = json.loads(raw_v)
        except Exception:
            v = raw_v.strip().strip('"')
        if v is None or (v == '' and not registry.is_required(schema[p])):
            continue
        decl = schema[p].get('type')
        if decl and not types_compatible(decl, value_type(v)):
            c = safe_cast(v, decl)
            v = c if c is not None else v
        params[p] = v
        partial = stem + json.dumps(v)
    return {'function': best, 'parameters': params}
