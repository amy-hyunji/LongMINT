import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

DATASET_PROMPTS_DIR = Path(__file__).resolve().parent / "src" / "memagent" / "prompts"


SYSTEM_PROMPT = (
    "You answer questions based on the provided context. "
    "Output format: end your response with \\boxed{...} containing the final "
    "answer. ONLY the text inside the box is graded (exact-match), so it must "
    "be the precise value with no explanation, no quotes, and no trailing "
    "punctuation. If a list of candidate values is provided, the boxed answer "
    "MUST be one of those values verbatim. Keep any reasoning brief; what "
    "matters is the final \\boxed{...}."
)


def load_dataset_prompt(dataset: str) -> str:
    """Read `src/memagent/prompts/<dataset>.md` as a dataset-level instruction
    addendum (shared with the MemAgent runner). Returns "" when no file exists
    for the dataset."""
    if not dataset:
        return ""
    path = DATASET_PROMPTS_DIR / f"{dataset}.md"
    if not path.is_file():
        return ""
    return path.read_text().strip()


def build_question_block(qa: dict, dataset_prompt: str = "") -> str:
    """Render the QA question block.

    `qa['candidates']` and ordering / comma-list phrasing are independent of
    the dataset prompt: they describe the *answer shape* and are kept separate
    so a caller can opt into them without forcing a dataset-level instruction.
    `dataset_prompt` (loaded from `src/memagent/prompts/<dataset>.md`) is a
    free-form addendum and is appended LAST so it parallels MemAgent's
    `format_question()` ordering — the model sees it after the boxed-output
    instruction, in both the user-message context and any downstream prompt
    that reuses this block.
    """
    parts = ["=== QUESTION ===", qa["question"]]
    cands = qa.get("candidates")
    if cands:
        if isinstance(cands, (list, tuple)):
            cands_str = "\n".join(f"- {c}" for c in cands)
        else:
            cands_str = str(cands)
        parts.append(
            "Pick the answer from this candidate list (the boxed value MUST "
            f"appear verbatim, with no surrounding quotes or list brackets):\n{cands_str}"
        )
    else:
        parts.append("Answer in free form: a single short string.")
    if qa.get("question_type") == "ordering":
        parts.append("Output your answer as a comma-separated list.")
    parts.append("Put your final answer inside \\boxed{...}.")
    if dataset_prompt:
        parts.append(dataset_prompt)
    return "\n\n".join(parts)


CONTEXT_HEADER = "=== CONTEXT ==="


def build_user_prompt(passages, question_block: str) -> str:
    s = CONTEXT_HEADER + "\n\n"
    for p in passages:
        s += f"{p}\n\n"
    s += question_block
    return s


def truncate_user_prompt(user_prompt: str, char_budget: int):
    """Cap the assembled user message at `char_budget` chars without dropping
    the question block. Trim docs from the tail (worst-scoring first)."""
    orig_len = len(user_prompt)
    if char_budget <= 0 or orig_len <= char_budget:
        return user_prompt, False, orig_len
    cut = "\n=== QUESTION ==="
    if cut not in user_prompt:
        return user_prompt[-char_budget:], True, orig_len
    i = user_prompt.rindex(cut)
    docs_part = user_prompt[:i]
    q_part = user_prompt[i:]
    if len(q_part) >= char_budget:
        return q_part[-char_budget:], True, orig_len
    docs_budget = char_budget - len(q_part)
    return docs_part[:docs_budget] + q_part, True, orig_len


def _last_boxed_span(s: str):
    """Locate the LAST \\boxed{...} with proper brace counting (MATH /
    Qwen-math style). Returns (start, end) of the inner content, or None.
    `rfind` picks the rightmost `\\boxed{`, then we scan forward tracking
    depth so nested `{...}` inside the answer (e.g. `\\boxed{f(x)={a,b}}`)
    don't truncate the match."""
    key = "\\boxed{"
    idx = s.rfind(key)
    if idx < 0:
        return None
    start = idx + len(key)
    depth = 1
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, i
    return start, len(s)


def extract_boxed(text: str) -> str:
    """Strip <think>...</think> then return the content of the LAST
    \\boxed{...}. Handles nested braces via brace counting (matches
    Qwen-math / MATH-benchmark behavior). Falls back to the stripped text
    when no boxed segment is present."""
    s = text or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    if "<think>" in s and "</think>" not in s:
        s = s.split("<think>")[0]
    span = _last_boxed_span(s)
    if span is not None:
        a, b = span
        return s[a:b].strip()
    return s.strip()



def normalize_answer(s):

    s = s.replace(',', ' ')
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_f1(pred: str, golds):
    pn = normalize_answer(pred)
    em, f1 = 0.0, 0.0
    for g in golds:
        gn = normalize_answer(str(g))
        if pn == gn:
            em = max(em, 1.0)
        pt, gt = pn.split(), gn.split()
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        f1 = max(f1, 2 * prec * rec / (prec + rec))
    return em, f1


def render_doc(ctx: dict, max_doc_chars: int) -> str:
    """Render one unified context as a doc string.
    `timestamp` is prepended when present so temporal questions still hit
    a useful retrieval signal. `max_doc_chars` caps the rendered length
    (0 = disabled)."""
    content = ctx.get("content", "") or ""
    timestamp = ctx.get("timestamp")
    if timestamp:
        doc = f"[{timestamp}]\n{content}"
    else:
        doc = content
    if max_doc_chars and len(doc) > max_doc_chars:
        doc = doc[:max_doc_chars] + "\n[…truncated…]"
    return doc


HF_DATASET_ID = "dinobby/LongMINT"
HF_SPLIT_MAP = {
    "babi":         "state_tracking",
    "wiki":         "wiki_revisions",
    "github":       "github_commits",
    "horizonbench": "multi_turn_dialogue",
}


def _qmeta(raw):
    """Unified `questions[].metadata` is a dict in the JSON copy and a JSON-
    encoded string in the HF parquet copy. Normalize to dict (or {} when
    missing/unparseable)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _items_to_instances(items, max_doc_chars: int):
    """Common reshape step shared by the JSON and HF loaders. `items` is an
    iterable of dicts following the unified schema."""
    instances = []
    for item in items:
        docs = [render_doc(ctx, max_doc_chars) for ctx in item.get("contexts") or []]

        qas = defaultdict(list)
        for q in item.get("questions") or []:
            ans = q.get("answer")
            if ans is None:
                continue
            qmeta = _qmeta(q.get("metadata"))
            qtype = q.get("question_type") or "default"
            qas[qtype].append({
                "question":      q["question"],
                "answer":        str(ans),
                "question_type": qtype,
                "candidates":    qmeta.get("candidates"),
                "n_steps_back":  qmeta.get("n_steps_back"),
            })

        instances.append({
            "instance_id":       item["id"],
            "facts":             docs,
            "qas":               dict(qas),
            "instance_metadata": item.get("metadata"),
        })
    return instances


def load_unified(path: str, max_doc_chars: int = 0):
    """Load a unified-format JSON file. See `load_dataset` for the more
    general entry point that also handles HF datasets."""
    with open(path) as fp:
        data = json.load(fp)
    return _items_to_instances(data, max_doc_chars)


def load_from_hf(dataset: str, max_doc_chars: int = 0, *,
                 repo_id: str = HF_DATASET_ID):
    """Load a split from the dinobby/LongMINT HF dataset (or any repo with
    the same split-naming convention) and reshape into our in-memory schema.

    `dataset` is one of our short names (babi / wiki / github / horizonbench);
    it maps to the corresponding HF split via HF_SPLIT_MAP. The HF copy
    serializes `questions[].metadata` as a JSON string — `_qmeta` parses it
    back into a dict so downstream code doesn't need to care which source it
    came from.
    """
    if dataset not in HF_SPLIT_MAP:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of {sorted(HF_SPLIT_MAP)}"
        )
    # Import lazily so callers that only use the JSON path don't pay the
    # `datasets` import cost (it's ~1 sec on first import).
    from datasets import load_dataset
    split = HF_SPLIT_MAP[dataset]
    ds = load_dataset(repo_id, split=split)
    return _items_to_instances(ds, max_doc_chars)


def load_raw_dataset(dataset: str, data_root: str | None, *,
                     repo_id: str = HF_DATASET_ID):
    """Return items in the *raw* unified JSON schema
    (id / contexts / questions / metadata) rather than the reshaped
    (instance_id / facts / qas) layout produced by load_dataset_unified.

    Source selection mirrors baserag/hipporag: when `data_root` is given,
    load `<data_root>/<dataset>.json`; otherwise pull the matching split
    from the HF dataset (default `dinobby/LongMINT`). In both cases
    `questions[].metadata` is normalized to a dict so callers indexing
    into the raw structure (atommem, fullcontext) don't need to care
    which source it came from.

    Returns (items, source_desc).
    """
    if data_root:
        path = Path(data_root) / f"{dataset}.json"
        with open(path) as fp:
            items = json.load(fp)
        src = f"local:{path}"
    else:
        if dataset not in HF_SPLIT_MAP:
            raise ValueError(
                f"Unknown dataset {dataset!r}; expected one of {sorted(HF_SPLIT_MAP)}"
            )
        from datasets import load_dataset
        split = HF_SPLIT_MAP[dataset]
        items = [dict(row) for row in load_dataset(repo_id, split=split)]
        src = f"hf:{repo_id}[{split}]"
    for it in items:
        for q in it.get("questions") or []:
            q["metadata"] = _qmeta(q.get("metadata"))
    return items, src


def load_dataset_unified(dataset: str, data_path: str | None,
                         max_doc_chars: int = 0):
    """Single entry point: when `data_path` is given, load that JSON file;
    otherwise pull the dataset from Hugging Face (`dinobby/LongMINT`). Returns
    the same in-memory schema in both cases."""
    if data_path:
        return load_unified(data_path, max_doc_chars)
    return load_from_hf(dataset, max_doc_chars)


def save_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as wf:
        json.dump(obj, wf)
    os.replace(tmp, path)


def init_pred_dict(elem, cat_ranges):
    pd = elem.setdefault("pred_dict", {})
    for cat in cat_ranges:
        pd.setdefault(cat, {
            "solutions": [], "rationales": [], "usage": [],
            "input_tokens": [], "output_tokens": [], "metrics": {},
        })
    return pd


def _safe_id(s: str) -> str:
    """instance_id can contain '/', '.', etc. Normalize to a filesystem-safe
    name for per-instance graph artifacts."""
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    return s[:120] or "untitled"


def derive_dataset_name(data_path: str) -> str:
    """Pull a dataset tag from the data file basename (e.g.
    `.../unified/horizonbench.json` → `horizonbench`)."""
    return os.path.splitext(os.path.basename(data_path))[0]


def apply_shard_and_limit(instances, shard_idx: int, num_shards: int, limit):
    """Tag instances with their original index, shard them across processes,
    and optionally truncate to `limit` items (smoke tests)."""
    for i, inst in enumerate(instances):
        inst["_orig_idx"] = i
    if num_shards > 1:
        sharded = [inst for inst in instances
                   if inst["_orig_idx"] % num_shards == shard_idx]
        print(f"[shard {shard_idx}/{num_shards}] {len(sharded)} of "
              f"{len(instances)} instances")
        instances = sharded
    if limit is not None:
        instances = instances[:limit]
        print(f"[smoke-test] truncated to first {len(instances)} instance(s)")
    return instances


def resume_pred_dicts(save_path: str, instances):
    """If `save_path` exists, copy previously-computed `pred_dict` entries
    onto the in-memory instances (matched by `instance_id`). Returns the
    count resumed."""
    if not os.path.exists(save_path):
        return 0
    with open(save_path) as rf:
        saved = json.load(rf)
    saved_by_id = {s["instance_id"]: s for s in saved}
    resumed = 0
    for inst in instances:
        prev = saved_by_id.get(inst["instance_id"])
        if prev and prev.get("pred_dict"):
            inst["pred_dict"] = prev["pred_dict"]
            resumed += 1
    print(f"Resuming from {save_path}: {resumed}/{len(instances)} instances "
          f"already have pred_dict")
    return resumed


def build_qlist(qas):
    """Flatten `qas` (dict of category → list) into per-qi parallel arrays.

    Returns (qlist, qa_items, gold_answers, cat_ranges) where:
      qlist:        [question_str, ...]
      qa_items:     [full qa dict, ...]  (for build_question_block)
      gold_answers: [[gold_str], ...]
      cat_ranges:   {cat: (start, end)}
    """
    qlist, qa_items, gold_answers, cat_ranges = [], [], [], {}
    for cat, questions in qas.items():
        start = len(qlist)
        for q in questions:
            qlist.append(q["question"])
            qa_items.append(q)
            gold_answers.append([str(q["answer"])])
        cat_ranges[cat] = (start, len(qlist))
    return qlist, qa_items, gold_answers, cat_ranges


def cuda_cleanup():
    """Best-effort GPU cache cleanup between instances. Silent no-op when
    torch/CUDA isn't available."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
