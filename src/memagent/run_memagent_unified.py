import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp

URL = os.getenv("URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "123-abc")
QA_URL = os.getenv("QA_URL", URL)
QA_API_KEY = os.getenv("QA_API_KEY", API_KEY)

DATASETS = ["babi", "github", "horizonbench", "wiki"]

DEFAULT_HF_REPO = "dinobby/LongMINT"
HF_SPLIT_MAP = {
    "babi":         "state_tracking",
    "github":       "github_commits",
    "horizonbench": "multi_turn_dialogue",
    "wiki":         "wiki_revisions",
}


def _maybe_unstring(v):
    """Parquet rows ship `metadata` as a JSON string; the rest of the runner
    expects a dict. Pass through anything that isn't a non-empty string."""
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def load_longmint_items(dataset: str, *, hf_repo: str, local_dir: str | None):
    """Load one unified dataset's items.

    Priority:
      1. local_dir (when given) -> read <dataset>.json from disk
      2. otherwise -> `datasets.load_dataset(hf_repo, split=mapped)` and
         normalize the per-row + per-question `metadata` strings into dicts.
    """
    if local_dir:
        p = Path(local_dir) / f"{dataset}.json"
        if not p.is_file():
            raise FileNotFoundError(
                f"--data-root={local_dir} but {p} doesn't exist. "
                f"Drop --data-root to fall back to the HF source ({hf_repo})."
            )
        with open(p) as f:
            return json.load(f), f"local:{p}"

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Loading from HF Hub requires the `datasets` package. "
            "Either `pip install datasets`, or pass --data-root <dir> to "
            "load JSON files from disk."
        ) from e

    split = HF_SPLIT_MAP[dataset]
    hf = load_dataset(hf_repo, split=split)
    items = []
    for row in hf:
        questions = []
        for q in row.get("questions", []) or []:
            qq = dict(q)
            qq["metadata"] = _maybe_unstring(qq.get("metadata"))
            questions.append(qq)
        items.append({
            "id":       row.get("id"),
            "contexts": list(row.get("contexts", []) or []),
            "questions": questions,
            "metadata": _maybe_unstring(row.get("metadata")),
        })
    return items, f"hf:{hf_repo}#{split}"

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_dataset_prompt(dataset: str) -> str:
    """Load a dataset-level prompt addendum from `prompts/<dataset>.md`.

    The text is appended ONLY to the final QA prompt — not to the per-chunk
    memory-update prompts — so the dataset notes don't bias every
    summarization step. Returns "" if no file exists for the dataset.
    """
    path = PROMPTS_DIR / f"{dataset}.md"
    if not path.is_file():
        return ""
    return path.read_text().strip()


def clip_long_string(s: str, max_length: int = 2000) -> str:
    if len(s) <= max_length:
        return s
    target = max_length - len("\n\n...(truncated)\n\n")
    return s[: target // 2] + "\n\n...(truncated)\n\n" + s[-target // 2 :]


def format_question(q: dict) -> str:
    """Render the question text with format/constraint instructions.

    Across all datasets every ordering answer is a comma-separated list (e.g.
    "Daniel" / "garden, bedroom" / "ThymeBoost/...py, ThymeBoost/...py, ...").
    Some datasets (babi, horizonbench) already say so inline, others (github,
    wiki) don't -- so we always append the instruction for ordering.

    If `metadata.candidates` is present (horizonbench), constrain the answer
    to those values. Phrasing depends on question_type: ordering takes a
    subset-in-order, the rest take a single value.

    The dataset-level addendum (prompts/<dataset>.md) is NOT folded in here —
    it's reserved for the final QA call only (see `with_dataset_prompt`). The
    memory-update loop sees the bare question + answer-shape instructions so
    the dataset notes don't bias every per-chunk summarization.
    """
    text = q["question"]
    qtype = q.get("question_type")
    candidates = (q.get("metadata") or {}).get("candidates")

    extras = []
    if qtype == "ordering":
        if candidates:
            cand_block = "\n".join(f"- {c}" for c in candidates)
            extras.append(
                "Output your answer as a comma-separated list. "
                "Each value must be one of the following candidates "
                "(use each candidate at most once):\n"
                f"{cand_block}"
            )
        else:
            extras.append("Output your answer as a comma-separated list.")
    elif candidates:
        cand_block = "\n".join(f"- {c}" for c in candidates)
        extras.append(
            "Choose exactly one of the following candidates as your answer:\n"
            f"{cand_block}"
        )

    if not extras:
        return text
    return text + "\n\n" + "\n\n".join(extras)


def with_dataset_prompt(text: str, dataset_prompt: str) -> str:
    """Append the dataset-level instruction to a question. Used ONLY at the
    final QA call so the memory-update loop stays on bare questions."""
    if not dataset_prompt:
        return text
    return text + "\n\n" + dataset_prompt


def build_chunks(item: dict, dataset: str, babi_facts_per_chunk: int) -> list[str]:
    """Turn an item's `contexts` list into a list of chunk strings.

    - babi: each context entry is one tiny fact; group `babi_facts_per_chunk`
      facts into a single chunk so the memory update sees a meaningful block.
    - github / horizonbench / wiki: each context entry is already a meaningful
      unit (a commit, a conversation turn, a wiki revision). One entry =
      one chunk. Prefix with timestamp when present so the model can reason
      about order.
    """
    contexts = item["contexts"]
    if dataset == "babi":
        chunks = []
        for i in range(0, len(contexts), babi_facts_per_chunk):
            group = contexts[i : i + babi_facts_per_chunk]
            chunks.append("\n".join(c["content"] for c in group))
        return chunks

    chunks = []
    for c in contexts:
        text = c["content"]
        ts = c.get("timestamp")
        if ts:
            text = f"[timestamp: {ts}]\n{text}"
        chunks.append(text)
    return chunks


async def post_chat(session, url, api_key, model, content, temperature, top_p, max_new):
    async with session.post(
        url=url + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "api-key": api_key},
        json=dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new,
        ),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"status={resp.status}, body={body[:500]}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def run_question(session, question_for_memory, question_for_qa, chunks, *,
                       mem_model, mem_url, mem_api_key,
                       qa_model, qa_url, qa_api_key,
                       temperature, top_p, max_new, verbose=False):
    """Memory-update loop uses `question_for_memory` (bare question + answer-
    shape hints). The final QA call uses `question_for_qa`, which additionally
    has the dataset-level addendum appended. Decoupling them keeps the
    dataset notes (e.g. horizonbench preference-event semantics) out of every
    per-chunk summarization step."""
    memory = NO_MEMORY
    for i, chunk in enumerate(chunks):
        msg = TEMPLATE.format(prompt=question_for_memory, chunk=chunk, memory=memory)
        if verbose and i == 0:
            print("user (first chunk):")
            print(clip_long_string(msg))
            print("-" * 20)
        memory = await post_chat(
            session, mem_url, mem_api_key, mem_model,
            msg, temperature, top_p, max_new,
        )
        if verbose and i == 0:
            print("assistant (updated memory):")
            print(clip_long_string(memory))
            print("=" * 40)

    final_msg = TEMPLATE_FINAL.format(prompt=question_for_qa, memory=memory)
    if verbose:
        print("user (final):")
        print(clip_long_string(final_msg))
        print("-" * 10)
    answer = await post_chat(
        session, qa_url, qa_api_key, qa_model,
        final_msg, temperature, top_p, max_new,
    )
    if verbose:
        print("assistant (final answer):")
        print(clip_long_string(answer))
        print("=" * 40)
    return answer, memory


async def process_item(session, item, dataset, *,
                       mem_model, mem_url, mem_api_key,
                       qa_model, qa_url, qa_api_key,
                       babi_facts_per_chunk, temperature, top_p, max_new,
                       dataset_prompt="",
                       verbose_item=False):
    chunks = build_chunks(item, dataset, babi_facts_per_chunk)
    rows = []
    for qi, q in enumerate(item["questions"]):
        question_for_memory = format_question(q)
        question_for_qa = with_dataset_prompt(question_for_memory, dataset_prompt)
        try:
            answer, memory = await run_question(
                session, question_for_memory, question_for_qa, chunks,
                mem_model=mem_model, mem_url=mem_url, mem_api_key=mem_api_key,
                qa_model=qa_model, qa_url=qa_url, qa_api_key=qa_api_key,
                temperature=temperature, top_p=top_p, max_new=max_new,
                verbose=(verbose_item and qi == 0),
            )
        except Exception:
            import traceback
            traceback.print_exc()
            answer, memory = "", ""
        rows.append({
            "item_id": item["id"],
            "question_idx": qi,
            "question": q["question"],
            "question_with_candidates": question_for_memory,
            "question_for_qa": question_for_qa,
            "gold": q["answer"],
            "question_type": q.get("question_type"),
            "metadata": q.get("metadata"),
            "predicted": answer,
            "final_memory": memory,
            "num_chunks": len(chunks),
        })
    return rows


async def amain(args):
    items, data_src = load_longmint_items(
        args.dataset, hf_repo=args.hf_repo, local_dir=args.data_root,
    )
    if args.limit is not None:
        items = items[: args.limit]

    out_path = Path(args.output) if args.output else Path(
        f"results_{args.dataset}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mem_model = args.model
    mem_url = args.url or URL
    mem_api_key = args.api_key or API_KEY
    qa_model = args.qa_model or args.model
    qa_url = args.qa_url or (QA_URL if args.qa_model else mem_url)
    qa_api_key = args.qa_api_key or (QA_API_KEY if args.qa_model else mem_api_key)

    dataset_prompt = load_dataset_prompt(args.dataset)

    print(f"dataset={args.dataset}  items={len(items)}  src={data_src}")
    print(f"memory model: {mem_model} @ {mem_url}")
    print(f"qa     model: {qa_model} @ {qa_url}")
    if args.dataset == "babi":
        print(f"babi_facts_per_chunk={args.babi_facts_per_chunk}")
    if dataset_prompt:
        print(f"dataset prompt: prompts/{args.dataset}.md "
              f"({len(dataset_prompt)} chars) -- appended to final QA only")
    print(f"writing -> {out_path}")

    sem = asyncio.Semaphore(args.concurrency)
    written_lock = asyncio.Lock()

    timeout = aiohttp.ClientTimeout(total=86400)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(out_path, "w") as out:

            async def worker(idx, item):
                async with sem:
                    rows = await process_item(
                        session, item, args.dataset,
                        mem_model=mem_model, mem_url=mem_url, mem_api_key=mem_api_key,
                        qa_model=qa_model, qa_url=qa_url, qa_api_key=qa_api_key,
                        babi_facts_per_chunk=args.babi_facts_per_chunk,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new=args.max_new,
                        dataset_prompt=dataset_prompt,
                        verbose_item=(args.verbose and idx == 0),
                    )
                    async with written_lock:
                        for r in rows:
                            out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                    return item["id"], len(rows)

            tasks = [worker(i, it) for i, it in enumerate(items)]
            done = 0
            for fut in asyncio.as_completed(tasks):
                item_id, n = await fut
                done += 1
                print(f"[{done}/{len(items)}] {item_id}  ({n} questions)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True,
                   help="Memory-management model (recurrent updates). "
                        "e.g. BytedTsinghua-SIA/RL-MemoryAgent-14B")
    p.add_argument("--qa-model", type=str, default="Qwen/Qwen3.6-35B-A3B",
                   help="Final QA model that reads the built memory and "
                        "produces the boxed answer "
                        "(default: Qwen/Qwen3.6-35B-A3B; matches the "
                        "memalpha / atommem / fullcontext / baserag / "
                        "hipporag runners). Pass the same name as --model "
                        "to reproduce the single-model behavior.")
    p.add_argument("--url", type=str, default=None,
                   help="Endpoint for the memory model. "
                        f"Defaults to $URL ({URL}).")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key for the memory model. Defaults to $API_KEY.")
    p.add_argument("--qa-url", type=str, default=None,
                   help="Endpoint for the QA model. Defaults to --url when "
                        "--qa-model is unset; otherwise to $QA_URL (or $URL).")
    p.add_argument("--qa-api-key", type=str, default=None,
                   help="API key for the QA model. Same fallback logic as --qa-url.")
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS,
                   help="Which unified dataset to run")
    p.add_argument("--data-root", type=str, default=None,
                   help="Optional: load <dataset>.json from this directory "
                        "instead of the HF Hub.")
    p.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO,
                   help="HF dataset repo to pull from when --data-root is "
                        f"not set (default: {DEFAULT_HF_REPO}).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Default: results_<dataset>.jsonl")
    p.add_argument("--babi-facts-per-chunk", type=int, default=15,
                   help="bAbI only: how many single-sentence facts to pack into "
                        "one memory-update chunk (default: 15). Other unified "
                        "datasets always use one chunk per context entry "
                        "(dialogue turn / commit / wiki revision).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N items (debug)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of items processed in parallel")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=8192,
                   help="max_tokens for both memory updates and final answer")
    p.add_argument("--verbose", action="store_true",
                   help="Print the first item's first-question trace")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
