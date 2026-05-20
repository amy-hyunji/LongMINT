import argparse
import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import load_raw_dataset

CHAT_URL = os.getenv("URL", "http://localhost:8000/v1")
QA_URL = os.getenv("QA_URL", CHAT_URL)
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:9007/v1")
API_KEY = os.getenv("API_KEY", "123-abc")
QA_API_KEY = os.getenv("QA_API_KEY", API_KEY)
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "sk-123")

DATASETS = ["babi", "github", "horizonbench", "wiki"]
# Local override; leave unset (= None) to fall back to the HF dataset
# (dinobby/MINTEval) the same way baserag/hipporag/memagent do.
DEFAULT_DATA_ROOT = os.environ.get("MINTEval_DATA_ROOT")

# Dataset-level instruction docs (per-dataset notes about data structure /
# answer conventions). Lives under src/memagent/prompts so the same file can
# be reused by both the memagent and atommem runners.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "memagent" / "prompts"

NO_MEMORY_TOKENS = "No previous memory"
DEFAULT_QUERY = "defult query, change it!"  # sic, matches upstream string

# Verbatim from recurrent/impls/memory_long_short_multiquery.py
SYSTEM_PROMPT = """You are presented with a section of an article and a previous memory. Please learn the provided section carefully and manage your memory to answer questions.

Your short-term memory is a step-wise updated summary, while your long-term memory is a vector database that can be updated through atomic operations.

short-term memory is updated using <update_memory>,you should use this every response.

four kinds of memory actions for database are available:
<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query.

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory.

<modify_memory>: updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify. To manage large volumes of memory effectively, you should prioritize using the "modify" function MORE frequently, rather than relying solely on "add" operations.

<delete_memory>: delete a memory. You need to enter the memory index "Memory i" to specify which memory to delete. You must delete duplicate memory entries!

Use paired XML tags as action markers so you can perform multiple actions—such as adding several memories—in a single response.

action example 1:
<update_query>dance partner; Yulia Zagoruychenko.</update_query>
action example 2:
<add_memory>
Document 10 indicates that the dance event took place in Moscow in October and that Yulia participated in it. I need to focus more on who else attended this event or who traveled to Moscow in October, in order to infer who Yulia's dance partner might be.
</add_memory>
action example 3:
<modify_memory>Memory 1: The current article provides updated competition records showing that Riccardo Cocchi is now partnered with Emily in the 2025 season, while no recent evidence confirms his continued partnership with Yulia Zagoruychenko. This conflicts with the previous memory stating that Yulia's partner is Riccardo. Since the information clearly supersedes the earlier record, the correct action is to modify the existing memory to reflect that Yulia's current dance partner is unknown as of 2025, and mark the entry for re-verification.</modify_memory>
action example 4:
It can be observed that Entry 2 and Entry 6 are largely duplicated. Since Entry 6 is more recent, I choose to delete Entry 2. <delete_memory>Memory 2</delete_memory>

Response example:
<update_memory>
**1. Memory Framework Overview**
The provided section describes a **memory-augmented reasoning framework** in which the agent maintains two complementary memory systems. **Short-term memory** functions as an evolving, step-wise summary that tracks the most immediate context, while **long-term memory** resides in a vector database capable of being updated through atomic operations.

**2. Memory Operations**
The text further introduces **four atomic memory actions**—**update_query**, **add_memory**, **modify_memory**, and **delete_memory**.
These collectively define how the agent:
- retrieves information through query management,
- adds new knowledge,
- updates outdated or inaccurate entries, and
- removes redundant or conflicting information.

**3. Functional Role of the Two Memory Types**
The mechanisms emphasize the importance of **active and continuous memory management**. Short-term memory captures high-level situational summaries that guide immediate reasoning, while long-term memory stores **fine-grained, reusable knowledge** that can support future inference, compensate for missing context, and improve the agent's overall consistency across steps.
</update_memory>

<update_query>
short-term vs long-term memory mechanism; atomic operations; memory workflow
</update_query>

<add_memory>
Long-term memory note: The system's memory architecture explicitly separates short-term and long-term roles. Short-term memory is updated every step and acts as a compressed running summary of what the agent has just read, inferred, or decided. In contrast, the long-term memory is a vector-database-backed store meant to hold more detailed, fine-grained, and broadly relevant information—such as definitions, recurring concepts, protocol rules, and any knowledge that may be useful across multiple future queries. The long-term store is maintained through atomic operations (query updating, adding new facts, modifying older entries, and deleting redundant ones), making it flexible and continually improvable as new context appears.
</add_memory>
"""

TEMPLATE_MEMORY = """
This is the question you need to solve:
{prompt}

This is your short term memory from the previous turn.
{short_memory}

This is current query to retrieve memory from database:
{query}

This is current memory related to the query:
{long_memory}

Tips:
 -DO NOT repeatly update query. If you don't have the desired memory, it means the entry does not exist in the knowledge base. AVOID using update_query multiple times within a single response, instead, you can use a long and composite query to retrieve document of different question. The query matches documents based on semantic embeddings, and composite queries are best composed of keywords.

This is the article:
{chunk}

Important:
Now you should follow this strategy to manage memory:
1) You should store only the most important information in short-term memory, while placing more detailed and broadly relevant information into long-term memory. This allows you to supplement missing information by retrieving it from long-term memory when needed.
2) The entries you write into the database using add_memory should contain information that your short-term memory tends to overlook—those with weaker relevance or even information that is entirely unrelated at the moment. The text stored in the database should not contain any reasoning or descriptions of the current state; it should consist solely of factual or knowledge-based information. Add at least 5-6 unrelated knowledge into the database.
3) When confronted with multiple question at the time, discuss the relevant information for each problem separately (using 1. 2. 3.) within your short-term memory.

Focus on your query times:
You have update query **{query_times}** times, exceed **3** will lead to the task failure.
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Based on the memory, use '<final_answer></final_answer>' to answer the problem in \\boxed{{}}. You can use update query to get memory.

Tips:
AVOID using update_query multiple times within a single response, instead, you can use a long and composite query to retrieve document of different question. The query matches documents based on semantic embeddings, and composite queries are best composed of keywords. When you choose update_query, the process will be frozen, meaning you can answer the question in the next round. Do not update memory in this stage!

problem:
{prompt}

This is your short term memory from the previous turn.
{short_memory}

This is current query to retrieve memory from database:
{query}

This is current memory related to the query:
{long_memory}

Output the answer only as an exact match—do not add any description.
Example 1:
<update_query>The dance partner; Alice.</update_query>
Example 2:
<final_answer>\\boxed{{Jacob}}</final_answer>.

Focus on your query times:
You have update query **{query_times}** times, exceed **3** will lead to the task failure. When you are about to exceed the limit, use <final_answer>don't know</final_answer> to skip this question.
"""

PAIRED_TAG_RE = re.compile(r"<(\w+)[^>]*>(.*?)</\1>", flags=re.S | re.I)
MEMORY_INDEX_RE = re.compile(r"Memory\s+(\d+)\s*:?\s*(.*)", flags=re.S)
BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", flags=re.S)


def clip_long_string(s: str, max_length: int = 2000) -> str:
    if len(s) <= max_length:
        return s
    target = max_length - len("\n\n...(truncated)\n\n")
    return s[: target // 2] + "\n\n...(truncated)\n\n" + s[-target // 2 :]


def load_dataset_instruction(dataset: str) -> Optional[str]:
    """Read PROMPTS_DIR/<dataset>.md if it exists, else None.

    Returned text is appended verbatim (after a blank-line separator) to the
    final-QA user message ONLY — the memory-building loop is intentionally
    left bare so the dataset notes don't bias every per-chunk summarization.
    Today only horizonbench.md ships, but adding babi.md / wiki.md /
    github.md later auto-enables the same wiring.
    """
    p = PROMPTS_DIR / f"{dataset}.md"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8").strip()


def append_dataset_instruction(user_msg: str, instruction: Optional[str]) -> str:
    if not instruction:
        return user_msg
    return f"{user_msg}\n\n{instruction}"


def format_question_for_prompt(q: dict) -> str:
    """Render a question for the model, including its candidate list when
    one is present in metadata (horizonbench has these on ~85% of items).

    We don't add a separate EM grading note — AtomMem's TEMPLATE_FINAL_BOXED
    already instructs the model to wrap its final answer in
    `<final_answer>\\boxed{...}</final_answer>`, and our extractor pulls the
    answer back out of `\\boxed{}`, mirroring MemAgent's convention. So
    candidates alone are enough — the format hook is handled by the
    upstream template, not by us re-explaining EM grading in every question.
    """
    text = q["question"]
    meta = q.get("metadata") or {}
    candidates = meta.get("candidates") or []
    is_ordering = q.get("question_type") == "ordering"

    parts = [text]
    if candidates:
        parts.append(
            "Each value in your answer must be exactly one of the following candidates, "
            "copied verbatim (use each candidate at most once):"
            if is_ordering else
            "The answer must be exactly one of the following candidates, copied verbatim:"
        )
        for c in candidates:
            parts.append(f"  - {c}")
    if is_ordering:
        parts.append("Output your answer as a comma-separated list.")

    if len(parts) == 1:
        return text
    return "\n".join(parts)


def format_all_questions(item: dict) -> str:
    """Concatenate every question on the item for the memory-building phase.

    Mirrors upstream's `'\\n'.join(self.prompt[idx])`, but each entry is the
    full formatted block (text + candidates + EM note) so the memory agent
    knows during ingestion which exact tokens it must surface later.
    """
    return "\n\n".join(format_question_for_prompt(q) for q in item["questions"])


def build_chunks(item: dict, dataset: str, babi_facts_per_chunk: int) -> list[str]:
    """Turn an item's `contexts` list into the chunks the memory agent sees.

    - horizonbench / github / wiki: one chunk per context entry
      (= one dialogue turn / commit / wiki revision). The unified data is
      already chunked at a meaningful semantic boundary, so we don't
      re-bundle. Each chunk is prefixed with `[timestamp: ...]` when
      present so the agent can reason about temporal ordering.
    - babi: each entry is a one-sentence fact; pack `babi_facts_per_chunk`
      facts into a single chunk so each memory-update turn sees a
      meaningful block instead of one sentence at a time.
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


async def post_chat(session, model, messages, temperature, top_p, max_new,
                    *, url: Optional[str] = None, api_key: Optional[str] = None):
    """OpenAI-compatible /chat/completions call. Endpoint and key default to
    the memory-model URL/API_KEY; pass `url=` / `api_key=` to target the QA
    endpoint instead."""
    target_url = url if url is not None else CHAT_URL
    target_key = api_key if api_key is not None else API_KEY
    async with session.post(
        url=target_url + "/chat/completions",
        headers={"Authorization": f"Bearer {target_key}", "api-key": target_key},
        json=dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new,
        ),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat status={resp.status}, body={body[:500]}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def post_embed(session, model, texts: list[str]) -> np.ndarray:
    async with session.post(
        url=EMBED_URL + "/embeddings",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}", "api-key": EMBED_API_KEY},
        json={"model": model, "input": texts},
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"embed status={resp.status}, body={body[:500]}")
        data = await resp.json()
    arr = np.asarray([d["embedding"] for d in data["data"]], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


class MemoryStore:
    """Tiny per-item vector store. Holds at most max_chunks entries in
    practice (one add_memory per chunk in the worst case), so a numpy
    cosine scan is faster than spinning up FAISS.
    """

    QUERY_INSTRUCTION_TMPL = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery:{query}"
    )

    def __init__(self, session, embed_model: str):
        self.session = session
        self.embed_model = embed_model
        self.contents: list[str] = []
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []

    async def add(self, content: str) -> None:
        content = content.strip()
        if not content:
            return
        vec = await post_embed(self.session, self.embed_model, [content])
        self.contents.append(content)
        self.ids.append(str(uuid.uuid4()))
        self.vectors.append(vec[0])

    async def delete_by_content(self, old_content: str) -> bool:
        """Match upstream `modify_document` which deletes by exact content."""
        for i, c in enumerate(self.contents):
            if c == old_content:
                self.contents.pop(i)
                self.ids.pop(i)
                self.vectors.pop(i)
                return True
        return False

    async def retrieve(self, query: str, k: int = 6) -> list[str]:
        if not self.contents:
            return []
        qtext = self.QUERY_INSTRUCTION_TMPL.format(query=query)
        qvec = await post_embed(self.session, self.embed_model, [qtext])
        mat = np.stack(self.vectors, axis=0)
        scores = mat @ qvec[0]
        order = np.argsort(-scores)[:k]
        return [self.contents[i] for i in order]

    def __len__(self):
        return len(self.contents)


def extract_paired_tags(text: str) -> list[tuple[str, str]]:
    return PAIRED_TAG_RE.findall(text)


def extract_boxed(text: str) -> Optional[str]:
    m = BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def truncate_by_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


async def memory_phase(session, model, store: MemoryStore, *,
                       all_questions_str: str, chunks: list[str],
                       max_chunks: int, temperature: float, top_p: float,
                       max_new: int, memory_char_budget: int,
                       verbose: bool = False) -> dict:
    
    short_memory = ""
    query = DEFAULT_QUERY
    query_times = 0
    last_retrieved: list[str] = []
    trace = []

    chunk_idx = 0
    for agent_step in range(max_chunks):
        if chunk_idx >= len(chunks):
            break
        retrieved = await store.retrieve(query, k=6)
        last_retrieved = retrieved
        memory_text = (
            "\n".join(f"Memory {i + 1}: {c}" for i, c in enumerate(retrieved))
            if retrieved
            else NO_MEMORY_TOKENS
        )
        memory_text = truncate_by_chars(memory_text, memory_char_budget)

        user_msg = TEMPLATE_MEMORY.format(
            prompt=all_questions_str,
            short_memory=short_memory,
            query=query,
            long_memory=memory_text,
            chunk=chunks[chunk_idx],
            query_times=query_times,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        if verbose and agent_step == 0:
            print("user (first chunk):")
            print(clip_long_string(user_msg))
            print("-" * 20)

        response = await post_chat(session, model, messages, temperature, top_p, max_new)
        if verbose and agent_step == 0:
            print("assistant (first chunk):")
            print(clip_long_string(response))
            print("=" * 40)

        tags = extract_paired_tags(response)
        move_on = False  # advance to next chunk only on add/modify
        for tag_name, content in tags:
            t = tag_name.lower()
            content_s = content.strip()
            if t == "update_memory":
                short_memory = content_s
                query_times = 0
            elif t == "update_query":
                if content_s:
                    query = content_s
                query_times += 1
            elif t == "add_memory":
                await store.add(content_s)
                query_times = 0
                move_on = True
            elif t == "modify_memory":
                m = MEMORY_INDEX_RE.match(content_s)
                if m:
                    idx = int(m.group(1)) - 1
                    new_c = m.group(2).strip()
                    if 0 <= idx < len(last_retrieved):
                        await store.delete_by_content(last_retrieved[idx])
                        if new_c:
                            await store.add(new_c)
                query_times = 0
                move_on = True
            elif t == "delete_memory":
                m = MEMORY_INDEX_RE.match(content_s)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(last_retrieved):
                        await store.delete_by_content(last_retrieved[idx])
                query_times = 0

        if move_on:
            chunk_idx += 1

        trace.append({
            "agent_step": agent_step,
            "chunk_idx": chunk_idx,
            "n_tags": len(tags),
            "advanced": move_on,
            "query_times": query_times,
            "db_size": len(store),
        })

    return {
        "short_memory": short_memory,
        "query": query,
        "chunks_consumed": chunk_idx,
        "trace": trace,
    }


async def answer_question(session, model, store: MemoryStore, question: str, *,
                          short_memory: str, query: str,
                          temperature: float, top_p: float, max_new: int,
                          memory_char_budget: int,
                          max_query_updates: int = 3,
                          dataset_instruction: Optional[str] = None,
                          qa_url: Optional[str] = None,
                          qa_api_key: Optional[str] = None,
                          verbose: bool = False) -> dict:
    
    query_times = 0
    final_answer_raw = ""
    boxed = None
    last_response = ""
    rounds = []

    for attempt in range(max_query_updates + 1):
        retrieved = await store.retrieve(query, k=6)
        memory_text = (
            "\n".join(f"Memory {i + 1}: {c}" for i, c in enumerate(retrieved))
            if retrieved
            else NO_MEMORY_TOKENS
        )
        memory_text = truncate_by_chars(memory_text, memory_char_budget)

        user_msg = TEMPLATE_FINAL_BOXED.format(
            prompt=question,
            short_memory=short_memory,
            query=query,
            long_memory=memory_text,
            query_times=query_times,
        )
        user_msg = append_dataset_instruction(user_msg, dataset_instruction)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        if verbose:
            print(f"user (final attempt {attempt}):")
            print(clip_long_string(user_msg))
            print("-" * 10)

        response = await post_chat(
            session, model, messages, temperature, top_p, max_new,
            url=qa_url, api_key=qa_api_key,
        )
        last_response = response
        if verbose:
            print(f"assistant (final attempt {attempt}):")
            print(clip_long_string(response))
            print("=" * 40)

        tags = extract_paired_tags(response)
        got_final = False
        updated_query = False
        for tag_name, content in tags:
            t = tag_name.lower()
            content_s = content.strip()
            if t == "final_answer":
                final_answer_raw = content_s
                boxed = extract_boxed(content_s)
                got_final = True
                break
            elif t == "update_query":
                if content_s:
                    query = content_s
                query_times += 1
                updated_query = True

        rounds.append({
            "query": query,
            "query_times": query_times,
            "got_final": got_final,
        })

        if got_final:
            break
        if not updated_query:
            break

    return {
        "final_answer_raw": final_answer_raw,
        "boxed": boxed,
        "predicted": boxed if boxed is not None else final_answer_raw,
        "last_response": last_response,
        "final_query": query,
        "final_rounds": rounds,
    }


async def process_item(session, chat_model, embed_model, item, dataset, *,
                       babi_facts_per_chunk, max_chunks,
                       temperature, top_p, max_new, memory_char_budget,
                       max_query_updates,
                       dataset_instruction: Optional[str] = None,
                       qa_model: Optional[str] = None,
                       qa_url: Optional[str] = None,
                       qa_api_key: Optional[str] = None,
                       verbose_item: bool = False):
    chunks = build_chunks(item, dataset, babi_facts_per_chunk)
    all_questions_str = format_all_questions(item)
    store = MemoryStore(session, embed_model)

    mem_result = await memory_phase(
        session, chat_model, store,
        all_questions_str=all_questions_str,
        chunks=chunks,
        max_chunks=max_chunks,
        temperature=temperature, top_p=top_p, max_new=max_new,
        memory_char_budget=memory_char_budget,
        verbose=verbose_item,
    )

    short_memory = mem_result["short_memory"]
    query = mem_result["query"]
    db_size_after_memory = len(store)
    chunks_consumed = mem_result["chunks_consumed"]

    rows = []
    for qi, q in enumerate(item["questions"]):
        question_text = format_question_for_prompt(q)
        try:
            ans = await answer_question(
                session, qa_model or chat_model, store, question_text,
                short_memory=short_memory, query=query,
                temperature=temperature, top_p=top_p, max_new=max_new,
                memory_char_budget=memory_char_budget,
                max_query_updates=max_query_updates,
                dataset_instruction=dataset_instruction,
                qa_url=qa_url,
                qa_api_key=qa_api_key,
                verbose=(verbose_item and qi == 0),
            )
            
            query = ans["final_query"]
        except Exception:
            import traceback
            traceback.print_exc()
            ans = {
                "final_answer_raw": "",
                "boxed": None,
                "predicted": "",
                "last_response": "",
                "final_query": query,
                "final_rounds": [],
            }

        rows.append({
            "item_id": item["id"],
            "question_idx": qi,
            "question": q["question"],
            "gold": q["answer"],
            "question_type": q.get("question_type"),
            "metadata": q.get("metadata"),
            "predicted": ans["predicted"],
            "boxed": ans["boxed"],
            "final_answer_raw": ans["final_answer_raw"],
            "last_response": ans["last_response"],
            "short_memory": short_memory,
            "final_query": ans["final_query"],
            "db_size_after_memory": db_size_after_memory,
            "db_size_final": len(store),
            "num_chunks": len(chunks),
            "num_chunks_consumed": chunks_consumed,
            "memory_trace": mem_result["trace"],
            "final_rounds": ans["final_rounds"],
        })
    return rows


async def amain(args):
    items, data_src = load_raw_dataset(args.dataset, args.data_root)
    if args.limit is not None:
        items = items[: args.limit]

    out_path = Path(args.output) if args.output else Path(
        f"results_atommem_{args.dataset}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_instruction = load_dataset_instruction(args.dataset)

    qa_model = args.qa_model or args.model
    qa_split = (qa_model != args.model) or (QA_URL != CHAT_URL)

    print(f"dataset={args.dataset}  items={len(items)}  src={data_src}")
    print(f"memory chat URL={CHAT_URL}  model={args.model}")
    if qa_split:
        print(f"QA     chat URL={QA_URL}  model={qa_model}")
    else:
        print(f"QA     chat URL={QA_URL}  model={qa_model}  (== memory endpoint)")
    print(f"embed URL={EMBED_URL}  model={args.embed_model}")
    print(f"max_chunks={args.max_chunks}  max_query_updates={args.max_query_updates}")
    if args.dataset == "babi":
        print(f"babi_facts_per_chunk={args.babi_facts_per_chunk}")
    if dataset_instruction is not None:
        print(f"dataset instruction: loaded {len(dataset_instruction)} chars from "
              f"{PROMPTS_DIR / (args.dataset + '.md')} (appended to final QA only)")
    else:
        print(f"dataset instruction: none (no {args.dataset}.md under {PROMPTS_DIR})")
    print(f"writing -> {out_path}")

    sem = asyncio.Semaphore(args.concurrency)
    written_lock = asyncio.Lock()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=86400)
    ) as session, open(out_path, "w") as out:

        async def worker(idx, item):
            async with sem:
                rows = await process_item(
                    session, args.model, args.embed_model, item, args.dataset,
                    babi_facts_per_chunk=args.babi_facts_per_chunk,
                    max_chunks=args.max_chunks,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new=args.max_new,
                    memory_char_budget=args.memory_char_budget,
                    max_query_updates=args.max_query_updates,
                    dataset_instruction=dataset_instruction,
                    qa_model=qa_model,
                    qa_url=QA_URL,
                    qa_api_key=QA_API_KEY,
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
                   help="Memory-builder model name registered at $URL "
                        "(--served_model_name on the AtomMem vLLM). Drives "
                        "the per-chunk atomic CRUD memory updates. Example: "
                        "'AtomMem'.")
    p.add_argument("--qa-model", type=str, default="Qwen/Qwen3.6-35B-A3B",
                   help="Final-answer model registered at $QA_URL "
                        "(default: Qwen/Qwen3.6-35B-A3B). When $QA_URL is "
                        "unset it falls back to $URL, so passing the same "
                        "name as --model reproduces upstream AtomMem's "
                        "single-model behavior.")
    p.add_argument("--embed-model", type=str, default="qwen3-embedding",
                   help="Embedding model name at the EMBED_URL endpoint "
                        "(default: qwen3-embedding)")
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS,
                   help="Which unified dataset to run")
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="Local directory containing <dataset>.json files. "
                        "Leave unset (default) to load the split from the "
                        "HF dataset dinobby/MINTEval, or set env "
                        "MINTEval_DATA_ROOT to override.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. "
                        "Default: results_atommem_<dataset>.jsonl")
    p.add_argument("--babi-facts-per-chunk", type=int, default=15,
                   help="bAbI only: how many single-sentence facts to pack "
                        "into one memory-update chunk (default: 15). Other "
                        "datasets always use one chunk per context entry "
                        "(dialogue turn / commit / wiki revision).")
    p.add_argument("--max-chunks", type=int, default=40,
                   help="Cap on memory-building steps per item (default: 40, "
                        "matches AtomMem's eval config)")
    p.add_argument("--max-query-updates", type=int, default=3,
                   help="Max consecutive <update_query> rounds per question "
                        "in the final stage (default: 3, matches upstream)")
    p.add_argument("--memory-char-budget", type=int, default=16000,
                   help="Char-level truncation cap applied to the retrieved "
                        "long-memory block before formatting into the prompt "
                        "(default: 16000, ~4k tokens)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N items (debug)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of items processed in parallel")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=2048,
                   help="max_tokens for each chat completion "
                        "(memory updates and final answers)")
    p.add_argument("--verbose", action="store_true",
                   help="Print the first item's first-chunk and first-question trace")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
