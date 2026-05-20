import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import load_raw_dataset

URL = os.getenv("URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "123-abc")

DATASETS = ["babi", "github", "horizonbench", "wiki"]
DEFAULT_DATA_ROOT = os.environ.get("MINTEval_DATA_ROOT")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "memagent" / "prompts"


def load_dataset_prompt(dataset: str) -> str:
    
    path = PROMPTS_DIR / f"{dataset}.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()

TEMPLATE = """You are given the full context for a problem. Read the context carefully and put your final answer inside \\boxed{{}}.

<context>
{context}
</context>

<problem>
{prompt}
</problem>

Your answer:
"""

TRUNCATION_MARKER = "\n\n...(context truncated)...\n\n"


def clip_long_string(s: str, max_length: int = 2000) -> str:
    if len(s) <= max_length:
        return s
    target = max_length - len("\n\n...(truncated)\n\n")
    return s[: target // 2] + "\n\n...(truncated)\n\n" + s[-target // 2 :]


def format_question(q: dict) -> str:
    
    text = q["question"]
    candidates = (q.get("metadata") or {}).get("candidates")
    is_ordering = q.get("question_type") == "ordering"
    if not candidates:
        if is_ordering:
            return f"{text}\n\nOutput your answer as a comma-separated list."
        return text
    cand_block = "\n".join(f"- {c}" for c in candidates)
    if is_ordering:
        instr = (
            "Output your answer as a comma-separated list. "
            "Each value must be one of the following candidates:"
        )
    else:
        instr = "Choose exactly one of the following candidates as your answer:"
    return f"{text}\n\n{instr}\n{cand_block}"


def with_dataset_prompt(text: str, dataset_prompt: str) -> str:
    
    if not dataset_prompt:
        return text
    return text + "\n\n" + dataset_prompt


def build_full_context(item: dict, dataset: str) -> str:
    
    contexts = item["contexts"]
    if dataset == "babi":
        return "\n".join(c["content"] for c in contexts)
    parts = []
    for c in contexts:
        text = c["content"]
        ts = c.get("timestamp")
        if ts:
            text = f"[timestamp: {ts}]\n{text}"
        parts.append(text)
    return "\n\n".join(parts)


def _tokenize_root(url: str) -> str:
    
    u = url.rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u


async def post_tokenize(session, url, api_key, model, text):
    """vLLM /tokenize. Returns the list of token ids."""
    async with session.post(
        url=_tokenize_root(url) + "/tokenize",
        headers={"Authorization": f"Bearer {api_key}", "api-key": api_key},
        json={"model": model, "prompt": text},
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"tokenize status={resp.status}, body={body[:500]}")
        data = await resp.json()
        return data["tokens"]


async def post_detokenize(session, url, api_key, model, tokens):
    """vLLM /detokenize. Returns the decoded prompt text."""
    async with session.post(
        url=_tokenize_root(url) + "/detokenize",
        headers={"Authorization": f"Bearer {api_key}", "api-key": api_key},
        json={"model": model, "tokens": tokens},
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"detokenize status={resp.status}, body={body[:500]}")
        data = await resp.json()
        return data["prompt"]


async def fit_prompt_to_window(session, url, api_key, model, *,
                                context, question_text,
                                vllm_max_len, max_new, safety_margin):
    
    prompt = TEMPLATE.format(context=context, prompt=question_text)
    prompt_tokens = await post_tokenize(session, url, api_key, model, prompt)
    budget = vllm_max_len - max_new - safety_margin
    if len(prompt_tokens) <= budget:
        return prompt, len(prompt_tokens), False

    overage = len(prompt_tokens) - budget
    context_tokens = await post_tokenize(session, url, api_key, model, context)
    
    target = max(0, len(context_tokens) - overage - 64)
    if target == 0:
        new_context = TRUNCATION_MARKER.strip()
    else:
        half = target // 2
        tail = target - half
        head_text = await post_detokenize(
            session, url, api_key, model, context_tokens[:half]
        )
        tail_text = await post_detokenize(
            session, url, api_key, model, context_tokens[-tail:]
        )
        new_context = head_text + TRUNCATION_MARKER + tail_text

    new_prompt = TEMPLATE.format(context=new_context, prompt=question_text)
    new_prompt_tokens = await post_tokenize(session, url, api_key, model, new_prompt)
    
    if len(new_prompt_tokens) > budget:
        excess = len(new_prompt_tokens) - budget
        target2 = max(0, target - excess - 32)
        if target2 == 0:
            new_context = TRUNCATION_MARKER.strip()
        else:
            half = target2 // 2
            tail = target2 - half
            head_text = await post_detokenize(
                session, url, api_key, model, context_tokens[:half]
            )
            tail_text = await post_detokenize(
                session, url, api_key, model, context_tokens[-tail:]
            )
            new_context = head_text + TRUNCATION_MARKER + tail_text
        new_prompt = TEMPLATE.format(context=new_context, prompt=question_text)
        new_prompt_tokens = await post_tokenize(session, url, api_key, model, new_prompt)
    return new_prompt, len(new_prompt_tokens), True


async def post_chat(session, url, api_key, model, content, temperature, top_p, max_new,
                    disable_thinking=False):
    body = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new,
    )
    if disable_thinking:
        
        body["chat_template_kwargs"] = {"enable_thinking": False}
    async with session.post(
        url=url + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "api-key": api_key},
        json=body,
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"status={resp.status}, body={body[:500]}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def process_item(session, item, dataset, *,
                       model, url, api_key,
                       temperature, top_p, max_new,
                       vllm_max_len, safety_margin,
                       dataset_prompt="",
                       disable_thinking=False,
                       verbose_item=False):
    raw_context = build_full_context(item, dataset)
    rows = []
    for qi, q in enumerate(item["questions"]):
        question_text = with_dataset_prompt(format_question(q), dataset_prompt)
        try:
            prompt, prompt_tokens, truncated = await fit_prompt_to_window(
                session, url, api_key, model,
                context=raw_context, question_text=question_text,
                vllm_max_len=vllm_max_len, max_new=max_new,
                safety_margin=safety_margin,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            rows.append({
                "item_id": item["id"],
                "question_idx": qi,
                "question": q["question"],
                "question_with_candidates": question_text,
                "gold": q["answer"],
                "question_type": q.get("question_type"),
                "metadata": q.get("metadata"),
                "predicted": "",
                "prompt_tokens": 0,
                "context_truncated": False,
                "error": "fit_prompt_to_window failed",
            })
            continue

        if verbose_item and qi == 0:
            print(f"prompt_tokens={prompt_tokens}  truncated={truncated}")
            print("user:")
            print(clip_long_string(prompt))
            print("-" * 20)
        try:
            answer = await post_chat(
                session, url, api_key, model,
                prompt, temperature, top_p, max_new,
                disable_thinking=disable_thinking,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            answer = ""
        if verbose_item and qi == 0:
            print("assistant:")
            print(clip_long_string(answer))
            print("=" * 40)
        rows.append({
            "item_id": item["id"],
            "question_idx": qi,
            "question": q["question"],
            "question_with_candidates": question_text,
            "gold": q["answer"],
            "question_type": q.get("question_type"),
            "metadata": q.get("metadata"),
            "predicted": answer,
            "prompt_tokens": prompt_tokens,
            "context_truncated": truncated,
        })
    return rows


async def amain(args):
    items, data_src = load_raw_dataset(args.dataset, args.data_root)
    if args.limit is not None:
        items = items[: args.limit]

    out_path = Path(args.output) if args.output else Path(
        f"results_fullcontext_{args.dataset}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = args.url or URL
    api_key = args.api_key or API_KEY

    dataset_prompt = load_dataset_prompt(args.dataset)

    print(f"dataset={args.dataset}  items={len(items)}  src={data_src}")
    print(f"model: {args.model} @ {url}")
    print(f"vllm_max_len={args.vllm_max_len}  max_new={args.max_new}  "
          f"safety_margin={args.safety_margin}  "
          f"=> context+prompt budget={args.vllm_max_len - args.max_new - args.safety_margin} tokens")
    if dataset_prompt:
        print(f"dataset prompt: {PROMPTS_DIR}/{args.dataset}.md "
              f"({len(dataset_prompt)} chars) -- appended to QA call")
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
                        model=args.model, url=url, api_key=api_key,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new=args.max_new,
                        vllm_max_len=args.vllm_max_len,
                        safety_margin=args.safety_margin,
                        dataset_prompt=dataset_prompt,
                        disable_thinking=args.disable_thinking,
                        verbose_item=(args.verbose and idx == 0),
                    )
                    async with written_lock:
                        for r in rows:
                            out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                    return item["id"], len(rows), sum(1 for r in rows if r.get("context_truncated"))

            tasks = [worker(i, it) for i, it in enumerate(items)]
            done = 0
            total_truncated = 0
            for fut in asyncio.as_completed(tasks):
                item_id, n, n_trunc = await fut
                done += 1
                total_truncated += n_trunc
                tag = f"  trunc={n_trunc}/{n}" if n_trunc else ""
                print(f"[{done}/{len(items)}] {item_id}  ({n} questions){tag}")
            print(f"truncated rows: {total_truncated}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True,
                   help="QA model that reads the full context and produces "
                        "the boxed answer. e.g. Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--url", type=str, default=None,
                   help=f"Endpoint for the QA model. Defaults to $URL ({URL}).")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key for the QA model. Defaults to $API_KEY.")
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS,
                   help="Which unified dataset to run")
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="Local directory containing <dataset>.json files. "
                        "Leave unset (default) to load the split from the "
                        "HF dataset dinobby/MINTEval, or set env "
                        "MINTEval_DATA_ROOT to override.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Default: results_fullcontext_<dataset>.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N items (debug)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of items processed in parallel")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=1024,
                   help="max_tokens for the answer call")
    p.add_argument("--vllm-max-len", type=int, default=65536,
                   help="vLLM engine's --max-model-len (default 65536 = 2**16). "
                        "Prompts whose token count + max_new + safety_margin "
                        "exceed this value get their context head+tail truncated.")
    p.add_argument("--safety-margin", type=int, default=256,
                   help="Token buffer for chat-template overhead and "
                        "tokenization fuzz when re-joining head/tail slices "
                        "(default: 256).")
    p.add_argument("--disable-thinking", action="store_true",
                   help="For Qwen3 / Qwen3.6 hybrid-thinking models, suppress "
                        "the reasoning trace by sending "
                        "chat_template_kwargs.enable_thinking=False. Same "
                        "knob as tests_baserag.py's --disable_thinking.")
    p.add_argument("--verbose", action="store_true",
                   help="Print the first item's first-question trace")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
