import argparse
import atexit
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from openai_embedding_client import OpenAIEmbeddingClient, load_api_key

from _common import (
    HF_SPLIT_MAP,
    SYSTEM_PROMPT,
    apply_shard_and_limit,
    build_qlist,
    build_question_block,
    build_user_prompt,
    em_f1,
    extract_boxed,
    init_pred_dict,
    load_dataset_prompt,
    load_dataset_unified,
    resume_pred_dicts,
    save_atomic,
    truncate_user_prompt,
)


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Dense-retrieval BaseRAG over a unified-format MINTEval dataset.")

    # I/O
    p.add_argument("--dataset", required=True, choices=sorted(HF_SPLIT_MAP),
                   help="Dataset name (babi | wiki | github | horizonbench). "
                        "Selects the corresponding split when loading from "
                        "Hugging Face, and tags the output filename.")
    p.add_argument("--data_path", default=None,
                   help="Optional path to a local unified-format JSON file. "
                        "When omitted, the dataset is loaded from the "
                        "dinobby/MINTEval Hugging Face dataset.")
    p.add_argument("--output_dir", default="outputs",
                   help="Base output directory. Final path is "
                        "<output_dir>/<dataset>/<llm_suffix>/<basename>.json.")

    # LLM (QA)
    p.add_argument("--llm_model_name", required=True)
    p.add_argument("--llm_base_url", required=True,
                   help="OpenAI-compatible base URL of the served LLM.")
    p.add_argument("--llm_api_key", default="EMPTY")
    p.add_argument("--max_tokens", type=int, default=4096,
                   help="Completion-token budget for the QA call.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_input_chars", type=int, default=85000,
                   help="Char-budget cap for the user message of each chat "
                        "call. 0 = disabled.")
    p.add_argument("--disable_thinking", type=int, default=1,
                   help="When 1, pass Qwen's `chat_template_kwargs="
                        "{'enable_thinking': False}` via extra_body so the "
                        "server skips the <think>...</think> trace.")
    p.add_argument("--llm_concurrency", type=int, default=32,
                   help="Thread fanout for QA calls per instance.")

    # Embedding
    p.add_argument("--embedding_model_name", required=True,
                   help="HF model id (sentence-transformers / Transformers) "
                        "for --embedder_provider=hf, or the API model name "
                        "for openai.")
    p.add_argument("--embedder_provider", choices=["hf", "openai"], default="hf")
    p.add_argument("--embedding_model_dtype", default="bfloat16",
                   choices=["float16", "float32", "bfloat16", "auto"])
    p.add_argument("--encode_batch_size", type=int, default=1,
                   help="Per-GPU batch inside encode(). Default 1 is safe "
                        "next to a co-located vLLM.")
    p.add_argument("--embed_devices", default="",
                   help="Comma-separated CUDA device ids for multi-GPU pool "
                        "(e.g. '0,1'). Empty = single device (auto).")
    # OpenAI-compatible embedder
    p.add_argument("--embedder_base_url", default="",
                   help="Used only when --embedder_provider=openai (e.g. "
                        "https://generativelanguage.googleapis.com/v1beta/openai/).")
    p.add_argument("--embedder_api_key_file", default="",
                   help="Path to file containing the embedder API key. "
                        "Falls back to $EMBEDDER_API_KEY env var if empty/missing.")
    p.add_argument("--embedder_concurrency", type=int, default=8,
                   help="Concurrent embedding API requests per process.")
    p.add_argument("--embedder_batch_size", type=int, default=50,
                   help="Inputs per embedding API request.")
    p.add_argument("--embedder_output_dim", type=int, default=0,
                   help="If >0, request reduced output dim (Gemini supports "
                        "MRL via `dimensions=`). 0 = default.")
    p.add_argument("--embedder_cache_db", default="",
                   help="Optional sqlite path for embedding cache. Empty = "
                        "in-memory.")

    # Retrieval / corpus
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_doc_chars", type=int, default=0,
                   help="Char cap for each rendered context doc. 0 = disabled.")

    # Run control
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)

    args = p.parse_args()
    if not (0 <= args.shard_idx < args.num_shards):
        p.error(f"shard_idx={args.shard_idx} must be in [0, {args.num_shards})")
    if args.embedder_provider == "openai" and not args.embedder_base_url:
        p.error("--embedder_base_url is required when --embedder_provider=openai")
    return args


def build_embedder(args):
    """Returns (embedder, embed_pool) where `embed_pool` is the
    multi-process pool handle (or None)."""
    if args.embedder_provider == "openai":
        api_key = load_api_key(args.embedder_api_key_file) if args.embedder_api_key_file \
                  else os.environ.get("EMBEDDER_API_KEY", "")
        if not api_key:
            raise SystemExit(
                "embedder_provider=openai but no API key: pass "
                "--embedder_api_key_file or set $EMBEDDER_API_KEY."
            )
        print(f"Loading embedder (API): {args.embedding_model_name} "
              f"@ {args.embedder_base_url} (concurrency={args.embedder_concurrency}, "
              f"batch={args.embedder_batch_size}, output_dim="
              f"{args.embedder_output_dim or 'default'}, cache="
              f"{args.embedder_cache_db or 'in-memory'})")
        embedder = OpenAIEmbeddingClient(
            model=args.embedding_model_name,
            base_url=args.embedder_base_url,
            api_key=api_key,
            max_concurrency=args.embedder_concurrency,
            batch_size=args.embedder_batch_size,
            output_dim=args.embedder_output_dim or None,
            cache_db=args.embedder_cache_db or None,
        )
        return embedder, None

    print(f"Loading embedder: {args.embedding_model_name} "
          f"(dtype={args.embedding_model_dtype})")
    devices = [d.strip() for d in args.embed_devices.split(",") if d.strip()]
    target_devices = [f"cuda:{d}" if d.isdigit() else d for d in devices]
    torch_dtype = _DTYPE_MAP[args.embedding_model_dtype]
    embedder = SentenceTransformer(
        args.embedding_model_name,
        trust_remote_code=True,
        device="cpu" if target_devices else None,
        model_kwargs={"torch_dtype": torch_dtype} if torch_dtype != "auto" else None,
    )
    embedder.eval()

    pool = None
    if target_devices:
        print(f"Multi-GPU embedding across: {target_devices}")
        pool = embedder.start_multi_process_pool(target_devices=target_devices)
        atexit.register(embedder.stop_multi_process_pool, pool)
    return embedder, pool


def encode_pair(embedder, embed_pool, docs, queries, batch_size):
    """Encode `docs` and `queries` into L2-normalized matrices and return
    the similarity matrix (queries × docs)."""
    if isinstance(embedder, OpenAIEmbeddingClient):
        doc_embs = embedder.encode(docs, normalize_embeddings=True)
        q_embs = embedder.encode(queries, normalize_embeddings=True)
        return q_embs @ doc_embs.T

    if embed_pool is not None:
        doc_embs = embedder.encode_multi_process(
            docs, embed_pool,
            batch_size=batch_size, normalize_embeddings=True,
        )
        q_embs = embedder.encode_multi_process(
            queries, embed_pool,
            batch_size=batch_size, normalize_embeddings=True,
        )
        return q_embs @ doc_embs.T

    with torch.no_grad():
        doc_embs = embedder.encode(
            docs, convert_to_tensor=True, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )
        q_embs = embedder.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )
        sims = (q_embs @ doc_embs.T).float().cpu().numpy()
    del doc_embs, q_embs
    torch.cuda.empty_cache()
    return sims


def main():
    args = parse_args()

    dataset = args.dataset
    llm_suffix = args.llm_model_name.split("/")[-1].lower()
    emb_suffix = args.embedding_model_name.split("/")[-1].lower()

    save_basename = (
        f"baserag.{dataset}.k{args.top_k}.mc{args.max_doc_chars}."
        f"emb-{emb_suffix}.llm-{llm_suffix}"
    )
    if args.num_shards > 1:
        save_basename = f"{save_basename}.shard{args.shard_idx}of{args.num_shards}"
    save_subdir = os.path.join(args.output_dir, dataset, llm_suffix)
    save_path = os.path.join(save_subdir, save_basename + ".json")
    os.makedirs(save_subdir, exist_ok=True)

    tmp_leftover = save_path + ".tmp"
    if os.path.exists(tmp_leftover):
        os.remove(tmp_leftover)

    instances = load_dataset_unified(
        dataset, args.data_path, max_doc_chars=args.max_doc_chars
    )
    instances = apply_shard_and_limit(
        instances, args.shard_idx, args.num_shards, args.limit
    )
    resume_pred_dicts(save_path, instances)

    embedder, embed_pool = build_embedder(args)
    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)

    dataset_prompt = load_dataset_prompt(dataset)

    print(f"BaseRAG · dataset={dataset} · top_k={args.top_k} · "
          f"max_doc_chars={args.max_doc_chars} · {len(instances)} instances")
    print(f"LLM: {args.llm_model_name} @ {args.llm_base_url}")
    if dataset_prompt:
        print(f"Dataset prompt: {len(dataset_prompt)} chars from "
              f"src/memagent/prompts/{dataset}.md (appended to QA call)")
    print(f"Output: {save_path}")

    for elem in tqdm(instances):
        docs = elem["facts"]
        qas = elem["qas"]
        if not docs or not qas:
            continue

        qlist, qa_items, gold_answers, cat_ranges = build_qlist(qas)
        if not qlist:
            continue

        init_pred_dict(elem, cat_ranges)
        n_done = sum(len(elem["pred_dict"][cat]["solutions"]) for cat in cat_ranges)
        if n_done >= len(qlist):
            continue

        sims = encode_pair(embedder, embed_pool, docs, qlist, args.encode_batch_size)
        k = min(args.top_k, len(docs))
        top_idx = np.argsort(-sims, axis=1)[:, :k]

        def answer_one(qi):
            qa = qa_items[qi]
            retrieved_idx = top_idx[qi].tolist()
            retrieved = [docs[i] for i in retrieved_idx]
            retrieved_scores = [float(sims[qi, i]) for i in retrieved_idx]

            question_block = build_question_block(qa, dataset_prompt=dataset_prompt)
            user_prompt = build_user_prompt(retrieved, question_block)
            user_prompt, truncated, orig_chars = truncate_user_prompt(
                user_prompt, args.max_input_chars
            )

            extra_body = None
            if args.disable_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            t0 = time.time()
            resp = client.chat.completions.create(
                model=args.llm_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            pred = extract_boxed(content)
            usage = (resp.usage.model_dump()
                     if hasattr(resp.usage, "model_dump") else dict(resp.usage))
            em, f1 = em_f1(pred, gold_answers[qi])
            return {
                "question":          qlist[qi],
                "retrieved":         retrieved,
                "retrieved_scores":  retrieved_scores,
                "response_content":  content,
                "pred_ans":          pred,
                "elapsed":           elapsed,
                "usage":             usage,
                "em":                em,
                "f1":                f1,
                "input_truncated":   truncated,
                "input_orig_chars":  orig_chars,
                "input_sent_chars":  len(user_prompt),
            }

        chunk_size = max(1, args.llm_concurrency)
        for chunk_start in range(n_done, len(qlist), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(qlist))
            todo = list(range(chunk_start, chunk_end))
            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                results = list(pool.map(answer_one, todo))

            for qi, r in zip(todo, results):
                for cat, (start, end) in cat_ranges.items():
                    if start <= qi < end:
                        target = elem["pred_dict"][cat]
                        break
                target["solutions"].append({
                    "question":         r["question"],
                    "docs":             r["retrieved"],
                    "doc_scores":       r["retrieved_scores"],
                    "answer":           r["pred_ans"],
                    "gold_answers":     list(gold_answers[qi]),
                    "gold_docs":        None,
                    "input_truncated":  r["input_truncated"],
                    "input_orig_chars": r["input_orig_chars"],
                    "input_sent_chars": r["input_sent_chars"],
                })
                target["rationales"].append(r["response_content"])
                target["usage"].append({"latency_s": r["elapsed"], **r["usage"]})
                target["input_tokens"].append(r["usage"].get("prompt_tokens", 0))
                target["output_tokens"].append(r["usage"].get("completion_tokens", 0))
                if r["input_truncated"]:
                    print(f"  [trim] qi={qi}: {r['input_orig_chars']} -> "
                          f"{r['input_sent_chars']} chars")

            save_atomic(save_path, instances)

        for cat, bucket in elem["pred_dict"].items():
            ems, f1s = [], []
            for sol in bucket["solutions"]:
                em, f1 = em_f1(sol["answer"], sol["gold_answers"])
                ems.append(em)
                f1s.append(f1)
            bucket["metrics"] = {
                "ExactMatch": float(np.mean(ems)) if ems else 0.0,
                "F1Score":    float(np.mean(f1s)) if f1s else 0.0,
            }

        save_atomic(save_path, instances)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
