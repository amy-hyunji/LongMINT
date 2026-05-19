import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

from _common import (
    HF_SPLIT_MAP,
    SYSTEM_PROMPT,
    apply_shard_and_limit,
    build_qlist,
    build_question_block,
    build_user_prompt,
    cuda_cleanup,
    em_f1,
    extract_boxed,
    load_dataset_prompt,
    load_dataset_unified,
    resume_pred_dicts,
    save_atomic,
    truncate_user_prompt,
    _safe_id,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="HippoRAG over a unified-format LongMINT dataset.")

    # I/O
    p.add_argument("--dataset", required=True, choices=sorted(HF_SPLIT_MAP),
                   help="Dataset name (babi | wiki | github | horizonbench). "
                        "Selects the corresponding split when loading from "
                        "Hugging Face, and tags the output filename.")
    p.add_argument("--data_path", default=None,
                   help="Optional path to a local unified-format JSON file. "
                        "When omitted, the dataset is loaded from the "
                        "dinobby/LongMINT Hugging Face dataset.")
    p.add_argument("--output_dir", default="outputs",
                   help="Base output directory. Final path is "
                        "<output_dir>/<dataset>/<llm_suffix>/<basename>.json.")

    # LLM
    p.add_argument("--llm_model_name", required=True,
                   help="Served by --llm_base_url (e.g. Qwen/Qwen3-32B).")
    p.add_argument("--llm_base_url", required=True,
                   help="OpenAI-compatible base URL of the served LLM "
                        "(e.g. http://localhost:8000/v1).")
    p.add_argument("--llm_api_key", default="EMPTY")
    p.add_argument("--qa_max_tokens", type=int, default=4096,
                   help="Completion-token budget for the QA call. "
                        "Reasoning models (Qwen3.x) need headroom for "
                        "thought + \\boxed{...}.")
    p.add_argument("--qa_temperature", type=float, default=0.0)
    p.add_argument("--qa_concurrency", type=int, default=32,
                   help="Thread fanout for QA calls per instance.")
    p.add_argument("--qa_max_input_chars", type=int, default=85000,
                   help="Cap on QA user-prompt char length. When exceeded, "
                        "the docs tail is trimmed (worst-scoring first); "
                        "the question is preserved. 0 = disabled.")
    p.add_argument("--llm_disable_thinking", type=int, default=1,
                   help="When 1, ask Qwen3's chat template to skip "
                        "<think>...</think>. Applies both to HippoRAG-"
                        "internal calls (NER, triple extraction) and the "
                        "QA call.")
    p.add_argument("--max_new_tokens", type=int, default=8192,
                   help="Used for HippoRAG-internal LLM calls. "
                        "Separate from --qa_max_tokens.")

    # Embedding model (HippoRAG-internal)
    p.add_argument("--embedding_model_name", required=True,
                   help="Embedder used by HippoRAG (e.g. nvidia/NV-Embed-v2).")
    p.add_argument("--embedding_batch_size", type=int, default=1,
                   help="Per-GPU embedder batch. Long contexts (tens of KB) "
                        "warrant batch=1 next to a co-located vLLM.")
    p.add_argument("--embedding_max_seq_len", type=int, default=4096,
                   help="Max tokens per embedded text.")
    p.add_argument("--embedding_model_dtype", default="bfloat16",
                   choices=["float16", "float32", "bfloat16", "auto"])

    # Retrieval / corpus
    p.add_argument("--qa_top_k", type=int, default=5,
                   help="How many retrieved docs to feed into the QA prompt.")
    p.add_argument("--max_doc_chars", type=int, default=0,
                   help="Char cap for each rendered context doc. 0 = "
                        "disabled. Set positive if long docs OOM the "
                        "embedder or blow HippoRAG's KG-extraction context.")

    # Run control
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N instances (smoke test).")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Shard index (0..num_shards-1).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total shards. Instances assigned by index %% num_shards.")

    args = p.parse_args()
    if not (0 <= args.shard_idx < args.num_shards):
        p.error(f"shard_idx={args.shard_idx} must be in [0, {args.num_shards})")
    return args


def main():
    args = parse_args()

    dataset = args.dataset
    llm_suffix = args.llm_model_name.split("/")[-1].lower()
    _emb = args.embedding_model_name.split("Transformers/", 1)[-1]
    emb_suffix = _emb.split("/")[-1].lower()

    save_basename = (
        f"hipporag.{dataset}.k{args.qa_top_k}.mc{args.max_doc_chars}."
        f"emb-{emb_suffix}.llm-{llm_suffix}"
    )
    if args.num_shards > 1:
        save_basename = f"{save_basename}.shard{args.shard_idx}of{args.num_shards}"
    save_subdir = os.path.join(args.output_dir, dataset, llm_suffix)
    save_path = os.path.join(save_subdir, save_basename + ".json")
    save_root = os.path.join(save_subdir, save_basename)
    os.makedirs(save_root, exist_ok=True)

    instances = load_dataset_unified(
        dataset, args.data_path, max_doc_chars=args.max_doc_chars
    )
    instances = apply_shard_and_limit(
        instances, args.shard_idx, args.num_shards, args.limit
    )
    resume_pred_dicts(save_path, instances)

    dataset_prompt = load_dataset_prompt(dataset)

    print(f"HippoRAG · dataset={dataset} · top_k={args.qa_top_k} · "
          f"max_doc_chars={args.max_doc_chars} · {len(instances)} instances")
    print(f"LLM: {args.llm_model_name} @ {args.llm_base_url}")
    print(f"Embedder: {args.embedding_model_name} (dtype={args.embedding_model_dtype})")
    print(f"Output: {save_path}")
    if dataset_prompt:
        print(f"Dataset prompt: {len(dataset_prompt)} chars from "
              f"src/memagent/prompts/{dataset}.md")

    qa_client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)

    for elem in tqdm(instances):
        if "pred_dict" in elem:
            continue

        facts = elem["facts"]
        qas = elem["qas"]
        if not facts or not qas:
            continue

        qlist, qa_items, gold_answers, cat_ranges = build_qlist(qas)
        if not qlist:
            continue

        instance_dir = os.path.join(save_root, _safe_id(elem["instance_id"]))
        os.makedirs(instance_dir, exist_ok=True)

        global_config = BaseConfig(
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_seq_len=args.embedding_max_seq_len,
            embedding_model_dtype=args.embedding_model_dtype,
            qa_max_input_chars=args.qa_max_input_chars,
            max_new_tokens=args.max_new_tokens,
            llm_disable_thinking=bool(args.llm_disable_thinking),
        )
        hipporag = HippoRAG(
            global_config=global_config,
            save_dir=instance_dir,
            llm_model_name=args.llm_model_name,
            embedding_model_name=args.embedding_model_name,
            llm_base_url=args.llm_base_url,
        )

        hipporag.index(docs=facts)

        # HippoRAG retrieval (raw question only — never the candidate list).
        retrieve_solutions = hipporag.retrieve(
            queries=qlist, num_to_retrieve=args.qa_top_k
        )

        def answer_one(qi):
            qa = qa_items[qi]
            qs = retrieve_solutions[qi]
            retrieved = list(qs.docs[: args.qa_top_k])
            scores = qs.doc_scores
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            retrieved_scores = [float(s) for s in scores[: args.qa_top_k]]

            question_block = build_question_block(qa, dataset_prompt=dataset_prompt)
            user_prompt = build_user_prompt(retrieved, question_block)
            user_prompt, truncated, orig_chars = truncate_user_prompt(
                user_prompt, args.qa_max_input_chars
            )

            extra_body = None
            if args.llm_disable_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            t0 = time.time()
            resp = qa_client.chat.completions.create(
                model=args.llm_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.qa_max_tokens,
                temperature=args.qa_temperature,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            pred = extract_boxed(content)
            usage = (resp.usage.model_dump()
                     if hasattr(resp.usage, "model_dump") else dict(resp.usage))
            em, f1 = em_f1(pred, gold_answers[qi])
            return {
                "qi": qi,
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

        with ThreadPoolExecutor(
            max_workers=min(args.qa_concurrency, max(1, len(qlist)))
        ) as pool:
            results = list(pool.map(answer_one, range(len(qlist))))

        elem["pred_dict"] = {}
        for cat, (start, end) in cat_ranges.items():
            sols, rats, usg, in_toks, out_toks = [], [], [], [], []
            ems, f1s = [], []
            for qi in range(start, end):
                r = results[qi]
                sols.append({
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
                rats.append(r["response_content"])
                usg.append({"latency_s": r["elapsed"], **r["usage"]})
                in_toks.append(r["usage"].get("prompt_tokens", 0))
                out_toks.append(r["usage"].get("completion_tokens", 0))
                ems.append(r["em"])
                f1s.append(r["f1"])
            elem["pred_dict"][cat] = {
                "solutions":     sols,
                "rationales":    rats,
                "usage":         usg,
                "metrics": {
                    "ExactMatch": float(np.mean(ems)) if ems else 0.0,
                    "F1Score":    float(np.mean(f1s)) if f1s else 0.0,
                },
                "input_tokens":  in_toks,
                "output_tokens": out_toks,
            }

        save_atomic(save_path, instances)

        try:
            del hipporag
        except NameError:
            pass
        cuda_cleanup()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
