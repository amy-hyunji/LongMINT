from typing import List
import json

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
from sentence_transformers import SentenceTransformer

class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("Transformers/"):]
        self.embedding_type = 'float'
        # Honor global_config.embedding_batch_size — for long-doc corpora
        # (HorizonBench conversations ≈ 10–50KB) the hardcoded default of 64
        # blows the GPU. Fall back to 64 only when caller didn't set one.
        self.batch_size = max(1, getattr(global_config, "embedding_batch_size", 64) or 64)

        # SentenceTransformer defaults to fp32, which on a 4B model
        # (Qwen3-Embedding-4B) costs ~16 GB on GPU and OOMs when the GPU is
        # shared with vLLM TP workers. Honor `embedding_model_dtype`, default
        # to bf16 to halve memory.
        dtype_name = getattr(global_config, "embedding_model_dtype", "bfloat16") or "bfloat16"
        torch_dtype = (
            getattr(torch, dtype_name, torch.bfloat16)
            if isinstance(dtype_name, str) and dtype_name != "auto"
            else torch.bfloat16
        )
        self.model = SentenceTransformer(
            self.model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"dtype": torch_dtype, "torch_dtype": torch_dtype},
        )
        # Cap input tokens — SentenceTransformer truncates internally past
        # `model.max_seq_length`. Keeping this small saves activation memory.
        max_seq = getattr(global_config, "embedding_max_seq_len", 0)
        if max_seq:
            self.model.max_seq_length = max_seq

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str]) -> None:
        try:
            response = self.model.encode(texts, batch_size=self.batch_size)
        except Exception as err:
            raise Exception(f"An error occurred: {err}")
        return np.array(response)

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if len(texts) < self.batch_size:
            return self.encode(texts)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size]))
        return np.concatenate(results)
