from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedModel


# transformers ≥5 dropped `DynamicCache.from_legacy_cache`, but the cached
# NV-Embed-v2 modeling code calls it during `forward` to wrap a tuple-format
# `past_key_values` into a `DynamicCache`. The new `DynamicCache.__init__`
# accepts the legacy tuple/iterable directly via its first arg, so restoring
# the classmethod as a thin wrapper is enough.
if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        return cls(past_key_values)
    DynamicCache.from_legacy_cache = _from_legacy_cache

# Same vintage: `get_usable_length(seq_length)` was dropped in favor of
# `get_seq_length()`. For DynamicCache the two were always equivalent in
# practice (no max-length truncation), so a thin shim is safe.
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length=None, layer_idx=0):
        return self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
    DynamicCache.get_usable_length = _get_usable_length

# `to_legacy_cache()` was the inverse of `from_legacy_cache` — flatten the
# DynamicCache layers back to a tuple-of-tuples. NV-Embed-v2's forward calls
# this on the return path when the caller passed a tuple-format
# `past_key_values`. Newer transformers no longer ship it; reconstruct from
# the per-layer key/value tensors stored on the cache layers.
if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        legacy = []
        for layer in getattr(self, "layers", []) or []:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None and hasattr(layer, "key_cache"):
                k, v = layer.key_cache, layer.value_cache
            legacy.append((k, v))
        return tuple(legacy)
    DynamicCache.to_legacy_cache = _to_legacy_cache

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


# transformers ≥5.7 hard-codes references to `model.all_tied_weights_keys`
# inside `_finalize_model_loading`/`_move_missing_keys_from_meta_to_device`,
# expecting every PreTrainedModel subclass to set this in `post_init()`.
# NV-Embed-v2's custom `modeling_nvembed.py` (HF cache hash 3fa5965) doesn't
# call `self.post_init()` in either `NVEmbedModel.__init__` or
# `LatentAttentionModel.__init__`, so the attribute never gets set and loading
# crashes with AttributeError. We can't edit the HF cache locally, but we can
# install a class-level default on PreTrainedModel: instances that DO go
# through post_init() overwrite it with their real instance attribute (no
# functional change), while instances that skip post_init() inherit the empty
# dict and the missing-key code path becomes a no-op.
if not getattr(PreTrainedModel, "_HIPPORAG_TIED_WEIGHTS_FALLBACK", False):
    PreTrainedModel.all_tied_weights_keys = {}
    PreTrainedModel._HIPPORAG_TIED_WEIGHTS_FALLBACK = True


class NVEmbedV2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        # Newer transformers (>=4.55ish) call `model.all_tied_weights_keys`
        # during the auto-device-map inference path, which NV-Embed-v2's custom
        # model class doesn't define (only `_tied_weights_keys`). When CUDA is
        # available we bypass that path entirely by loading on the single
        # visible CUDA device and `.to('cuda')`-ing afterwards. Falls back to
        # the original device_map="auto" route if no GPU is visible.
        init_kwargs = dict(self.embedding_config.model_init_params)
        # NV-Embed-v2's `NVEmbedModel.__init__` builds its submodels via
        # `AutoModel.from_config(config.text_config)` without forwarding the
        # caller's dtype, so even with `dtype=bfloat16` to `from_pretrained` the
        # submodels (Mistral-7B base ~7B params) get instantiated in fp32 and
        # the checkpoint is loaded into the fp32 buffers — ~31 GiB on CPU/GPU,
        # which OOMs on a 47-GiB GPU shared with vLLM (~26 GiB). Resolve the
        # requested dtype, load, then cast on CPU before moving to GPU so the
        # GPU only ever sees the bf16-sized model (~15 GiB).
        target_dtype_kwarg = init_kwargs.get("dtype") or init_kwargs.get("torch_dtype")
        target_dtype = (
            getattr(torch, target_dtype_kwarg)
            if isinstance(target_dtype_kwarg, str)
            else target_dtype_kwarg
        )
        if torch.cuda.is_available():
            init_kwargs.pop("device_map", None)
            self.embedding_model = AutoModel.from_pretrained(**init_kwargs)
            if target_dtype is not None:
                self.embedding_model = self.embedding_model.to(target_dtype)
            self.embedding_model = self.embedding_model.to("cuda")
        else:
            self.embedding_model = AutoModel.from_pretrained(**init_kwargs)
            if target_dtype is not None:
                self.embedding_model = self.embedding_model.to(target_dtype)
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                # transformers ≥5 renamed `torch_dtype` → `dtype` and silently
                # ignores the old kwarg, so the model would load in fp32 (~28 GiB
                # for NV-Embed-v2's Mistral-7B base) instead of the requested
                # bf16 (~14 GiB), OOMing on a GPU shared with vLLM. Pass both
                # so it works on either transformers version.
                "torch_dtype": self.global_config.embedding_model_dtype,
                "dtype": self.global_config.embedding_model_dtype,
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    # def _add_eos(self, texts: List[str]) -> List[str]:
    #     # Adds EOS token to each text
    #     return [text + self.embedding_model.tokenizer.eos_token for text in texts]

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["prompts"] = texts  # self._add_eos(texts=texts)
            print(params) 
            """ 
            {'max_length': 2048, 'instruction': '', 'num_workers': 32, 'prompts': ['Oliver Badman is a polit
ician.', 'George Rankin is a politician.', 'Thomas Marwick is a politician.']}
            """ 
            results = self.embedding_model.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["prompts"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
