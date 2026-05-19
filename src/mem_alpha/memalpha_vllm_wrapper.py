"""Long-lived vLLM offline-API wrapper for Memalpha-4B.

The vLLM OpenAI-compat server hits a tokenizer-caching bug under
transformers 5.x (Qwen2Tokenizer lacks `all_special_tokens_extended`),
but the offline `LLM.generate` path works fine. This wrapper holds the
model in a single long-lived process so the GPU stays claimed across
runs of `main.py`, and main.py POSTs prompt batches to `/generate`.
"""

import os
import logging

# Monkey-patch vLLM 0.8.5 ↔ transformers 5.x compat: vLLM's tokenizer cache
# reads `tokenizer.all_special_tokens_extended`, which transformers 5.x
# dropped from Qwen2Tokenizer. We re-add it as a property delegating to
# `all_special_tokens` BEFORE letting vLLM build its CachedTokenizer
# wrapper (which is needed downstream — it provides `max_token_id` etc.).
import vllm.transformers_utils.tokenizer as _vllm_tok
_orig_get_cached = _vllm_tok.get_cached_tokenizer

def _patched_get_cached(tokenizer):
    cls = type(tokenizer)
    if not hasattr(cls, 'all_special_tokens_extended'):
        cls.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
    return _orig_get_cached(tokenizer)

_vllm_tok.get_cached_tokenizer = _patched_get_cached

from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = os.environ.get('MEMALPHA_MODEL', 'YuWangX/Memalpha-4B')
MAX_MODEL_LEN = int(os.environ.get('MEMALPHA_MAX_MODEL_LEN', '16384'))
GPU_MEM = float(os.environ.get('MEMALPHA_GPU_MEM_UTIL', '0.40'))
TP = int(os.environ.get('MEMALPHA_TP', '1'))
PORT = int(os.environ.get('MEMALPHA_PORT', '8011'))

logger.info(f"Loading {MODEL} (tp={TP}, max_len={MAX_MODEL_LEN}, gpu_mem={GPU_MEM})")
llm = LLM(
    model=MODEL,
    dtype='bfloat16',
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=TP,
    gpu_memory_utilization=GPU_MEM,
)
logger.info("Model loaded — ready to serve")

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL})


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    prompts = data['prompts']
    sp_kwargs = data.get('sampling_params', {}) or {}
    sp = SamplingParams(
        temperature=sp_kwargs.get('temperature', 0.0),
        max_tokens=sp_kwargs.get('max_tokens', 2048),
        stop_token_ids=sp_kwargs.get('stop_token_ids'),
    )
    outputs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text for o in outputs]
    return jsonify({'responses': responses})


if __name__ == '__main__':
    # threaded=False: serialize HTTP requests so vLLM's internal batching
    # decides on the within-request prompt list, not Python threads. Each
    # main.py batch is a single POST with all per-batch prompts already.
    app.run(host='0.0.0.0', port=PORT, threaded=False)
