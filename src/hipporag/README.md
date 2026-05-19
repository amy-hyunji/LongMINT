# HippoRAG (vendored fork)

This directory is a vendored fork of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
by the OSU NLP Group, included here so that LongMINT can run the HippoRAG and
BaseRAG baselines out of the box.

- Upstream repository: https://github.com/OSU-NLP-Group/HippoRAG
- Upstream paper: [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](https://arxiv.org/abs/2502.14802)
- Upstream license: MIT (see `LICENSE` in this directory)

Code in this directory has been modified from upstream to integrate with
LongMINT's evaluation. Modifications by the
LongMINT authors are also released under the MIT License.

If you use this code, please cite the original HippoRAG paper in addition to
LongMINT.

```bibtex
@misc{gutiérrez2025ragmemorynonparametriccontinual,
      title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
      year={2025},
      eprint={2502.14802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14802}, 
}
```