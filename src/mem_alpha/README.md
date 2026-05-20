# Mem-alpha (vendored fork)

This directory contains a vendored fork of [Mem-alpha](https://github.com/wangyu-ustc/Mem-alpha)
by Wang Yu et al., included here so that MINTEval can run the Mem-alpha
baseline out of the box.

- Upstream repository: https://github.com/wangyu-ustc/Mem-alpha
- Upstream paper: [Mem-alpha: Learning Memory Construction via Reinforcement Learning](https://arxiv.org/abs/2509.25911)

## Copyright and license

At the time of this fork, the upstream repository did not include an explicit
license file. All copyright in the vendored code therefore remains with the
original Mem-alpha authors. The MINTEval project includes this code in good
faith for research-reproducibility purposes. Users who wish to redistribute,
modify, or use this code beyond running the MINTEval benchmark should refer to
the upstream repository and contact the original authors for permission.

Modifications introduced by the MINTEval authors (integration glue, evaluation
harness, dataset loaders) are released under the same license as the rest of
MINTEval.

## Citation

If you use this code, please cite the original Mem-alpha paper in addition to
MINTEval:

```bibtex
@misc{wang2025memalphalearningmemoryconstruction,
      title={Mem-{\alpha}: Learning Memory Construction via Reinforcement Learning}, 
      author={Yu Wang and Ryuichi Takanobu and Zhiqi Liang and Yuzhen Mao and Yuanzhe Hu and Julian McAuley and Xiaojian Wu},
      year={2025},
      eprint={2509.25911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.25911}, 
}
```
