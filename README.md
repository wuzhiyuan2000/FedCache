# FedCache

This repository is the official Pytorch implementation DEMO of **FedCache**:

[FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence.](https://arxiv.org/abs/2308.07816) *IEEE Transactions on Mobile Computing (TMC)*. 2024 (Major Revision)

## FedCache Family

- **Foundation Works:** [FedICT](https://ieeexplore.ieee.org/abstract/document/10163770/), [FedDKC](https://dl.acm.org/doi/10.1145/3639369), [FD](https://arxiv.org/abs/1811.11479), [DS-FL](https://ieeexplore.ieee.org/abstract/document/9392310), [FedMD](https://arxiv.org/abs/1910.03581) 
- **Derivative Works:**
  - Communication: [ALU](https://arxiv.org/abs/2312.04166)
  - Poisoning Attack: [FDLA](https://arxiv.org/abs/2401.03685)
  - Generalization: Coming Soon...
  - Security: Coming Soon...
  - Applications: Coming Soon...
  - Scaling: TBD
  - Robustness: TBD
  - Fairness: TBD
  - Deployment: TBD

**If you have any ideas or questions about FedCache, please feel free to contact wuzhiyuan22s@ict.ac.cn.**

## Requirements

- Python:  3.10
- Pytorch:  1.13.1
- torchvision:  0.14.1
- hnswlib
- Other dependencies

-------
## Run this DEMO
```python main_fedcache.py```

-------

## Cite this work
```bibtex
@article{wu2023fedcache,
  title={FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence},
  author={Wu, Zhiyuan and Sun, Sheng and Wang, Yuwei and Liu, Min and Xu, Ke and Wang, Wen and Jiang, Xuefeng and Gao, Bo and Lu, Jinda},
  journal={arXiv preprint arXiv:2308.07816},
  year={2023}
}
```

-------

## Related Works

[FedICT: Federated Multi-task Distillation for Multi-access Edge Computing.](https://ieeexplore.ieee.org/abstract/document/10163770/) *IEEE Transactions on Parallel and Distributed Systems (TPDS).* 2023

[Agglomerative Federated Learning: Empowering Larger Model Training via End-Edge-Cloud Collaboration.](https://arxiv.org/abs/2312.11489) *IEEE International Conference on Computer Communications (INFOCOM).* 2024

[Exploring the Distributed Knowledge Congruence in Proxy-data-free Federated Distillation.](https://dl.acm.org/doi/10.1145/3639369) *ACM Transactions on Intelligent Systems and Technology (TIST)*. 2023

[Federated Class-Incremental Learning with New-Class Augmented Self-Distillation.](https://arxiv.org/abs/2401.00622) *arXiv preprint arXiv:2401.00622.* 2024

[Survey of Knowledge Distillation in Federated Edge Learning.](https://arxiv.org/abs/2301.05849) *arXiv preprint arXiv:2301.05849.* 2023

## Thanks

We thank Xuefeng Jiang for his guidance and suggestions on this repository.