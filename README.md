# FedCache

This repository is the official Pytorch implementation DEMO of [**FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence.**](https://ieeexplore.ieee.org/document/10420495) ***IEEE Transactions on Mobile Computing (TMC)*. 2024**

## News

[Feb. 2024] Our FedCache is accepted by *IEEE Transactions on Mobile Computing (TMC)*.

[Jan. 2024] One follow-up paper examines the impact of logits poisoning attack on FedCache. Access [here](https://arxiv.org/abs/2401.03685).

[Dec. 2023] We discover a Chinese blog that interprets FedCache on CSDN. Access [here](https://blog.csdn.net/weixin_51306020/article/details/134983174).

[Dec. 2023] One follow-up paper confirms the further potential of FedCache for enhanced communication efficiency by accumulating local updates. Access [here](https://arxiv.org/abs/2312.04166).

[Aug. 2023] Our FedCache is released on arxiv. Access [here](https://arxiv.org/abs/2308.07816).

## Highlight

- FedCache is a **device** **friendly, scalable and effective** personalized federated learning architecture tailored for edge computing. 
- FedCache guarantees satisfactory performance while conforming to multiple personalized devices-side limitations.
- FedCache **improves communication efficiency by up to x200 over previous architectures** and can accommodate heterogeneous devices and asynchronous interactions among devices and the server.

## Family of FedCache

- **Foundation Works:** [FedICT](https://ieeexplore.ieee.org/abstract/document/10163770/), [FedDKC](https://dl.acm.org/doi/10.1145/3639369), [MTFL](https://ieeexplore.ieee.org/abstract/document/9492755), [DS-FL](https://ieeexplore.ieee.org/abstract/document/9392310), [FD](https://arxiv.org/abs/1811.11479)
- **Derivative Works:**
  - Communication: [ALU](https://arxiv.org/abs/2312.04166)
  - Poisoning Attack: [FDLA](https://arxiv.org/abs/2401.03685)
  - Generalization: Coming Soon......
  - Security: Coming Soon......
  - Application: Coming Soon......
  - Robustness: Coming Soon......
  - Scaling: TBD
  - Fairness: TBD
  - Deployment: TBD

**If you have any ideas or questions regarding to FedCache, please feel free to contact wuzhiyuan22s@ict.ac.cn.**

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

## Evaluation

### Model Homogeneous Setting

#### MNIST Dataset

| Method       | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :----------- | :------- | :--------------------- | :------------- |
| pFedMe       | 94.89    | 13.25                  | ×1.0           |
| MTFL         | 95.59    | 7.77                   | ×1.7           |
| FedDKC       | 89.62    | 9.13                   | ×1.5           |
| FedICT       | 84.62    | -                      | -              |
| FD           | 84.19    | -                      | -              |
| **FedCache** | 87.77    | **0.99**               | **×13.4**      |

#### FashionMNIST Dataset

| Method       | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :----------- | :------- | :--------------------- | :------------- |
| pFedMe       | 81.57    | 20.71                  | ×1.0           |
| MTFL         | 83.92    | 12.33                  | ×1.7           |
| FedDKC       | 78.24    | 8.43                   | ×2.5           |
| FedICT       | 76.90    | 13.34                  | ×1.6           |
| FD           | 76.32    | -                      | -              |
| **FedCache** | 77.71    | **0.08**               | **×258.9**     |

#### CIFAR-10 Dataset

| Method       | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :----------- | :------- | :--------------------- | :------------- |
| pFedMe       | 37.49    | -                      | -              |
| MTFL         | 43.43    | 52.99                  | ×1.0           |
| FedDKC       | 45.87    | 11.46                  | ×4.6           |
| FedICT       | 43.61    | 10.69                  | ×5.0           |
| FD           | 42.77    | -                      | -              |
| **FedCache** | 44.42    | **0.19**               | **×278.9**     |

#### CINIC-10 Dataset

| Method       | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :----------- | :------- | :--------------------- | :------------- |
| pFedMe       | 31.65    | -                      | -              |
| MTFL         | 34.09    | -                      | -              |
| FedDKC       | 43.95    | 4.12                   | ×1.3           |
| FedICT       | 42.79    | 5.50                   | ×1.0           |
| FD           | 39.36    | -                      | -              |
| **FedCache** | 40.45    | **0.07**               | **×78.6**      |

### Model Heterogeneous Setting

#### MNIST Dataset

| Method   | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :------- | :------- | :--------------------- | :------------- |
| FedDKC   | 85.38    | 10.53                  | ×1.0           |
| FedICT   | 80.53    | -                      | -              |
| FD       | 79.90    | -                      | -              |
| FedCache | 83.94    | **0.10**               | **×105.3**     |

#### FashionMNIST Dataset

| Method   | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :------- | :------- | :--------------------- | :------------- |
| FedDKC   | 77.96    | 12.64                  | ×1.0           |
| FedICT   | 76.11    | -                      | -              |
| FD       | 75.57    | -                      | -              |
| FedCache | 77.26    | **0.08**               | **×158.0**     |

#### CIFAR-10 Dataset

| Method   | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :------- | :------- | :--------------------- | :------------- |
| FedDKC   | 44.53    | 4.58                   | ×1.2           |
| FedICT   | 43.96    | 5.35                   | ×1.0           |
| FD       | 40.40    | -                      | -              |
| FedCache | 41.59    | **0.05**               | **×107.0**     |

#### CINIC-10 Dataset

| Method   | MAUA (%) | Communication Cost (G) | Speed-up Ratio |
| :------- | :------- | :--------------------- | :------------- |
| FedDKC   | 44.80    | 4.12                   | ×1.3           |
| FedICT   | 43.40    | 5.50                   | ×1.0           |
| FD       | 40.76    | -                      | -              |
| FedCache | 41.71    | **0.07**               | **×78.6**      |

## Cite this work

```bibtex
@article{wu2024fedcache,
  title={FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence},
  author={Wu, Zhiyuan and Sun, Sheng and Wang, Yuwei and Liu, Min and Xu, Ke and Wang, Wen and Jiang, Xuefeng and Gao, Bo and Lu, Jinda},
  journal={IEEE Transactions on Mobile Computing},
  year={2024},
  publisher={IEEE}
}
```

-------

## Related Works

[FedICT: Federated Multi-task Distillation for Multi-access Edge Computing.](https://ieeexplore.ieee.org/abstract/document/10163770/) *IEEE Transactions on Parallel and Distributed Systems (TPDS).* 2023

[Agglomerative Federated Learning: Empowering Larger Model Training via End-Edge-Cloud Collaboration.](https://arxiv.org/abs/2312.11489) *IEEE International Conference on Computer Communications (INFOCOM).* 2024

[Exploring the Distributed Knowledge Congruence in Proxy-data-free Federated Distillation.](https://dl.acm.org/doi/10.1145/3639369) *ACM Transactions on Intelligent Systems and Technology (TIST)*. 2023

[Federated Class-Incremental Learning with New-Class Augmented Self-Distillation.](https://arxiv.org/abs/2401.00622) *arXiv preprint arXiv:2401.00622.* 2024

[Survey of Knowledge Distillation in Federated Edge Learning.](https://arxiv.org/abs/2301.05849) *arXiv preprint arXiv:2301.05849.* 2023
