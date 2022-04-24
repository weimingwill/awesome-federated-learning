[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/weimingwill/awesome-federeated-learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/weimingwill/awesome-federeated-learning.svg)](https://github.com/weimingwill/awesome-federeated-learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/weimingwill/awesome-federeated-learning.svg?color=blue)](https://github.com/weimingwill/awesome-federeated-learning/blob/master/LICENSE)

# Awesome Federated Learning

A curated list of materials for federated learning, including blogs, surveys, research papers, and projects. You are very welcome to star it and create a pull request to update it.

Federated learning (FL) is attracting considerable attention these years. We organize these materials for you to learn federated learning and further facilitate your research and projects. 

We organize the papers by [research areas](#paper-by-research-area) for challenges in FL and by [conferences and journals](#paper-by-conference-and-journal). 

💡 We are thrilled to open-source our federated learning platform, [EasyFL](https://github.com/EasyFL-AI/EasyFL), to enable users with various levels of expertise to experiment and prototype FL applications with little/no coding. It is based on our years of research and we have used it to publish numerous papers in top-tier conferences and journals. You can also use it to get started with federated learning and implement your projects.

- [Awesome Federated Learning](#awesome-federated-learning)
  - [Paper (By conference and journal)](#paper-by-conference-and-journal)
  - [Paper (By research area)](#paper-by-research-area)
  - [General Resources](#general-resources)
    - [Blogs](#blogs)
    - [Survey](#survey)
    - [Benchmarks](#benchmarks)
    - [Video](#video)
    - [Frameworks](#frameworks)
    - [Company](#company)

## Paper (By conference and journal)

- [Federated learning paper by conferences](conferences.md): NeurIPS, ICML, ICLR, CVPR, ICCV, AAAI, IJCAI, ACMMM, etc.
- [Federated learning paper by journal](journal.md)

## Paper (By research area)

- [Statistical Heterogeneity](./areas/statistical-heterogeneity.md)
- [Communication Efficiency](./areas/communication-efficiency.md)
- [System](./areas/system.md): federated learning system design, frameworks, edge AI, etc.
- [Trustworthiness](./areas/trustworthiness.md): privacy, security, fairness
- [Decentralized FL](./areas/decentralized-fl.md)
- [Applications](./areas/applications.md)
- [Vertical FL](./areas/vertical-fl.md)
- [FL + {X}](./areas/fl+x-learning.md): FL + reinforcement learning, FL + transfer learning, etc. 

* **Communication-Efficient Learning of Deep Networks from Decentralized Data** [[Paper]](https://arxiv.org/abs/1602.05629) [[Github]](https://github.com/roxanneluo/Federated-Learning) [Google] **[Must Read]**

---

## General Resources

### Blogs

* Federated Learning Comic [[Google Blog]](https://federated.withgoogle.com/)
* Federated Learning: Collaborative Machine Learning without Centralized Training Data [[Google Blog]](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)


### Survey

* **Federated Machine Learning: Concept and Applications** [[Paper]](https://dl.acm.org/citation.cfm?id=3298981)
* **Federated Learning: Challenges, Methods, and Future Directions** [[Paper]](https://arxiv.org/abs/1908.07873)
* **Advances and Open Problems in Federated Learning** [[Paper]](https://arxiv.org/abs/1912.04977)
* Federated Learning White Paper V1.0 [[Paper]](https://www.fedai.org/static/flwp-en.pdf)
* Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection [[Paper]](https://arxiv.org/abs/1907.09693)
* Federated Learning in Mobile Edge Networks: A Comprehensive Survey [[Paper]](https://arxiv.org/abs/1909.11875)
* Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges [[Paper]](https://arxiv.org/abs/1908.06847)
* A Review of Applications in Federated Learning [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0360835220305532)


### Benchmarks

* LEAF: A Benchmark for Federated Settings [[Paper]](https://arxiv.org/abs/1812.01097) [[Github]](https://github.com/TalwalkarLab/leaf) [Recommend]
* The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems [[Paper]](https://arxiv.org/abs/2006.07856)
* Performance Optimization for Federated Person Re-identification via Benchmark Analysis [[Paper]](https://arxiv.org/abs/2008.11560) [ACMMM20] [[Github]](https://github.com/cap-ntu/FedReID)
* A Performance Evaluation of Federated Learning Algorithms [[Paper]](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)
* Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking [[Paper]](https://arxiv.org/abs/1908.01924)

### Video

* GDPR, Data Shotrage and AI (AAAI-19) [[Video]](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang)
* Federated Learning: Machine Learning on Decentralized Data (Google I/O'19) [[Youtube]](https://www.youtube.com/watch?v=89BGjQYA0uE)

### Frameworks

* EasyFL [[Github]](https://github.com/EasyFL-AI/EasyFL) [[Paper]](https://arxiv.org/abs/2105.07603)
* PySyft [[Github]](https://github.com/OpenMined/PySyft)
  * A Generic Framework for Privacy Preserving Peep Pearning [[Paper]](https://arxiv.org/abs/1811.04017)
* Tensorflow Federated [[Web]](https://www.tensorflow.org/federated)
* FATE [[Github]](https://github.com/FederatedAI/FATE)
* FedLearner [[Github]](https://github.com/bytedance/fedlearner) ByteDance
* Baidu PaddleFL [[Github]](https://github.com/PaddlePaddle/PaddleFL)
* Nvidia Clara SDK [[Web]](https://developer.nvidia.com/clara)
* [Flower.dev](https://flower.dev/)
* [OpenFL](https://github.com/intel/openfl)


### Company

* Adap [[Website]](https://adap.com/): Fleet Intelligence
* Privacy.ai [[Website]](https://privacy.ai/)
* OpenMined [[Website]](https://www.openmined.org/)
* Arkhn [[Website]](https://arkhn.org/en/): Healthcare data
* Owkin [[Website]](https://owkin.com/): Medical research
* Snips [[Website]](https://snips.ai/): Voice platform
* XAIN [[Website]](https://www.xain.io/) [[Github]](https://github.com/xainag/xain-fl): Automated Invoicing
* S20 [[Website]](https://www.s20.ai/): Multiple third party collaboration
* DataFleets [[Website]](https://www.datafleets.com/)
* Decentralized Machine Learning [[Website]](https://decentralizedml.com/)
