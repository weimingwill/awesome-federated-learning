[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/weimingwill/awesome-federeated-learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/weimingwill/awesome-federeated-learning.svg)](https://github.com/weimingwill/awesome-federeated-learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/weimingwill/awesome-federeated-learning.svg?color=blue)](https://github.com/weimingwill/awesome-federeated-learning/blob/master/LICENSE)

# Awesome Federated Learning
This repository maintains a collection of papers, articles, videos, frameworks, etc of federated learing, for the purpose of learning and research.

Note:

* **Italic** contents are the ones worth reading
* **Pull requests** are highly welcomed



- [Awesome Federated Learning](#awesome-federated-learning)
  - [Introduction](#introduction)
  - [Survey](#survey)
  - [Benchmarks](#benchmarks)
  - [Model Aggregation](#model-aggregation)
  - [Statistical Heterogeneity](#statistical-heterogeneity)
    - [Meta Learning](#meta-learning)
    - [Multi-task Learning](#multi-task-learning)
    - [Convergence](#convergence)
  - [System](#system)
  - [Communication Efficiency](#communication-efficiency)
    - [Compression](#compression)
    - [Important-Based Updating](#important-based-updating)
  - [Resource Allocation](#resource-allocation)
    - [Participants Selection](#participants-selection)
    - [Adaptive Aggregation](#adaptive-aggregation)
    - [Incentive Mechanism](#incentive-mechanism)
  - [Data Privacy and Confidentiality](#data-privacy-and-confidentiality)
    - [Courses](#courses)
    - [Differential Privacy](#differential-privacy)
      - [PATE](#pate)
    - [Secure Multi-party Computation](#secure-multi-party-computation)
      - [Secret Sharing](#secret-sharing)
      - [SPDZ](#spdz)
        - [Advance (Not Recommended For Beginners)](#advance-not-recommended-for-beginners)
      - [Build Safe AI Series](#build-safe-ai-series)
      - [MPC related Paper](#mpc-related-paper)
    - [Privacy Preserving Machine Learning](#privacy-preserving-machine-learning)
  - [Other Learning](#other-learning)
  - [Frameworks](#frameworks)
  - [Workshops](#workshops)
  - [Applications](#applications)
    - [Healthcare](#healthcare)
    - [Language Model](#language-model)
    - [Computer Vision](#computer-vision)
    - [Recommendation](#recommendation)
  - [Company](#company)


## Introduction

* Federated Learning Comic [[Google Blog]](https://federated.withgoogle.com/)
* Federated Learning: Collaborative Machine Learning without Centralized Training Data [[Google Blog]](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* GDPR, Data Shotrage and AI (AAAI-19) [[Video]](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang)
* Federated Learning: Machine Learning on Decentralized Data (Google I/O'19) [[Youtube]](https://www.youtube.com/watch?v=89BGjQYA0uE)
* Federated Learning White Paper V1.0 [[Paper]](https://www.fedai.org/static/flwp-en.pdf)



## Survey

* Federated Machine Learning: Concept and Applications [[Paper]](https://dl.acm.org/citation.cfm?id=3298981)
* Federated Learning: Challenges, Methods, and Future Directions [[Paper]](https://arxiv.org/abs/1908.07873)
* Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection [[Paper]](https://arxiv.org/abs/1907.09693)
* Federated Learning in Mobile Edge Networks: A Comprehensive Survey [[Paper]](https://arxiv.org/abs/1909.11875)
* Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges [[Paper]](https://arxiv.org/abs/1908.06847)
* Convergence of Edge Computing and Deep Learning: A Comprehensive Survey [[Paper]](https://arxiv.org/pdf/1907.08349.pdf)
* Advances and Open Problems in Federated Learning [[Paper]](https://arxiv.org/abs/1912.04977)



## Benchmarks

* LEAF: A Benchmark for Federated Settings [[Paper]](https://arxiv.org/abs/1812.01097) [[Github]](https://github.com/TalwalkarLab/leaf) [Recommend]
* A Performance Evaluation of Federated Learning Algorithms [[Paper]](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)
* Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking [[Paper]](https://arxiv.org/abs/1908.01924)



## Model Aggregation

* One-Shot Federated Learning [[Paper]](https://arxiv.org/abs/1902.11175)
* Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating [[Paper]](https://arxiv.org/abs/1910.08234) (NIPS2019 Workshop)
* Bayesian Nonparametric Federated Learning of Neural Networks [[Paper]](https://arxiv.org/abs/1905.12022) (ICML 2019)
* Agnostic Federated Learning [[Paper]](https://arxiv.org/abs/1902.00146) (ICML 2019)
* Federated Learning with Matched Averaging [[Paper]](https://openreview.net/forum?id=BkluqlSFDS) (ICLR 2020)
* Astraea: Self-balancing federated learning for improving classification accuracy of mobile deep learning applications [[Paper]](https://arxiv.org/abs/1907.01132)



## Statistical Heterogeneity

* Federated Learning with Non-IID Data [[Paper]](https://arxiv.org/abs/1806.00582) 
* The Non-IID Data Quagmire of Decentralized Machine Learning [[Paper]](https://arxiv.org/abs/1910.00189)
* Robust and Communication-Efficient Federated Learning from Non-IID Data [[Paper]](https://arxiv.org/pdf/1903.02891)
* FedMD: Heterogenous Federated Learning via Model Distillation [[Paper]](https://arxiv.org/abs/1910.03581) (NIPS2019 Workshop)
* First Analysis of Local GD on Heterogeneous Data [[Paper]](https://arxiv.org/abs/1909.04715)
* SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning [[Paper]](https://arxiv.org/abs/1910.06378)
* Federated Optimization for Heterogeneous Networks [[Paper]](https://arxiv.org/pdf/1812.06127)
* On the Convergence of FedAvg on Non-IID Data [[Paper]](https://arxiv.org/abs/1907.02189) [[OpenReview]](https://openreview.net/forum?id=HJxNAnVtDS)
* Agnostic Federated Learning [[Paper]](https://arxiv.org/abs/1902.00146) (ICML 2019)
* Local SGD Converges Fast and Communicates Little [[Paper]](https://arxiv.org/abs/1805.09767)
* Improving Federated Learning Personalization via Model Agnostic Meta Learning [[Paper]](https://arxiv.org/abs/1909.12488) (NIPS2019 Workshop)
* Adaptive Gradient-Based Meta-Learning Methods [[Paper]](https://arxiv.org/abs/1906.02717) (NIPS2019 Workshop)
* Federated Adversarial Domain Adaptation [[Paper]](https://arxiv.org/abs/1911.02054) (ICLR 2020)
* LoAdaBoost: Loss-Based AdaBoost Federated Machine Learning on Medical Data [[Paper]](https://arxiv.org/pdf/1811.12629)
* On Federated Learning of Deep Networks from Non-IID Data: Parameter Divergence and the Effects of Hyperparametric Methods [[Paper]](https://openreview.net/forum?id=SJeOAJStwB) [Rejected in ICML 2020]
* Overcoming Forgetting in Federated Learning on Non-IID Data [[Paper]](https://arxiv.org/abs/1910.07796) [NIPS2019 Workshop]
* FedMAX: Activation Entropy Maximization Targeting Effective Non-IID Federated Learning [[Video]](#workshop) [NIPS2019 Workshop]
* Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification [[Paper]](https://arxiv.org/abs/1909.06335) [NIPS2019 Workshop]
* Fair Resource Allocation in Federated Learning [[Paper]](https://arxiv.org/abs/1905.10497)
*  Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data [[Paper]](https://arxiv.org/abs/1811.11479)


### Meta Learning

* Federated Meta-Learning with Fast Convergence and Efficient Communication [[Paper]](https://arxiv.org/abs/1802.07876)
* Federated Meta-Learning for Recommendation [[Paper]](https://www.semanticscholar.org/paper/Federated-Meta-Learning-for-Recommendation-Chen-Dong/8e21d353ba283bee8fd18285558e5e8df39d46e8#paper-header)
* Adaptive Gradient-Based Meta-Learning Methods [[Paper]](https://arxiv.org/abs/1906.02717)

### Multi-task Learning

* MOCHA: Federated Multi-Task Learning [[Paper]](https://arxiv.org/abs/1705.10467) [[NIPS 2017]](https://papers.nips.cc/paper/7029-federated-multi-task-learning) [[Slides]](http://learningsys.org/nips17/assets/slides/mocha-NIPS.pdf)
* Variational Federated Multi-Task Learning [[Paper]](https://arxiv.org/abs/1906.06268)
* Federated Kernelized Multi-Task Learning [[Paper]](https://mlsys.org/Conferences/2019/doc/2018/30.pdf)


### Convergence

* A Linear Speedup Analysis of Distributed Deep Learning with Sparse and Quantized Communication [[Paper]](https://papers.nips.cc/paper/7519-a-linear-speedup-analysis-of-distributed-deep-learning-with-sparse-and-quantized-communication) [NIPS 2018]
* SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning [[Paper]](https://arxiv.org/abs/1910.06378)
* Federated Optimization for Heterogeneous Networks [[Paper]](https://arxiv.org/pdf/1812.06127)
* On the Convergence of FedAvg on Non-IID Data [[Paper]](https://arxiv.org/abs/1907.02189) [[OpenReview]](https://openreview.net/forum?id=HJxNAnVtDS)
* Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent [[Paper]](https://arxiv.org/abs/1705.09056) [NIPS 2017]
* Communication Efficient Decentralized Training with Multiple Local Updates [[Paper]](https://arxiv.org/abs/1910.09126)
* First Analysis of Local GD on Heterogeneous Data [[Paper]](https://arxiv.org/abs/1909.04715)
* MATCHA: Speeding Up Decentralized SGD via Matching Decomposition Sampling [[Paper]](https://arxiv.org/abs/1905.09435)
* Local SGD Converges Fast and Communicates Little [[Paper]](https://arxiv.org/abs/1805.09767)
* SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum [[Paper]](https://arxiv.org/abs/1910.00643)
* Adaptive Federated Learning in Resource Constrained Edge Computing Systems [[Paper]](https://arxiv.org/abs/1804.05271) [IEEE Journal on Selected Areas in Communications, 2019]
* Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning [[Paper]](https://arxiv.org/abs/1807.06629) [AAAI 2018]
* On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization [[Paper]](https://arxiv.org/abs/1905.03817) [ICML 2019]
*  Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data [[Paper]](https://arxiv.org/abs/1811.11479)



## System

* *Towards Federated Learning at Scale: System Design* [[Paper]](https://arxiv.org/abs/1902.01046) **[Must Read]**
* Demonstration of Federated Learning in a Resource-Constrained Networked Environment [[Paper]](https://ieeexplore.ieee.org/document/8784064)
* Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection [[Paper]](https://arxiv.org/abs/1907.09693)



## Communication Efficiency

* *Communication-Efficient Learning of Deep Networks from Decentralized Data* [[Paper]](https://arxiv.org/abs/1602.05629) [[Github]](https://github.com/roxanneluo/Federated-Learning) [Google] **[Must Read]**
* Federated Optimization: Distributed Machine Learning for On-Device Intelligence [[Paper]](https://arxiv.org/abs/1610.02527) [Google]
* cpSGD: Communication-efficient and differentially-private distributed SGD [[Paper]](https://arxiv.org/abs/1805.10559)
* FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization [[Paper]](https://arxiv.org/abs/1909.13014)
* Two-streamfederatedlearning:Reduce the communication costs,” in *2018 IEEE Visual Communications and Image Processing (VCIP)*. IEEE, 2018, pp. 1–4.
* “Edge-assisted hierarchical federated learning with non-iid data,” 
* Deep gradient compression: Reducing the communication bandwidth for distributed training [[Paper]](), Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally
* Robust and communication efficient federated learning from non-IID data [[Paper]](), Felix Sattler, Simon Wiedemann, Klaus-Robert Muller, and Wojciech Samek. 2019.
* PowerSGD: Practical low-rank gradient compression for distributed optimization [[Paper]]() [NeurIPS 2019], Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. 
* DeepSqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression [[Paper]]() Hanlin Tang, Xiangru Lian, Shuang Qiu, Lei Yuan, Ce Zhang, Tong Zhang, and Ji Liu, 2019.
* The error-feedback framework: Better rates for SGD with delayed gradients and compressed communication [[Paper]]() Sebastian U Stich and Sai Praneeth Karimireddy, 2019.
* vqSGD: Vector quantized stochastic gradient descent [[Paper]]() Venkata Gandikota, Raj Kumar Maity, and Arya Mazumdar, 2019.
* Robust and communication-efficient federated learning from non-iid data [[Paper]]() F. Sattler, S. Wiedemann, K.-R. Müller, and W. Samek, 2019.


### Compression

* Expanding the reach of federated learning by reducing client resource requirements [[Paper]]() Sebastian Caldas, Jakub Konecnˇ y, H Brendan McMahan, and Ameet Talwalkar, 2018.
* Communication compression for decentralized training [[Paper]]() [NIPS 2018], H. Tang, S. Gan, C. Zhang, T. Zhang, and J. Liu.
* Federated Learning: Strategies for Improving Communication Efficiency [[Paper]](https://arxiv.org/abs/1610.05492) [NIPS2016 Workshop] [Google]
* Distributed mean estimation with limited communication [[Paper]]() [ICML 2017], Ananda Theertha Suresh, Felix X. Yu, Sanjiv Kumar, and H Brendan McMahan. 
* Randomized distributed mean estimation: Accuracy vs communication. [[Paper]]() Frontiers in Applied Mathematics and Statistics, Jakub Konecny and Peter Richtarik, 2018
* Natural compression for distributed deep learning [[Paper]]() Samuel Horvath, Chen-Yu Ho, Ludovit Horvath, Atal Narayan Sahu, Marco Canini, and Peter Richtarik, 2019.
* QSGD: Communication-efficient SGD via gradient quantization and encoding [[Paper]]() [NIPS 2017], Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic.
* Error feedback fixes SignSGD and other gradient compression schemes [[Paper]]() [ICML 2019], Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi.
* Atomo: Communication-efficient learning via atomic sparsification [[Paper]]()  [NIPS 2018], H. Wang, S. Sievert, S. Liu, Z. Charles, D. Papailiopoulos, and S. Wright. 
* ZipML: Training linear models with end-to-end low precision, and a little bit of deep learning [[Paper]]() [ICML 2017], H. Zhang, J. Li, K. Kara, D. Alistarh, J. Liu, and C. Zhang.



### Important-Based Updating

*  Z. Tao and Q. Li, “esgd: Communication efficient distributed deep learning on the edge,” in {*USENIX*} *Workshop on Hot Topics in Edge Computing (HotEdge 18)*, 2018.
* Cmfl: Mitigating communication overhead for federated learning.



## Resource Allocation

### Participants Selection

* Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge [[Paper]](https://arxiv.org/abs/1804.08333) (FedCS)
* Hybrid-FL for Wireless Networks: Cooperative Learning Mechanism Using Non-IID Data [[Paper]](https://arxiv.org/abs/1905.07210)
  * Ask to upload some data from client to server
* Efficient Training Management for Mobile Crowd-Machine Learning: A Deep Reinforcement Learning Approach [[Paper]](https://arxiv.org/abs/1812.03633)
  * Reward function: accumulated data, energy consumption, training accuracy

* Fair Resource Allocation in Federated Learning [[Paper]](https://arxiv.org/abs/1905.10497)
* Low-latency Broadband Analog Aggregation For Federated Edge Learning [[Paper]](https://arxiv.org/abs/1812.11494)
* Federated Learning over Wireless Fading Channels [[Paper]](https://arxiv.org/pdf/1907.09769.pdf)
* Federated Learning via Over-the-Air Computation [[Paper]](https://arxiv.org/abs/1812.11750)

### Adaptive Aggregation

* Asynchronous Federated Learning for Geospatial Applications [[Paper]](https://link.springer.com.remotexs.ntu.edu.sg/chapter/10.1007/978-3-030-14880-5_2) [ECML PKDD Workshop 2018] 
* Asynchronous Federated Optimization [[Paper]](https://arxiv.org/abs/1903.03934)
* Adaptive Federated Learning in Resource Constrained Edge Computing Systems [[Paper]](https://arxiv.org/abs/1804.05271) [IEEE Journal on Selected Areas in Communications, 2019]

### Incentive Mechanism

* Incentive Mechanism for Reliable Federated Learning: A Joint Optimization Approach to Combining Reputation and Contract Theory [[Paper]](https://ieeexplore.ieee.org/document/8832210)
* Motivating Workers in Federated Learning: A Stackelberg Game Perspective [[Paper]](https://arxiv.org/abs/1908.03092)
* Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach [2019] [[Paper]](https://arxiv.org/abs/1905.07479)




## Data Privacy and Confidentiality

### Courses

* Applied Cryptography [[Udacity]](https://www.udacity.com/course/applied-cryptography--cs387)
  * Cryptography basics



### Differential Privacy

* A Brief Introduction to Differential Privacy [[Blog]](https://medium.com/georgian-impact-blog/a-brief-introduction-to-differential-privacy-eacf8722283b)
* *Deep Learning with Differential Privacy* [[Paper]](http://doi.acm.org/10.1145/2976749.2978318)
  * Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
* Learning Differentially Private Recurrent Language Models [[Paper]](https://arxiv.org/abs/1710.06963)
* Federated Learning with Bayesian Differential Privacy [[Paper]](https://arxiv.org/abs/1911.10071) (NIPS2019 Workshop)
* Private Federated Learning with Domain Adaptation [[Paper]](https://arxiv.org/abs/1912.06733) (NIPS2019 Workshop)

#### PATE

* *Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data* [[Paper]](http://dblp.uni-trier.de/db/journals/corr/corr1610. )
  * Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian J. Goodfellow, and Kunal Talwar.
  * Private Aggregation of Teacher Ensembles (PATE)
* *Scalable Private Learning with PATE* [[Paper]](https://arxiv.org/abs/1802.08908)
  * Extension of PATE
- The [original PATE paper](https://arxiv.org/abs/1610.05755) at ICLR 2017 and recording of the ICLR [oral](https://www.youtube.com/watch?v=bDayquwDgjU)
- The [ICLR 2018 paper](https://arxiv.org/abs/1802.08908) on scaling PATE to large number of classes and imbalanced data.
- GitHub [code repo for PATE](https://github.com/tensorflow/models/tree/master/research/differential_privacy/multiple_teachers)
- GitHub [code repo for the refined privacy analysis of PATE](https://github.com/tensorflow/models/tree/master/research/differential_privacy/pate)



### Secure Multi-party Computation

#### Secret Sharing
* Simple Introduction to Sharmir's Secret Sharing and Lagrange Interpolation [[Youtube]](https://www.youtube.com/watch?v=kkMps3X_tEE)
* Secret Sharing, Part 1 [[Blog]](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/): Shamir's Secret Sharing & Packed Variant
* Secret Sharing, Part 2 [[Blog]](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/): Improve efficiency
* Secret Sharing, Part 3 [[Blog]](https://mortendahl.github.io/2017/08/13/secret-sharing-part3/)



#### SPDZ

* Basics of Secure Multiparty Computation [[Youtube]](https://www.youtube.com/watch?v=_mDlLKgiFDY): based on Shamir's Secret Sharing

* What is SPDZ?
  * Part 1: MPC Circuit Evaluation Overview [[Blog]](https://bristolcrypto.blogspot.com/2016/10/what-is-spdz-part-1-mpc-circuit.html)
  * Part 2: Circuit Evaluation [[Blog]](https://bristolcrypto.blogspot.com/2016/10/what-is-spdz-part-2-circuit-evaluation.html)

* The SPDZ Protocol [[Blog]](https://mortendahl.github.io/2017/09/03/the-spdz-protocol-part1/): implementation codes included



##### Advance (Not Recommended For Beginners)

* Multiparty Computation from Somewhat Homomorphic Encryption [[Paper]](https://eprint.iacr.org/2011/535)
  * SPDZ introduction
* Practical Covertly Secure MPC for Dishonest Majority – or: Breaking the SPDZ Limits [[Paper]](https://eprint.iacr.org/2012/642)
* MASCOT: Faster Malicious Arithmetic Secure Computation with Oblivious Transfer [[Paper]](https://eprint.iacr.org/2016/505)
* Removing the crypto provider and instead letting the parties generate these triples on their own
* Overdrive: Making SPDZ Great Again [[Paper]](https://eprint.iacr.org/2017/1230)
* Safetynets: Verifiable execution of deep neural networks on an untrusted cloud

#### Build Safe AI Series
* Building Safe A.I. [[Blog]](http://iamtrask.github.io/2017/03/17/safe-ai/)
  * A Tutorial for Encrypted Deep Learning
  * Use Homomorphic Encryption (HE)

* Private Deep Learning with MPC [[Blog]](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
  * A Simple Tutorial from Scratch
  * Use Multiparty Compuation (MPC)

* Private Image Analysis with MPC [[Blog]](https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/)
  * Training CNNs on Sensitive Data
  * Use SPDZ as MPC protocol

#### MPC related Paper
Helen: Maliciously Secure Coopetitive Learning for Linear Models [[Paper]](https://arxiv.org/abs/1907.07212) (NIPS2019 Workshop)



### Privacy Preserving Machine Learning

* Privacy-Preserving Deep Learning [[Paper]](https://www.comp.nus.edu.sg/~reza/files/Shokri-CCS2015.pdf)
* Privacy Partition: A Privacy-Preserving Framework for Deep Neural Networks in Edge Networks [[Paper]](http://mews.sv.cmu.edu/papers/archedge-18.pdf)
* *Practical Secure Aggregation for Privacy-Preserving Machine Learning* [[Paper]](https://eprint.iacr.org/2017/281.pdf) (Google)
  * Secure Aggregation: The problem of computing a multiparty sum where no party reveals its update in the clear—even to the aggregator
  * Goal: securely computing sums of vectors, which has a constant number of rounds, low communication overhead, robustness to failures, and which requires only one server with limited trust
  * Need to have basic knowledge of cryptographic algorithms such as secret sharing, key agreement, etc.
* *Practical Secure Aggregation for Federated Learning on User-Held Data* [[Paper]](https://arxiv.org/abs/1611.04482) (Google)
  * Highly related to *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
  * Proposed 4 protocol one by one with gradual improvement to meet the requirement of secure aggregation propocol.
* SecureML: A System for Scalable Privacy-Preserving Machine Learning [[Paper]](https://eprint.iacr.org/2017/396.pdf)
* DeepSecure: Scalable Provably-Secure Deep Learning [[Paper]](https://arxiv.org/abs/1705.08963)
* Chameleon: A Hybrid Secure Computation Framework for Machine Learning Applications [[Paper]](https://arxiv.org/pdf/1801.03239.pdf)
* Contains several MPC frameworks



## Other Learning

- Federated Reinforcement Learning [[Paper]](https://arxiv.org/abs/1901.08277)
- Secure Federated Transfer Learning [[Paper]](https://arxiv.org/abs/1812.03337)
- Federated Multi-Task Learning [[Paper]](http://papers.nips.cc/paper/7029-federated-multi-task-learning.pdf)


## Frameworks

* PySyft [[Github]](https://github.com/OpenMined/PySyft)
  * A Generic Framework for Privacy Preserving Peep Pearning [[Paper]](https://arxiv.org/abs/1811.04017)
* Tensorflow Federated [[Web]](https://www.tensorflow.org/federated)
* FATE [[Github]](https://github.com/FederatedAI/FATE)
* Baidu PaddleFL [[Github]](https://github.com/PaddlePaddle/PaddleFL)
* Nvidia Clara SDK [[Web]](https://developer.nvidia.com/clara)


## Workshops

* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 1 [[Video]](https://slideslive.com/38921898/workshop-on-federated-learning-for-data-privacy-and-confidentiality-1)
* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 2 [[Video]](https://slideslive.com/38921899/workshop-on-federated-learning-for-data-privacy-and-confidentiality-2)
* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 3 [[Video]](https://slideslive.com/38921900/workshop-on-federated-learning-for-data-privacy-and-confidentiality-3)


## Applications

### Healthcare

* Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records [[Paper]](https://arxiv.org/ftp/arxiv/papers/1903/1903.09296.pdf) [[News]](https://venturebeat.com/2019/03/25/federated-learning-technique-predicts-hospital-stay-and-patient-mortality/)
  * MIT CSAI, Harvard Medical School, Tsinghua University
* Federated learning of predictive models from federated Electronic Health Records. [[Paper]](https://www.ncbi.nlm.nih.gov/pubmed/29500022)
  * Boston University, Massachusetts General Hospital
* FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare [[Paper]](https://arxiv.org/pdf/1907.09173.pdf)
  * Microsoft Research Asia
* Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation [[Paper]](https://arxiv.org/pdf/1810.04304.pdf)
  * Intel
* NVIDIA Clara Federated Learning to Deliver AI to Hospitals While Protecting Patient Data [[Blog]](https://blogs.nvidia.com/blog/2019/12/01/clara-federated-learning/)
  * Nvidia
* What is Federated Learning [[Blog]](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/)
  * Nvidia

### Language Model

Google

* Federated Learning for Mobile Keyboard Prediction [[Paper]](https://arxiv.org/abs/1811.03604)
 * Applied federated learning: Improving google keyboard query suggestions [[Paper]](https://arxiv.org/abs/1812.02903)
 * Federated Learning Of Out-Of-Vocabulary Words [[Paper]](https://arxiv.org/abs/1903.10635)

### Computer Vision

* Real-World Image Datasets for Federated Learning [[Paper]](https://arxiv.org/abs/1910.11089)
  * Webank & Extreme Vision

### Recommendation

 * Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System [[Paper]](https://arxiv.org/abs/1901.09888)
    * Huawei
 * Federated Meta-Learning with Fast Convergence and Efficient Communication [[Paper]](https://arxiv.org/abs/1802.07876)
    * Huawei

## Company

* Privacy.ai [[Website]](https://privacy.ai/)
* OpenMined [[Website]](https://www.openmined.org/)
* Arkhn [[Website]](https://arkhn.org/en/)
* Snips [[Website]](https://snips.ai/)


