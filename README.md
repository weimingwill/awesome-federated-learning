[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/weimingwill/awesome-federeated-learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/weimingwill/awesome-federeated-learning.svg)](https://github.com/weimingwill/awesome-federeated-learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/weimingwill/awesome-federeated-learning.svg?color=blue)](https://github.com/weimingwill/awesome-federeated-learning/blob/master/LICENSE)

# Awesome Federated Learning
This repository maintains a collection of papers, articles, videos, frameworks, etc of federated learing, for the purpose of learning and research.



## Introduction

* Federated Learning Comic [[Google Blog]](https://federated.withgoogle.com/)
* Federated Learning: Collaborative Machine Learning without Centralized Training Data [[Google Blog]](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* GDPR, Data Shotrage and AI (AAAI-19) [[Paper]](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang)



## Survey

* Federated Machine Learning: Concept and Applications [[Paper]](https://dl.acm.org/citation.cfm?id=3298981) 
* Federated Learning: Challenges, Methods, and Future Directions [[Paper]](https://arxiv.org/abs/1908.07873)



## Frameworks

* PySyft [[Github]](https://github.com/OpenMined/PySyft)
* Tensorflow Federated [[Web]](https://www.tensorflow.org/federated)
* Baidu PaddleFL [[Github]](https://github.com/PaddlePaddle/PaddleFL)
* Nvidia Clara SDK [[Web]](https://developer.nvidia.com/clara)



## Data Privacy and Confidentiality

### Workshops

* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 1 [[Video]](https://slideslive.com/38921898/workshop-on-federated-learning-for-data-privacy-and-confidentiality-1)
* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 2 [[Video]](https://slideslive.com/38921899/workshop-on-federated-learning-for-data-privacy-and-confidentiality-2)
* NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 3 [[Video]](https://slideslive.com/38921900/workshop-on-federated-learning-for-data-privacy-and-confidentiality-3)

### Courses

* Applied Cryptography [[Udacity]](https://www.udacity.com/course/applied-cryptography--cs387)
  * Cryptography basics


### Differential Privacy

* Deep learning with differential privacy [[Paper]](http://doi.acm.org/10.1145/2976749.2978318)
  * Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
* Multiparty computation from somewhat homomorphic encryption
  * Ivan Damgård, Valerio Pastro, Nigel Smart, and Sarah Zakarias. (Advances in Cryptology)
* Practical covertly secure mpc for dishonest majority–or: breaking the spdz limits.
  * Ivan Damgård, Marcel Keller, Enrique Larraia, Valerio Pastro, Peter Scholl, and Nigel P Smart.(*European Symposium on Research in Computer Security*)
* Safetynets: Verifiable execution of deep neural networks on an untrusted cloud
  * Zahra Ghodsi, Tianyu Gu, and Siddharth Garg. . In *Advances in Neural Information Processing Systems*

#### PATE

* Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data [[Paper]](http://dblp.uni-trier.de/db/journals/corr/corr1610. )
  * Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian J. Goodfellow, and Kunal Talwar.
  * Private Aggregation of Teacher Ensembles (PATE)
* Scalable Private Learning with PATE [[Paper]](https://arxiv.org/abs/1802.08908)
  * Extension of PATE
- The [original PATE paper](https://arxiv.org/abs/1610.05755) at ICLR 2017 and recording of the ICLR [oral](https://www.youtube.com/watch?v=bDayquwDgjU)
- The [ICLR 2018 paper](https://arxiv.org/abs/1802.08908) on scaling PATE to large number of classes and imbalanced data.
- GitHub [code repo for PATE](https://github.com/tensorflow/models/tree/master/research/differential_privacy/multiple_teachers)
- GitHub [code repo for the refined privacy analysis of PATE](https://github.com/tensorflow/models/tree/master/research/differential_privacy/pate)



### Secret Sharing
* Simple Introduction to Sharmir's Secret Sharing and Lagrange Interpolation [[Youtube]](https://www.youtube.com/watch?v=kkMps3X_tEE)
* Secret Sharing, Part 1 [[Blog]](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/): Shamir's Secret Sharing & Packed Variant
* Secret Sharing, Part 2 [[Blog]](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/): Improve efficiency
* Secret Sharing, Part 3 [[Blog]](https://mortendahl.github.io/2017/08/13/secret-sharing-part3/)

### SPDZ
* Basics of Secure Multiparty Computation [[Youtube]](https://www.youtube.com/watch?v=_mDlLKgiFDY): based on Shamir's Secret Sharing

* What is SPDZ?
  * Part 1: MPC Circuit Evaluation Overview [[Blog]](https://bristolcrypto.blogspot.com/2016/10/what-is-spdz-part-1-mpc-circuit.html)
  * Part 2: Circuit Evaluation [[Blog]](https://bristolcrypto.blogspot.com/2016/10/what-is-spdz-part-2-circuit-evaluation.html)

* The SPDZ Protocol [[Blog]](https://mortendahl.github.io/2017/09/03/the-spdz-protocol-part1/): implementation codes included

#### Advance (Not Recommended For Beginners)

* Multiparty Computation from Somewhat Homomorphic Encryption [[Paper]](https://eprint.iacr.org/2011/535)
  * SPDZ introduction

* Practical Covertly Secure MPC for Dishonest Majority – or: Breaking the SPDZ Limits [[Paper]](https://eprint.iacr.org/2012/642)

* MASCOT: Faster Malicious Arithmetic Secure Computation with Oblivious Transfer [[Paper]](https://eprint.iacr.org/2016/505)
* Removing the crypto provider and instead letting the parties generate these triples on their own

* Overdrive: Making SPDZ Great Again [[Paper]](https://eprint.iacr.org/2017/1230)

### Build Safe AI Series
* Building Safe A.I. [[Blog]](http://iamtrask.github.io/2017/03/17/safe-ai/)
  * A Tutorial for Encrypted Deep Learning
  * Use Homomorphic Encryption (HE)

* Private Deep Learning with MPC [[Blog]](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
  * A Simple Tutorial from Scratch
  * Use Multiparty Compuation (MPC)

* Private Image Analysis with MPC [[Blog]](https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/)
  * Training CNNs on Sensitive Data
  * Use SPDZ as MPC protocol


### Privacy Preserving Machine Learning

* Privacy Partition: A Privacy-Preserving Framework for Deep Neural Networks in Edge Networks [[Paper]](http://mews.sv.cmu.edu/papers/archedge-18.pdf)
* Practical Secure Aggregation for Privacy-Preserving Machine Learning [[Paper]](https://eprint.iacr.org/2017/281.pdf) (Google)
  * Secure Aggregation: The problem of computing a multiparty sum where no party reveals its update in the clear—even to the aggregator
  * Goal: securely computing sums of vectors, which has a constant number of rounds, low communication overhead, robustness to failures, and which requires only one server with limited trust
  * Need to have basic knowledge of cryptographic algorithms such as secret sharing, key agreement, etc.

* Practical Secure Aggregation for Federated Learning on User-Held Data [[Paper]](https://arxiv.org/abs/1611.04482) (Google)
  * Highly related to *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
  * Proposed 4 protocol one by one with gradual improvement to meet the requirement of secure aggregation propocol.

* SecureML: A System for Scalable Privacy-Preserving Machine Learning [[Paper]](https://eprint.iacr.org/2017/396.pdf)

* DeepSecure: Scalable Provably-Secure Deep Learning [[Paper]](https://arxiv.org/abs/1705.08963)

* Chameleon: A Hybrid Secure Computation Framework for Machine Learning Applications [[Paper]](https://arxiv.org/pdf/1801.03239.pdf)
* Contains several MPC frameworks
