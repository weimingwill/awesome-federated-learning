[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/weimingwill/awesome-federeated-learning/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/weimingwill/awesome-federeated-learning.svg)](https://github.com/weimingwill/awesome-federeated-learning/commits/master)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/weimingwill/awesome-federeated-learning.svg?color=blue)](https://github.com/weimingwill/awesome-federeated-learning/blob/master/LICENSE)

# Awesome Federated Learning
This repository maintains a collection of papers, articles, videos, frameworks, etc of federated learing, for the purpose of learning and research.

## Privacy

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
