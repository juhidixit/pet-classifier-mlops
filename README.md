# Pet Breed Classifier: an end-to-end MLOps pipeline

[![CI](https://github.com/juhidixit/pet-classifier-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/juhidixit/pet-classifier-mlops/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

Fine-grained image classification of **37 cat and dog breeds** (Oxford-IIIT Pet), built the way a
production ML team would: reproducible configs, experiment tracking, hyper-parameter sweeps,
tested code, explainability, a containerised API, CI, and infrastructure-as-code.

The core question: **how much does transfer learning buy you on a small, fine-grained dataset?**
An ImageNet-pretrained ResNet-50 is compared against a CNN trained from scratch under the same
data pipeline, and both are dissected with Grad-CAM, calibration and error analysis.

> Training runs are in progress; the results section is filled in from the generated
> `reports/` once they finish.
