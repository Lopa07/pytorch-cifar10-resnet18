# pytorch-model-training

Train `model` with `dataset` in PyTorch:
- `model`:
    - VGG16
    - ResNet18
    - DLA
- `dataset`:
    - CIFAR10
    - CIFAR100
    - SVHN

## Getting Started
Clone this repo, then install all dependencies:
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.9.

## Accuracy
| Model             | CIFAR10     | CIFAR100    | SVHN        |
| ----------------- | ----------- | ----------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)      | 93.72%    | 74.87%    | 96.17%    |
| [ResNet18](https://arxiv.org/abs/1512.03385)  | 95.69%    | 74.87%    | 96.64%    |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)   | 95.87%    | 78.62%    | 96.83%    |
