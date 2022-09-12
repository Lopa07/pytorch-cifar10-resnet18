# pytorch-model-training

Train `model` with `dataset` in PyTorch:
- `model`:
    - VGG16
    - ResNet18
    - DLA
    - WRN-28-10
- `dataset`:
    - CIFAR10
    - CIFAR100
    - SVHN
    - MNIST
    - FashionMNIST

## Getting Started
Clone this repo, then install all dependencies:
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.9.

## Accuracy
| Model             | CIFAR10   | CIFAR100  | SVHN  | MNIST | FashionMNIST  |
| ----------------- | --------- | --------- | ----- | ----- | ------------- |
| [VGG16](https://arxiv.org/abs/1409.1556)          | 94.32%    | 74.87%    | 96.17%    | 99.64%    | 93.88%    |
| [ResNet18](https://arxiv.org/abs/1512.03385)      | 95.69%    | 78.24%    | 96.64%    | 99.70%    | 94.10%    |
| [DLA](https://arxiv.org/abs/1707.06484.pdf)       | 95.87%    | 78.62%    | 96.83%    | 99.63%    | 94.06%    |
| [WRN-28-10](https://arxiv.org/abs/1605.07146.pdf) | 95.94%    | 81.33%    | 97.02%    | 99.71%    | 94.36%    |

Training configurations, checkpoint models, and output logs can be found [here](https://drive.google.com/drive/folders/1ElZTzo-PAht4uwOsYotIyrd4kVisY1jV?usp=sharing).
