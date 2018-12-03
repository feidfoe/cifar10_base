# CIFAR10 with PyTorch

[Pytorch](http://pytorch.org) implementation for [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) classification.

## Accuracy
| Models   |Acc  |
|----------|-----|
|VGG16     |.9387| 
|resnet18  |.9525| 
|resnet50  |.9540|  
|resnet101 |.9559|  


## Issue

Multi-gpu training is implemented, but not yet tested.

Memory issue when evaluating.

## Reference

Model files are the copy of [https://github.com/kuangliu/pytorch-cifar]
