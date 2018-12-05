# CIFAR10 with PyTorch

[Pytorch](http://pytorch.org) implementation for [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) classification.

## Accuracy
| Models   |Acc  |
|----------|-----|
|VGG16     |.9387| 
|resnet18  |.9525| 
|resnet34  |.9555|
|resnet50  |.9521|  
|resnet101 |.9572|  


## Issue

Multi-gpu training is implemented, but not yet tested.

Memory issue when evaluating.

## Reference

Model files are the copies of [https://github.com/kuangliu/pytorch-cifar]
