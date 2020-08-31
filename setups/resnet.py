from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.resnet import ResNet, BasicBlock

def ResNet18():
    model = ResNet(BasicBlock,[2,2,2,2],num_classes=10)
    return model


if __name__ == '__main__':

    net = ResNet18().cuda()
    y = net((torch.randn(100,3,32,32)).cuda())
    print(y.size())
