from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from math import ceil
from random import Random
from itertools import compress

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class iidDataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    
class labelDataPartitioner(object):
    """ Partitions a dataset by label. """

    def __init__(self, data, seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        #(tensor, label) tuple
        self.labels=set(x[1] for x in data)
        for i in self.labels:            
            label_iloc=list(compress(range(data_len),list(map(lambda x: x[1]==i,data))))
            self.partitions.append(label_iloc)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition%len(self.labels)])    
    
class iidLoader():
    def __init__(self,size,dataset,bsz=128):
        self.size=size
        self.bsz=bsz
        self.dataset = dataset
        partition_sizes = [1.0 / size for _ in range(size)]
        self.partition = iidDataPartitioner(dataset, partition_sizes)
    def __getitem__(self,rank):
        assert rank<self.size, 'partition index should be smaller than the size of the partition'
        """ Partitioning MNIST """
        dataset=self.dataset
        size = self.size
        bsz = self.bsz
        
        partition = self.partition.use(rank)
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=int(bsz), shuffle=True)
        return train_set
class labelLoader():
    def __init__(self,size,dataset,bsz=128):
        self.size=size
        self.bsz=bsz
        self.dataset = dataset
        self.partition = labelDataPartitioner(dataset)
        
    def __getitem__(self,rank): 
        assert rank<self.size, 'partition index should be smaller than the size of the partition'
        """ Partitioning MNIST 
        rank:= the index of the desired label
        """
        dataset=self.dataset
        size = self.size
        bsz = self.bsz
       
        partition = self.partition.use(rank%self.size)
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=int(bsz), shuffle=True)
        return train_set
