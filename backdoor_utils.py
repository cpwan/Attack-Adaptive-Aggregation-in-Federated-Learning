from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

'''
Modified upon
https://github.com/howardmumu/Attack-Resistant-Federated-Learning/blob/70db1edde5b4b9dfb75633ca5dd5a5a7303c1f4c/FedAvg/Update.py#L335

@ Feb 20, 2020

'''

class Backdoor_Utils():
    
    def __init__(self):
        self.backdoor_label = 2
        self.trigger_position =[[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                [0, 0, 4], [0, 0, 5], [0, 0, 6],
                                [0, 2, 0], [0, 2, 1], [0, 2, 2],
                                [0, 2, 4], [0, 2, 5], [0, 2, 6],
                                ]
        self.trigger_value= [1,1,1,1,1,1,1,1,1,1,1,1,]

    def get_poison_batch(self, data, targets, backdoor_fraction, backdoor_label, evaluation=False):
#         poison_count = 0
        new_data = torch.empty(data.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(data)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_data[index] = self.add_backdoor_pixels(data[index])
#                 poison_count += 1

            else:  # will poison only a fraction of data when training
                if torch.rand(1) < backdoor_fraction:
                    new_targets[index] = backdoor_label
                    new_data[index] = self.add_backdoor_pixels(data[index])
#                     poison_count += 1
                else:
                    new_data[index] = data[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_data.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_data, new_targets

    def add_backdoor_pixels(self, item):
       
        for i in range(0, len(self.trigger_position)):
            pos = self.trigger_position[i]
            item[pos[0]][pos[1]][pos[2]] = self.trigger_value[i]
        return item
    