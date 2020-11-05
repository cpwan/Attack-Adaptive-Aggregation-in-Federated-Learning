from __future__ import print_function

import torch

from tasks import cifar

'''
Generate batches of backdoored cifar10 images

the list of images for semantic backdoor is retrieved from
https://github.com/ebagdasa/backdoor_federated_learning/blob/master/utils/params_runner.yaml


Reference:
Bagdasaryan, Eugene, et al. "How to backdoor federated learning." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
'''


class SemanticBackdoor_Utils():
    '''
    Given a batch of benign images,
    replace a fraction of them to be the semantic backdoor, relabel them to the attack target
    
    By default, the attack target is 'bird' (2).
    The semantic backdoor are car images with strips pattern/ colored in green
    
    '''

    def __init__(self):
        self.backdoor_label = 2
        self.semanticBackdoorCandidate = \
            [2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, \
             11395, 11744, 14209, 14238, 18716, 19793, 20781, 21529, \
             31311, 40518, 40633, 42119, 42663, 49392, 389, 561, \
             874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, \
             21422, 22984, 32941, 34287, 34385, 36005, 37365, 37533, \
             38658, 38735, 39824, 40138, 41336, 41861, 47001, 47026, \
             48003, 48030, 49163, 49588, 330, 568, 3934, 12336, \
             30560, 30696, 33105, 33615, 33907, 36848, 40713, 41706]  ## 64 of them
        self.semanticBackdoorDataSet = torch.utils.data.Subset(cifar.getDataset(), self.semanticBackdoorCandidate)
        self.semanticBackdoorDataLoader = torch.utils.data.DataLoader(self.semanticBackdoorDataSet, batch_size=1,
                                                                      shuffle=True)

    def get_poison_batch(self, data, targets, backdoor_fraction, backdoor_label, evaluation=False):
        #         poison_count = 0
        new_data = torch.empty(data.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(data)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_data[index] = self.replaceWithSemanticBackdoor()
            #                 poison_count += 1

            else:  # will poison only a fraction of data when training
                if torch.rand(1) < backdoor_fraction:
                    new_targets[index] = backdoor_label
                    new_data[index] = self.replaceWithSemanticBackdoor()
                #                     poison_count += 1
                else:
                    new_data[index] = data[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_data.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_data, new_targets

    def replaceWithSemanticBackdoor(self):
        data, target = next(iter(self.semanticBackdoorDataLoader))
        data = data + torch.randn(data.size()) * 0.05  ## 'adding gaussian noise so that the backdoor can generalize'
        return data
