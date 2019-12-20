from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

class Client():
    def __init__(self,cid,model,dataLoader,optimizer,device):
        self.cid=cid
        self.model=model
        self.dataLoader=dataLoader
        self.optimizer=optimizer
        self.device=device
        self.log_interval=len(dataLoader)-1
        self.init_stateChange()
        self.originalState=deepcopy(model.state_dict())
        self.isTrained=False
        self.epoch=0
        
    def init_stateChange(self):
        states=deepcopy(self.model.state_dict())
        for param,values in states.items():
            values*=0
        self.stateChange=states
    def setModelParameter(self,states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState=deepcopy(states)
        self.model.zero_grad()
    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dataLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('client {} ## Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.cid, self.epoch, batch_idx * len(data), len(self.dataLoader.dataset),
                    100. * batch_idx / len(self.dataLoader), loss.item()))
        self.epoch+=1
        self.isTrained=True
        
#     def test(self,testDataLoader,steps,writer):
    def test(self,testDataLoader):

        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testDataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testDataLoader)
#         writer.add_scalar('test/loss', test_loss, steps)
#         writer.add_scalar('test/accuracy', correct / len(testDataLoader.dataset), steps)

#         print('client {} ##  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(self.cid,
#             test_loss, correct, len(testDataLoader.dataset),
#             100. * correct / len(testDataLoader.dataset)))
        
        
    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState=self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param]=newState[param]-self.originalState[param]
        self.isTrained=False
    def getDelta(self):
        return self.stateChange