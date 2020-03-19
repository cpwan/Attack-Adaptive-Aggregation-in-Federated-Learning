from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

class Aggregator():
    def __init__(self,clients):
        assert len(clients)>0, "Empty list of clients to be aggregated"
        self.deltas=[c.getDelta() for c in clients]
        
        
        emptyStates=deepcopy(clients[0].getDelta) # get the shape of the model
        for param,values in emptyStates.items():
            emptyStates.update({param:values*0})            
        self.Delta=emptyStates
        
    def func():
        raise NotImplementedError()
        
        
    def FedFunc(self):
        '''
        apply func to each paramters across clients
        '''
        func=self.func
        
        for param in self.Delta:

            ##stacking the weight in the innerest dimension
            param_stack=torch.stack([delta[param].cpu() for delta in deltas],-1) # d1 x d2 x d3 x... xnum clients
            shaped=param_stack.view(-1,len(clients)) #d1*d2*d3*...  x num clients
            ##applying `func` to every array (of size `num_clients`) in the innerest dimension
            buffer=torch.stack(list(map(func,[shaped[i] for i in range(shaped.size(0))]))).reshape(self.Delta[param].shape)
            self.Delta[param]=buffer

    def FedFuncPerLayer(self,clients,func):
        '''
        apply func to each layer across clients
        '''
        func=self.func

        for param in self.Delta:
            
            ##stacking the weight in the innerest dimension
            ## size of layer x1x number of clients
            param_stack=torch.stack([delta[param].cpu() for delta in deltas],-1) # d1 x d2 x d3 x... xnum clients
            shaped=param_stack.view(-1,1,len(clients)) #d1*d2*d3*... x 1 x num clients
            
            dset=torch.utils.data.TensorDataset(shaped)
            dloader=torch.utils.data.DataLoader(dset,batch_size=8192)
            result=[]
            for data in dloader:
                #data: bacth_size x 1 x num clients
                result.append(func(data[0]))
            result_tensor=torch.stack(result)
            buffer=result_tensor.reshape(self.Delta[param].shape)
            self.Delta[param]=buffer
            
    def FedFuncNbhPerLayer(self,clients,func,vd=1):
        '''
        apply func to each layer across clients
        for each entry in the layer, sample (vd-1) other entries in the layer with that entry,
        feed it to the GAR
        '''
        func=self.func

        
        def getNbh(t1,vd):
            '''
            -in
            t1: tensor with shape d1 x 1 x num clients
            vd: the dimension of the vector to be fed to deep GAR, equals to the number of nbh  to be sampled
            -out
            entriesWithNbh: tensor with shape d1 x vd x num clients, with entriesWithNbh[:,0,:] being the original entry 
            '''
            d1=t1.size(0)
            num_clients=t1.size(2)
            randperms=[torch.tensor(range(d1))]+[torch.randperm(t1.size(0)) for i in range(vd-1)]
            randperms_index=torch.stack(randperms,dim=1)
            entriesWithNbh=t1[randperms_index].view(-1,vd,num_clients)
            return entriesWithNbh
            
        for param in self.Delta:
            
            ##stacking the weight in the innerest dimension
            ## size of layer x1x number of clients
            param_stack=torch.stack([delta[param] for delta in deltas],-1) # d1 x d2 x d3 x... xnum clients
            shaped=param_stack.view(-1,1,len(clients)) #d1*d2*d3*... x 1 x num clients
            dset=torch.utils.data.TensorDataset(getNbh(shaped,vd))
            dloader=torch.utils.data.DataLoader(dset,batch_size=8192)
            result=[]
            for data in dloader:
                result.append(func(data[0]))
            result_tensor=torch.stack(result)
            buffer=result_tensor.reshape(self.Delta[param].shape)
            self.Delta[param]=buffer
        return self.Delta
   