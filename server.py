from __future__ import print_function
from runx.logx import logx

from copy import deepcopy

import torch
import torch.nn.functional as F

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
import time
from utils import metrics

class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        logx.msg("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        
        accuracy = metrics.AverageMeter('Acc')
        losses = metrics.AverageMeter('loss')
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    
                num_instance=data.shape[0]
                num_correct_pred=pred.eq(target.view_as(pred)).sum().item()
            
                accuracy.update(num_correct_pred/(num_instance+1e-16),num_instance)
                losses.update(loss,num_instance)

        self.model.cpu()  ## avoid occupying gpu when idle
        logx.msg(
            f'[Server] Test set: Average loss: {losses.avg:.4f}, Accuracy: {accuracy.sum}/{accuracy.count} ({accuracy.avg*100:.0f}%)\n')
        return losses.avg, accuracy.avg*100
    
    def test_labelFlipping(self):
        logx.msg("[Server] Start testing label flipping\n")
        self.model.to(self.device)
        self.model.eval()        
        success_rates = metrics.AverageMeter('ASR')
        from clients_attackers import config
        attacked_classes = config['label_flipping']['attacked_classes']
        target_class = config['label_flipping']['target_class']
        
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                success_rate, n_instance = metrics.LF_success_rate(pred.cpu(), target.cpu(), 
                                                                   attacked_classes, target_class)
                success_rates.update(success_rate, n_instance)


        self.model.cpu()  ## avoid occupying gpu when idle
        logx.msg(
            f'[Server] Test set (label flipping): Success rate: \
            {success_rates.sum}/{success_rates.count} ({success_rates.avg*100:.0f}%)\n'
        )
        
        return success_rates.avg*100
    def test_backdoor(self):
        logx.msg("[Server] Start testing backdoor\n")
        self.model.to(self.device)
        self.model.eval()
        
        success_rates = metrics.AverageMeter('ASR')
        losses = metrics.AverageMeter('Attack loss')
#         test_loss = 0
#         correct = 0
        utils = Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
#                 test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                num_success_attack = pred.eq(target.view_as(pred)).sum().item()
                
                num_instance=data.shape[0]
                loss=self.criterion(output, target, reduction='sum').item()
                losses.update(loss, num_instance)
                success_rates.update(num_success_attack/(num_instance+1e-16), num_instance)

#         test_loss /= len(self.dataLoader.dataset)
#         accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logx.msg(
            f'[Server] Test set (Backdoored): Average loss: {losses.avg:.4f}, Success rate: {success_rates.sum}/{success_rates.count} ({success_rates.avg*100:.0f}%)\n')
        return losses.avg, success_rates.avg*100

    def test_semanticBackdoor(self):
        logx.msg("[Server] Start testing semantic backdoor")

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logx.msg(
            '[Server] Test set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                                             correct,
                                                                                                             len(
                                                                                                                 self.dataLoader.dataset),
                                                                                                             accuracy))
        return test_loss, accuracy, data, pred

    def train(self, group):
        client_runtime={}
        
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            tic = time.perf_counter()
            c.train()
            toc = time.perf_counter()
            client_runtime[c.cid]=toc - tic
            logx.msg(f"[Client {c.cid}] The training takes {toc - tic:0.6f} seconds.\n")
            c.update()
        crt_list=client_runtime.values()
        crt_avg=sum(crt_list)/len(crt_list)
        logx.msg(f"[Client] The average training time of {len(client_runtime)} clients is {crt_avg:0.6f} seconds.\n")
        
        if self.isSaveChanges:
            self.saveChanges(selectedClients)
            
        Delta, aggregation_runtime = self.AR(selectedClients)
        
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1
        return crt_avg, aggregation_runtime

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        logx.msg(f"[Server] Saving the model weight of the trainable paramters:\n {len(Delta.keys())}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = True
        saveOriginal = False
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            logx.msg(f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
#             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            logx.msg(f'[Server] Update vectors have been saved to {savepath}')

    ## Aggregation functions ##

    def set_AR(self, ar):
        AR_dict={
            'fedavg':self.FedAvg,
            'median':self.FedMedian,
            'gm':self.geometricMedian,
            'krum':self.krum,
            'mkrum':self.mkrum,
            'foolsgold':self.foolsGold,
            'residualbase':self.residualBase,
            'attention':self.net_attention,
            'mlp':self.net_mlp
        }
        if ar in AR_dict:
            self.AR=AR_dict[ar]
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

###
# The aggregation has to be done on cpu if there are insufficient gpu memory (e.g. when there are too many clients)
###
    def geometricMedian(self, clients):
        from rules.geometricMedian import Net
        self.Net = Net
#         out = self.FedFuncWholeNet(clients, lambda arr: Net()(arr))
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def krum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
#         out = self.FedFuncWholeNet(clients, lambda arr: Net('krum')(arr))
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    def mkrum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
#         out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum')(arr))
        out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum').cpu()(arr.cpu()))
        return out

    def foolsGold(self, clients):
        from rules.foolsGold import Net
        self.Net = Net
#         out = self.FedFuncWholeNet(clients, lambda arr: Net()(arr))
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def residualBase(self, clients):
        from rules.residualBase import Net
        out = self.FedFuncWholeStateDict(clients, Net().main)
        return out

    def net_attention(self, clients):
        from aaa.attention import Net

        net = Net(self.path_to_aggNet)

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

    def net_mlp(self, clients):
        from aaa.mlp import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

        ## Helper functions, act as adaptor from aggregation function to the federated learning system##

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        stackedVecs = torch.stack(vecs, 1).unsqueeze(0) # input as 1 by d by n
        stackedVecs=stackedVecs.cpu()
        
        tic = time.perf_counter()
        result = func(stackedVecs)  
        toc = time.perf_counter()
        aggregation_runtime=toc - tic
        logx.msg(f"[Server] The aggregation takes {aggregation_runtime:0.6f} seconds. (only the computation part)\n")
        result = result.view(-1)
        result=result.cpu()
        utils.vec2net(result, Delta)
        return Delta, aggregation_runtime

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]
        
        
        stacked = utils.stackStateDicts(deltas)

        param_trainable = utils.getTrainableParameters(self.model)
        param_nontrainable = [param for param in stacked.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del stacked[param]
        from utils import convert_pca
        tic = time.perf_counter()
        proj_vec = convert_pca._convertWithPCA(stacked)
            

        resultDelta = func((proj_vec,deltas))
        toc = time.perf_counter()
        aggregation_runtime=toc - tic
        logx.msg(f"[Server] The aggregation takes {aggregation_runtime:0.6f} seconds. (only the computation part)\n")
        
        Delta.update(resultDelta)
        return Delta, aggregation_runtime
