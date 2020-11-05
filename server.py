from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F

import utils
from backdoor_semantic_utils import SemanticBackdoor_Utils
from backdoor_utils import Backdoor_Utils


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
        self.GAR = self.FedAvg
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
        print("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, count, accuracy))
        return test_loss, accuracy

    def test_backdoor(self):
        print("[Server] Start testing backdoor")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = Backdoor_Utils()
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
        print(
            '\nTest set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                                    len(
                                                                                                        self.dataLoader.dataset),
                                                                                                    accuracy))
        return test_loss, accuracy

    def test_semanticBackdoor(self):
        print("[Server] Start testing semantic backdoor")

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
        print(
            '\nTest set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                                             correct,
                                                                                                             len(
                                                                                                                 self.dataLoader.dataset),
                                                                                                             accuracy))
        return test_loss, accuracy, data, pred

    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()

        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        Delta = self.GAR(selectedClients)
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        print(f"Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = True
        if saveAsPCA:
            import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            print(f'The PCA projections of Weight delta has been saved to {savepath} (with shape {proj_vec.shape})')
            return
        savepath = f'{self.savePath}/{self.iter}.pt'

        torch.save(Delta, savepath)
        print(f'Weight delta has been saved to {savepath}')

    ## Aggregation functions ##

    def set_GAR(self, gar):
        if gar == 'fedavg':
            self.GAR = self.FedAvg
        elif gar == 'median':
            self.GAR = self.FedMedian
        elif gar == 'gm':
            self.GAR = self.geometricMedian
        elif gar == 'krum':
            self.GAR = self.krum
        elif gar == 'mkrum':
            self.GAR = self.mkrum
        elif gar == 'foolsgold':
            self.GAR = self.foolsGold
        elif gar == 'residualbase':
            self.GAR = self.residualBase
        elif gar == 'attention':
            self.GAR = self.net_attention
        elif gar == 'mlp':
            self.GAR = self.net_mlp
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    def geometricMedian(self, clients):
        from rules.geometricMedian import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def krum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    def mkrum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum').cpu()(arr.cpu()))
        return out

    def foolsGold(self, clients):
        from rules.foolsGold import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def residualBase(self, clients):
        from rules.residualBase import Net
        out = self.FedFuncWholeStateDict(clients, Net().main)
        return out

    def net_attention(self, clients):
        from aaa.attention import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

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
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]

        resultDelta = func(deltas)

        Delta.update(resultDelta)
        return Delta
