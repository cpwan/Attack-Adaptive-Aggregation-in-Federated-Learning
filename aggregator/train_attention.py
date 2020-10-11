import sys
import os
sys.path.append(os.getcwd())
print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from aggregator.attention import AttentionLoop
from torch.utils.data import Dataset, DataLoader,ConcatDataset



def getTensorData(path_to_folder,idx):

    data = torch.load(f'{path_to_folder}/pca_{idx}.pt')
    label = torch.load(f'{path_to_folder}/label.pt') 

    perm = torch.randperm(len(label))
    data = data[:,perm]
    label = label[perm]

    label_norm = F.normalize(label,p=1,dim=-1)
    label_norm.unsqueeze(0)

    center = data.matmul(label_norm.unsqueeze(-1))

    x = data
    y = label.expand_as(x)
    c = center.expand_as(x)
    y_ = label.unsqueeze(-1)
    return x,y,c

# getting a hard prediction by binarizing the affinity matrix
def getBinaryPred(model,x,beta):
    weight = model.getWeight(beta,x)
    weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1],0)(weight)
    weight = F.normalize(weight,p=1,dim=-1)
    predB = torch.einsum('bqi,bji -> bjq', weight, x)
    return predB
# helper class to record the accumulating loss
# +=: add loss
# print(**): print the average loss
class loss_acc():
    def __init__(self):
        self.sum = 0.0
        self.n = 0
    def __iadd__(self,x):
        self.sum+=x
        self.n+=1
        return self
    def value(self):
        return self.sum / self.n
    def __str__(self):
        return f'{self.sum/self.n:.6f}'
    
def test(model,testloader):
    lossCounter = loss_acc()
    lossCounter2 = loss_acc()
    
    model.eval()
    
    with torch.no_grad():
        for x,y,c in testloader:

            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
            beta = x.median(dim=-1,keepdim=True)[0]

            pred = model.cuda()(beta,x)
            loss = loss_fn(pred,c[:,:,[0]])

            pred_b = getBinaryPred(model,x,beta)
            loss_b = loss_fn(pred_b,c[:,:,[0]])

            lossCounter+=loss.cpu().detach().numpy()
            lossCounter2+=loss_b.cpu().detach().numpy()
    # print(f'{loss:.4f},{loss_b:.6f}')
    return lossCounter, lossCounter2

def test_classes(model,testloader):
    def loss_fn(pred,gt):
        correct = (pred == gt).all(dim=-1).sum()
        n = pred.shape[0]
        return correct,n

    correct = 0
    n = 0
    
    model.eval()
    
    with torch.no_grad():
        for x,y,c in testloader:

            beta = x.median(dim=-1,keepdim=True)[0]


            weight = model.getWeight(beta,x)
            weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1],0)(weight)
            pred =  (weight != 0) * 1.0
            accuracy = loss_fn(pred,y[:,[0],:])
            correct+=accuracy[0]
            n+=accuracy[1]
    # print(f'{loss:.4f},{loss_b:.6f}')
    return correct * 1.0 / n


def test_classes_hamming(model,testloader):
    def loss_fn(pred,gt):
#         print(pred.shape,gt.shape)
#         print("\n\n\n",pred,"\n\n\n",gt)
        n = pred.shape[0]
        correct = (pred - gt/gt.sum(2,keepdim=True)).abs().sum()

        return correct,n

    correct = 0
    n = 0
    
    model.eval()
    
    with torch.no_grad():
        for x,y,c in testloader:

            beta = x.median(dim=-1,keepdim=True)[0]


            weight = model.getWeight(beta,x)
#             weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1],0)(weight)
#             pred =  (weight != 0) * 1.0
            pred = weight
           
    
            accuracy = loss_fn(pred,y[:,[0],:])
            correct+=accuracy[0]
            n+=accuracy[1]
    # print(f'{loss:.4f},{loss_b:.6f}')
    return correct * 1.0 / n

class FLdata(Dataset):

    def __init__(self, path_to_folder, indexes):
        self.path_to_folder = path_to_folder
        self.indexes = indexes
        self.size = len(indexes)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        sample = getTensorData(self.path_to_folder,self.indexes[idx])
        return sample

if __name__ == "__main__":
    

    
    import argparse
    # defined command line options
    # this also generates --help and error handling
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--path_prefix",type=str, help="for example: ./AggData/dirichlet/cifar/")    
    parser.add_argument("--save_path",type=str, help="for example: ./aggregator/attention.pt")
    parser.add_argument("--log_path",type=str, help="for example: ./results/ablation/")
    parser.add_argument("--train_path",  
      nargs="+",  
      type=str,
      help="for example: No_Attack(0)")
    parser.add_argument("--test_path",
      nargs="+",
      type=str,
      help="for example: No_Attack(1)")
    parser.add_argument("--eps",type=float, default=0.01)    
    parser.add_argument("--scale",type=float, default=10)    

    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg,"\n",getattr(args, arg),"\n")

    path_prefix = args.path_prefix
    save_path = args.save_path
    train_path = args.train_path
    test_path = args.test_path
    eps=args.eps
    scale=args.scale
    log_path=args.log_path
    
    print(f'train soft| train hard| valid soft|valid hard| median| mean \t\t train|valid|test', file=open(f"{log_path}{eps}_{scale}.txt","w"))
    
    
#     exit(0)
    import allocateGPU
    allocateGPU.allocate_gpu()
    
    epochs = 1000
    hidden_size = 21
    learning_rate = 1e-4


    trainDataset = ConcatDataset([FLdata(path_prefix + path_to_folder,list(range(0,30))) for path_to_folder in train_path])
    validDataset = ConcatDataset([FLdata(path_prefix + path_to_folder,list(range(21,30))) for path_to_folder in train_path])
    testSet=[FLdata(path_prefix + path_to_folder,list(range(0,30))) for path_to_folder in test_path]
    testDataset = ConcatDataset(testSet)
    print(*test_path,sep=',', file=open(f"{log_path}{eps}_{scale}_long.txt","w"))

    dataloader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    validloader = DataLoader(validDataset, batch_size=32, shuffle=True)
    testloader = DataLoader(testDataset, batch_size=32, shuffle=True)
    testloaderSeparate=[DataLoader(testItem, batch_size=30, shuffle=True) for testItem in testSet]
    
    k = trainDataset[0][0].shape[0]

    model = AttentionLoop(k, hidden_size, nloop=5,eps=eps,scale=scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss(reduction='mean')

    # for query=key:=identity, l1 score ~= 0.6
train_loss = []
train_loss_hard = []
valid_loss = []
valid_loss_hard = []
train_acc = []
valid_acc = []
test_acc = []

for i in range(epochs):
    lossCounter = loss_acc()
    lossCounter2 = loss_acc()
    lossCounter_median_ref = loss_acc()
    lossCounter_mean_ref = loss_acc()

    for x,y,c in dataloader:
        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()
        c = c.cuda()
        beta = x.median(dim=-1,keepdim=True)[0]

        # loss=loss_fn(model.cuda()(x),y[:,[0],:])
        # loss=loss_fn(model.cuda()(x,x),z)
        pred = model.cuda()(beta,x)
        loss = loss_fn(pred,c[:,:,[0]])

        pred_b = getBinaryPred(model,x,beta)
        loss_b = loss_fn(pred_b,c[:,:,[0]])

        lossCounter+=loss.cpu().detach().numpy()
        lossCounter2+=loss_b.cpu().detach().numpy()

        loss_median_ref = loss_fn(beta,c[:,:,[0]])
        loss_mean_ref = loss_fn(x.mean(dim=-1,keepdim=True),c[:,:,[0]])

        lossCounter_median_ref+=loss_median_ref.cpu().detach().numpy()
        lossCounter_mean_ref+=loss_mean_ref.cpu().detach().numpy()
        # print(f'{loss:.4f},{loss_b:.6f}')

        loss.backward()
        optimizer.step()
      # print(f'train: {lossCounter},{lossCounter2}')


    lossCounter_test, lossCounter2_test = test(model,testloader)

    train_score = test_classes_hamming(model.cpu(),dataloader)
    valid_score = test_classes_hamming(model.cpu(),validloader)
    test_score = test_classes_hamming(model.cpu(),testloader)

    print(f'{lossCounter}|{lossCounter2}|{lossCounter_test}|{lossCounter2_test}|{lossCounter_median_ref}|{lossCounter_mean_ref}\t \
    accuracy: {train_score:.6f}, {valid_score:.6f}, {test_score:.6f}')
    print(f'{lossCounter}|{lossCounter2}|{lossCounter_test}|{lossCounter2_test}|{lossCounter_median_ref}|{lossCounter_mean_ref}\t \
    accuracy: {train_score:.6f}, {valid_score:.6f}, {test_score:.6f}', file=open(f"{log_path}{eps}_{scale}.txt","a"))
    print()
    train_loss.append(lossCounter.value())
    train_loss_hard.append(lossCounter2.value())
    valid_loss.append(lossCounter_test.value())
    valid_loss_hard.append(lossCounter2_test.value())

    train_acc.append(train_score)
    valid_acc.append(valid_score)
    test_acc.append(test_score)
    
    test_acc_sep=[test_classes_hamming(model.cpu(),testloader_sep).item() for testloader_sep in testloaderSeparate]
    print(*test_acc_sep,sep=',', file=open(f"{log_path}{eps}_{scale}_long.txt","a"))
    
    if ((i+1)%100==0):
        torch.save(model.state_dict(),f"{save_path[:-3]}_{i}.pt")
    
    
torch.save(model.state_dict(),save_path)
