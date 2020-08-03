#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(1)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class Agg_dataset(torch.utils.data.Dataset):
    '''
        denote n be the number of clients,
        each entry of dataset is a 2-tuple of (weight delta, labels):= (1 x n tensor, 1 x n tensor)
        honest clients are labeled 1, malicious clients are labeled 0
    '''
    def __init__(self,path,attacks):
        super(Agg_dataset).__init__() 
        data=torch.load(path,map_location='cpu')
        data_tensors=torch.cat([data[param] for param in data],0)
        self.data=data_tensors
        self.label=attacks
        #center= mean of honest clients = sum/num
        label=attacks/torch.sum(attacks)
        self.center=torch.sum(data_tensors*label,1).view(-1,1)
        self.num_clients=attacks.shape[0]
        self.n=int(np.floor(self.data.shape[0]*sample_fraction))
        dimension=vector_dimension
        self.indexes=torch.randint(self.data.shape[0],(self.n,vector_dimension))
        
    def __getitem__(self, index):
#         

        data_out=self.data[self.indexes[index]]
        label_out=self.label
        center_out=self.center[self.indexes[index]]
        perm=torch.randperm(self.num_clients)
#         data_sorted=torch.sort(data_out)
#         perm=data_sorted[1]
        data_out_shuffled=torch.index_select(data_out, -1, perm)

        return data_out_shuffled, [label_out[perm],center_out]
#         return data_out,[label_out,center_out]
    def __len__(self):
#         return self.data.shape[0]//100
        return self.n

def get_concat_loader(validation_split=0.2,shuffle_dataset = True,path='./AggData',rounds=0):
    dict_datasets_test={}
    list_datasets_train=[]

    from os import listdir
    attacks=listdir(path)
    for attack in attacks:
        datasets_test_wrt_attack=[]
        path_to_data=f'{path}/{attack}/FedAvg_{rounds}.pt'
        label=torch.load(f'{path}/{attack}/label.pt')
        dataset=Agg_dataset(path_to_data,label)
        n_data=len(dataset)
        split=[int(n_data*(1-validation_split)),n_data-int(n_data*(1-validation_split))]
        split1,split2=torch.utils.data.random_split(dataset,split)
        list_datasets_train.append(split1)
        datasets_test_wrt_attack.append(split2)
        dict_datasets_test[attack]=datasets_test_wrt_attack
            
    dataset_train=torch.utils.data.dataset.ConcatDataset(list_datasets_train)
    data_test={}
    for attack, list_datasets_test in dict_datasets_test.items():
        data_test[attack]=torch.utils.data.dataset.ConcatDataset(list_datasets_test)
    len_data_test={key:len(item) for (key,item) in data_test.items()}
    
    print(f"Loaded {len(dataset_train)} training samples.")
    print(f"Loaded {sum(len_data_test.values())} validation samples.")


    

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                               shuffle=shuffle_dataset)
#     test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
#                                                     shuffle=shuffle_dataset)
    test_loaders={key:torch.utils.data.DataLoader(item, batch_size=batch_size)
                  for (key,item) in data_test.items()}
    return train_loader, test_loaders

# In[5]:



    
def test(net,test_loader,device,message_prefix):
    net.to(device)
    BCEloss = 0
    accuracy = 0
    accuracy_binary = 0
    accuracy_mean = 0
    accuracy_median = 0
    count = 0
    num_batches=len(test_loader)
#     printProgressBar(0, num_batches, prefix = 'Validation Progress:', suffix = 'Complete', length = 50)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            label = target[0].to(device)
            target = target[1].to(device)            
            outputs = net(data)
            BCEloss        +=nn.BCELoss(reduction='sum')(outputs[0],label)
            accuracy       +=F.l1_loss(outputs[1],                       target, reduction="sum")
            accuracy_binary+=F.l1_loss(outputs[2],                       target, reduction="sum")
            accuracy_mean  +=F.l1_loss(data.mean(-1).unsqueeze(-1),      target, reduction="sum")
            accuracy_median+=F.l1_loss(data.median(-1)[0].unsqueeze(-1), target, reduction="sum")
            count+=len(data)
#             printProgressBar(idx+1, num_batches, prefix = 'Validation Progress:', suffix = 'Complete', length = 50)
    print('%s: \t%.4E \t %.4E \t%.4E \t%.4E (BCE: %.4E)' % (message_prefix,accuracy/count, accuracy_binary/count, accuracy_mean/count,accuracy_median/count,BCEloss/count ))
    return accuracy/count, accuracy_binary/count, accuracy_mean/count, accuracy_median/count, BCEloss/count


# In[11]:
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-path" ,"--path_to_aggNet", type=str)
    parser.add_argument("-n" ,"--model_name", type=str)
    parser.add_argument("-net","--network", type=str)
    parser.add_argument("-vd","--vector_dimension", type=int, default =1)
    parser.add_argument("--path_to_training_data", type=str)
    parser.add_argument("-rd","--rounds", type=int, default =0)

    args=parser.parse_args()
    torch.manual_seed(0)
    num_epoch=200
    batch_size=8192
    device='cuda'
    data_root='./data'
    vector_dimension=args.vector_dimension
    sample_fraction=0.05 #number of sample to draw in each experiments

    import allocateGPU
    allocateGPU.allocate_gpu()
    
    
    import baseline
    import aggNet
    import aggNet_Blocks
    import agg_Blocks_Multiple
    import rand_net
    networks={'baseline'   :baseline.Net,
              'aggNetRes'  :aggNet.Net,
              'aggNetBlock':aggNet_Blocks.Net,
              'aggNetBlockMultiple':agg_Blocks_Multiple.Net,
              'random':rand_net.Net
              
             }
    Net=networks[args.network]
    

    from tensorboardX import SummaryWriter
    def write(name,tag, scalar,i=0):
        writer=SummaryWriter(f'./agg_logs/{name}')
        writer.add_scalar(tag, scalar, i)
        writer.close()


    train_loader, test_loaders=get_concat_loader(0.99,path=args.path_to_training_data,rounds=args.rounds)
    

    net=Net(vector_dimension,10)
    net.load_state_dict(torch.load(args.path_to_aggNet))



    Loss_net=[]
    Loss_mean=[]
    Loss_median=[]
    print('L1 loss:                                  model     model (binarized)     mean          median ')
    for (key,item) in sorted(test_loaders.items()):
        score=test(net,item,device,f'{key:<30}')
        Loss_net.append(score[1].item())
        Loss_mean.append(score[2].item())
        Loss_median.append(score[3].item())
        alias=f'validation/{key}/{args.model_name}'
        write(alias, 'BCE loss',  score[4],0)
        
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    def plot(xdata,ydata,xlabel,ylabel,title,savepath):
        _,ax=plt.subplots(figsize=(6,6))
        plt.scatter(xdata,ydata)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(left=-0.001)
        plt.ylim(bottom=-0.001)
        plt.xlim(right=0.004)
        plt.ylim(top=0.004)
        sup=max(ax.get_xlim()[1],ax.get_ylim()[1])
        plt.plot([0, sup], [0, sup], ls = "--",c =".3")
        plt.savefig(savepath)
    plot(Loss_net,Loss_mean,"l1: agg = net","l1: agg = mean",args.model_name,
        f"./results/aggResult/{args.model_name}_vsMean.svg")
    plot(Loss_net,Loss_median,"l1: agg = net","l1: agg = median",args.model_name,
        f"./results/aggResult/{args.model_name}_vsMedian.svg")
    import pandas as pd
    df=pd.DataFrame.from_dict({'net':Loss_net,'mean':Loss_mean,'median':Loss_median})
    df.index=sorted(test_loaders.keys())
    df.to_csv(f"./results/aggResult/{args.model_name}.csv")

         



