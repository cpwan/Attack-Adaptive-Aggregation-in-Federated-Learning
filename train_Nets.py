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
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#', printEnd="\r"):
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
    def __init__(self,path,label):
        super(Agg_dataset).__init__() 
        data = torch.load(path,map_location='cpu')
#         data_tensors=torch.cat([data[param] for param in data],0)
        data_tensors = torch.cat([data[param] for param in data if len(data[param].shape)],0)

        self.num_clients = label.shape[0] #n
        self.data = data_tensors # dimension of network x n
        self.label = label # 1 by n vector
        # samples only a fraction of data
        self.n = int(self.data.shape[0] * sample_fraction)
        dimension = vector_dimension
        self.indexes = torch.randint(self.data.shape[0],(self.n,vector_dimension)) # dimension of network by vdim
    def get_true_center(self, data, label):
        #center= mean of honest clients = sum values/num of clients
        weight = label / torch.sum(label)        
        center = torch.sum(data * weight,1).view(-1,1)
        return center
    def permute(self,data,label):
        perm = torch.randperm(self.num_clients)
        data_ = torch.index_select(data, -1, perm)
        label_ = label[perm]
        return data_,label_
    def linear_transform(self,data,label):
        a = 2 * torch.rand(1) - 1 #[-1,+1], a~ Uniform(-1,1)
        b = 2 * torch.rand(1) - 1 #[-1,+1]
        
#         a=128**a #[1/128,128], log(a)~ Uniform(-128,128)
#         b=128**b #[1/128,128]
        
#         a=torch.sign((torch.rand(1)-0.5))*a #a~
#         0.5(exp(Uniform(-128,128)))+0.5(-exp(Uniform(-128,128)))
#         b=torch.sign((torch.rand(1)-0.5))*b
#         #a and b are uniform in magnitude
        a*=100
        b*=100
        
        data_ = data * a + b
        label_ = label
        return data_,label_
    
    def __getitem__(self, index):         
        data_out = self.data[self.indexes[index]]
        label_out = self.label
        
        data_out, label_out = self.permute(data_out, label_out)
        
        data_out, label_out = self.linear_transform(data_out, label_out)

        center_out = self.get_true_center(data_out,label_out)
        return data_out, [label_out,center_out]
#         return data_out,[label_out,center_out]
    def __len__(self):
#         return self.data.shape[0]//100
        return self.n


# In[4]:
def get_concat_loader(validation_split=0.2,shuffle_dataset=True,path='./AggData',rounds=0):
    list_datasets_test = []
    list_datasets_train = []

    from os import listdir
    attacks = listdir(path)
    for attack in attacks:
        path_to_data = f'{path}/{attack}/FedAvg_{rounds}.pt'
        label = torch.load(f'{path}/{attack}/label.pt')
        dataset = Agg_dataset(path_to_data,label)
        n_data = len(dataset)
        split = [int(n_data * (1 - validation_split)),n_data - int(n_data * (1 - validation_split))]
        split1,split2 = torch.utils.data.random_split(dataset,split)
        list_datasets_train.append(split1)
        list_datasets_test.append(split2)

    dataset_train = torch.utils.data.dataset.ConcatDataset(list_datasets_train)
    dataset_test = torch.utils.data.dataset.ConcatDataset(list_datasets_test)
    
    print(f"Loaded {len(dataset_train)} training samples.")
    print(f"Loaded {len(dataset_test)} validation samples.")


    

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                               shuffle=shuffle_dataset)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                    shuffle=shuffle_dataset)

    return train_loader, test_loader


# In[5]:


# In[5]:

def train(net,train_loader,criterion,optimizer,device, on=0):
    net.to(device)
    num_batches = len(train_loader)
    printProgressBar(0, num_batches, prefix = '  Training Progress:', suffix = 'Complete', length = 50)
    total_loss = 0.0
    count = 0.0
    for idx, (data,target) in enumerate(train_loader):
        data = data.to(device)
        target = target[on].to(device)
        optimizer.zero_grad()   
        output = net(data)
        loss = criterion(output[on], target)
        total_loss+=loss.detach().item()
        loss.backward()
        optimizer.step()
        
        count+=target.shape[0]
        loss_msg = ("(loss:%.4E)" % (total_loss / count))
        printProgressBar(idx + 1, num_batches, prefix = '  Training Progress:', suffix = 'Complete.' + loss_msg, length = 50)
    
def test(net,test_loader,device,message_prefix):
    net.to(device)
    BCEloss = 0
    accuracy = 0
    accuracy_binary = 0
    accuracy_mean = 0
    accuracy_median = 0
    count = 0
    num_batches = len(test_loader)
#     printProgressBar(0, num_batches, prefix = 'Validation Progress:', suffix
#     = 'Complete', length = 50)

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            label = target[0].to(device)
            target = target[1].to(device)            
            outputs = net(data)
            BCEloss        +=nn.BCELoss(reduction='sum')(outputs[0],label).item()
            accuracy       +=F.l1_loss(outputs[1],                       target, reduction="sum").item()
            accuracy_binary+=F.l1_loss(outputs[2],                       target, reduction="sum").item()
            accuracy_mean  +=F.l1_loss(data.mean(-1).unsqueeze(-1),      target, reduction="sum").item()
            accuracy_median+=F.l1_loss(data.median(-1)[0].unsqueeze(-1), target, reduction="sum").item()
            count+=target.shape[0]          

            metric_msg = ('%s: \t%.4E \t %.4E \t%.4E \t%.4E (BCE: %.4E)\t\t' % (message_prefix,accuracy / count / vector_dimension, accuracy_binary / count / vector_dimension, accuracy_mean / count / vector_dimension,accuracy_median / count / vector_dimension,BCEloss / count))
                        
            printProgressBar(idx + 1, num_batches, prefix = metric_msg + 'Validation Progress:', suffix = 'Complete', length = 50)
    count*=vector_dimension

    return accuracy / count, accuracy_binary / count, accuracy_mean / count, accuracy_median / count, BCEloss * vector_dimension / count


# In[11]:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-vd","--vector_dimension",type=int,default=1,help="Dimension of weight feeding to the network")
    parser.add_argument("-l2","--weight_decay",type=float,default=0,help="Weight decay for l2 regularization")
    parser.add_argument("-n" ,"--model_name", type=str)
    parser.add_argument("-net","--network", type=str)
    parser.add_argument("-path","--path_to_training_data", type=str)
    parser.add_argument("-loss","--loss", type=str,default='BCE')

    
    args = parser.parse_args()
    torch.manual_seed(0)
    num_epoch = 200
    batch_size = 32768
    device = 'cuda'
    data_root = './data'
    vector_dimension = args.vector_dimension
    sample_fraction = 1.0 #number of sample to draw in each experiments

    import allocateGPU
    allocateGPU.allocate_gpu()
    
    
    import baseline
    import aggNet
    import aggNet_Blocks
    import agg_Blocks_Multiple
    import rand_net
    import aggNet_noMedian
    import aggNet_Blocks_normalize
    import geometricMedian
    import nnsort
    networks = {'baseline'   :baseline.Net,
              'aggNetRes'  :aggNet.Net,
              'aggNetBlock':aggNet_Blocks.Net,
              'aggNetBlockMultiple':agg_Blocks_Multiple.Net,
              'random':rand_net.Net,
            'aggNet_noMedian':aggNet_noMedian.Net  ,
              'aggNetBlockNormalize':aggNet_Blocks_normalize.Net,
              'gm':geometricMedian.Net,
                'netNeuralSort':nnsort.Net,
             }
    Net = networks[args.network]
    
    import torch.optim as optim





    from tensorboardX import SummaryWriter
    def write(name,tag, scalar,i=0):
        writer = SummaryWriter(f'./agg_logs/{name}')
        writer.add_scalar(tag, scalar, i)
        writer.close()


    # In[12]:


    accuracy_list = {}


    # In[13]:
    save_frequncy = 1

    train_loader, test_loader = get_concat_loader(path=args.path_to_training_data,rounds=0)

    if args.loss == 'BCE':
        criterion = torch.nn.BCELoss()
        mode = 0
    else:
        criterion = torch.nn.L1Loss()
        mode = 1
    for lr in [0.01]:

        net = Net(vector_dimension,10)

        # args.path_to_training_data = './xxxx'
        training_alias = f'{args.path_to_training_data[2:]}/{args.model_name}'
        if training_alias in accuracy_list:
            continue

        print('Start training of %s' % training_alias)
        if not args.network == 'random':
            optimizer = optim.Adam(net.parameters(), lr=lr , weight_decay=args.weight_decay)
        print('L1 loss:\tmodel \tmodel (binarized) \tmean \tmedian ')
        test(net,test_loader,device,f'Epoch -1')
        for epoch in range(num_epoch):
            if not args.network == 'random':
                train(net,train_loader,criterion,optimizer,device,mode)
            score = test(net,test_loader,device,f'Epoch {epoch}')
            if (epoch + 1) % save_frequncy == 0:
                torch.save(net.state_dict(),f'./aggNet/{args.model_name}_dim{vector_dimension}_{epoch}.pt')

            rounds = 0
            if epoch % 10 == 9:
                #use another set of data once in a while
                rounds = (rounds + 1) % 1
                train_loader, test_loader = get_concat_loader(path=args.path_to_training_data,rounds=rounds)
            write(training_alias + '/weighted', 'l1 loss', score[0],epoch)
            write(training_alias + '/binary', 'l1 loss',  score[1],epoch)  
            write(training_alias + '/mean', 'l1 loss',    score[2],epoch)
            write(training_alias + '/median', 'l1 loss',  score[3],epoch)
            write(training_alias + '/BCEloss', 'BCE loss',  score[4],epoch)



