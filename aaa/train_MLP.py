import os
import sys

sys.path.append(os.getcwd())
print(sys.path)
from runx.logx import logx

import torch
import torch.nn.functional as F
from aaa.mlp import MLP
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.metrics import AverageMeter

def getTensorData(path_to_folder, idx):
    try:
        data = torch.load(f'{path_to_folder}/pca_{idx}.pt')
        label = torch.load(f'{path_to_folder}/label.pt')
    except:
        return None
    # update vectors with extreme large magnitude can cause divergence, limit it here
    cap=5.0
    if data.max()>cap or data.min()<-cap:
        return None
    
    # scale each dimension randomly
    data*=(torch.randn((data.shape[0],1))*0.1+1)
    
    # permute the clients in random order
    perm = torch.randperm(len(label))
    data = data[:, perm]
    
    label = label[perm]

    label_norm = F.normalize(label, p=1, dim=-1)
    label_norm.unsqueeze(0)

    center = data.matmul(label_norm.unsqueeze(-1))

    x = data
    y = label.expand_as(x)
    c = center.expand_as(x)
    y_ = label.unsqueeze(-1)
    return x, y, c




class FLdata(Dataset):

    def __init__(self, path_to_folder, indexes):
        self.path_to_folder = path_to_folder
        self.indexes = []
        
        for i in indexes:
            sample = getTensorData(self.path_to_folder, i)
            if sample!=None:
                self.indexes.append(i)
        self.size = len(self.indexes)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = getTensorData(self.path_to_folder, self.indexes[idx])   
        if sample ==None:
            print('DEBUG: ',self.path_to_folder, self.indexes, idx)
        return sample

# getting a hard prediction by binarizing the affinity matrix
def getBinaryPred(model, x, beta):
    weight = model.getWeight(beta, x)
    weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1], 0)(weight)
    weight = F.normalize(weight, p=1, dim=-1)
    predB = torch.einsum('bqi,bji -> bjq', weight, x)
    return predB



def test(model, testloader):
    criterion=torch.nn.L1Loss(reduction='mean')
    losses_soft = AverageMeter('loss soft')
    losses_hard = AverageMeter('loss hard')

    model.eval()

    with torch.no_grad():
        for x, y, c in testloader:
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
#             beta = x.median(dim=-1, keepdim=True)[0]
            beta = x.mean(dim=-1, keepdim=True)

            pred = model.cuda()(beta, x)
            loss = criterion(pred, c[:, :, [0]])

            pred_b = getBinaryPred(model, x, beta)
            loss_b = criterion(pred_b, c[:, :, [0]])

            losses_soft.update(loss.cpu().detach().numpy(), n = x.shape[0])
            losses_hard.update(loss_b.cpu().detach().numpy(),n = x.shape[0])
    return losses_soft, losses_hard


def test_classes(model, testloader):
    def criterion(pred, gt):
        loss = (pred == gt).all(dim=-1).float().mean()
        return loss

    accuracies = AverageMeter('Acc')

    model.eval()

    with torch.no_grad():
        for x, y, c in testloader:
#             beta = x.median(dim=-1, keepdim=True)[0]
            beta = x.mean(dim=-1, keepdim=True)

            weight = model.getWeight(beta, x)
            weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1], 0)(weight)
            pred = (weight != 0) * 1.0
            accuracy = criterion(pred, y[:, [0], :])
            accuracies.update(accuracy.cpu().detach().numpy(), n = x.shape[0])

    return accuracies


def test_classes_l1(model, testloader):

    criterion=torch.nn.L1Loss(reduction='mean')
    losses = AverageMeter('l1 loss on classes')

    model.eval()

    with torch.no_grad():
        for x, y, c in testloader:
#             beta = x.median(dim=-1, keepdim=True)[0]
            beta = x.mean(dim=-1, keepdim=True)

            pred_weight = model.getWeight(beta, x)
            #             weight = torch.nn.Threshold(0.8 * 1.0 / weight.shape[-1],0)(weight)
            #             pred =  (weight != 0) * 1.0
            gt = y[:, [0], :]
            gt = gt / gt.sum(2, keepdim=True)
            loss = criterion(pred_weight, gt)
            losses.update(loss.cpu().detach().numpy(), n = x.shape[0])

    return losses

def test_scores_l1(model, testloader):

    criterion=torch.nn.L1Loss(reduction='mean')
    losses = AverageMeter('l1 loss on scores')
    
    model.eval()

    with torch.no_grad():
        for x, y, c in testloader:
#             beta = x.median(dim=-1, keepdim=True)[0]
            beta = x.mean(dim=-1, keepdim=True)

            pred_scores = model.getScores(beta, x)

            gt = y[:, [0], :]
            gt = gt*2-1
            loss = criterion(pred_scores, gt)
            losses.update(loss.cpu().detach().numpy(), n = x.shape[0])

    return losses

if __name__ == "__main__":

    import argparse

    # defined command line options
    # this also generates --help and error handling
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--path_prefix", type=str, help="for example: ./AggData/dirichlet/cifar/")
    parser.add_argument("--log_path", type=str, help="for example: ./results/ablation/")
    parser.add_argument("--train_path",
                        nargs="+",
                        type=str,
                        help="for example: No_Attack(0)")
    parser.add_argument("--test_path",
                        nargs="+",
                        type=str,
                        help="for example: No_Attack(1)")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_round", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--class_reg_scale", type=float, default=1e-2, help="from 0 to 1")
    args = parser.parse_args()
    args_dict=vars(args).copy()
    args_dict.pop('train_path')
    args_dict.pop('test_path')
    logx.initialize(logdir=args.log_path, coolname=True, tensorboard=True,
                    hparams=args_dict)
    for arg in vars(args):
        logx.msg(arg)
        logx.msg(str(getattr(args, arg)))

    path_prefix = args.path_prefix
    train_path = args.train_path
    test_path = args.test_path
    eps = args.eps
    scale = args.scale
    log_path = args.log_path
    epochs = args.epochs
    max_round = args.max_round
    batch_size = args.batch_size
    learning_rate = args.lr
    
    
    assert args.class_reg_scale<=1 and args.class_reg_scale>=0, "The scale of losses loss_1*t+loss_2*(1-t) should be from 0 to 1"
    #     exit(0)
    from utils import allocateGPU

    allocateGPU.allocate_gpu()

    
    trainDataset = ConcatDataset(
        list(filter(lambda dset:len(dset)>0, 
               [FLdata(path_prefix + path_to_folder, list(range(0, max_round ))) for path_to_folder in
         test_path]
              ))
    )

    validDataset = ConcatDataset(
        list(filter(lambda dset:len(dset)>0, 
               [FLdata(path_prefix + path_to_folder, list(range(0, max_round // 3 ))) for path_to_folder in
         test_path]
              ))
    )
    
    attack_types=['noAttack','backdoor','labelFlipping','omniscient']
    path_to_attacks=dict([(attack,[p for p in test_path if attack in p]) for attack in attack_types])
    path_to_attacks=dict([(attack,paths) for (attack,paths) in path_to_attacks.items() if len(paths)>0])
    
    # A dataset for each type of attacks
    testSet = [ConcatDataset(
        list(filter(lambda dset:len(dset)>0,
               [FLdata(path_prefix + p, list(range(0, max_round))) for p in paths]
              ))
    )
               for paths in path_to_attacks.values()]
    testDataset = ConcatDataset(testSet)
    print(*path_to_attacks.keys(), sep=',', file=open(f"{log_path}/l1.csv", "w"))
    print(*path_to_attacks.keys(), sep=',', file=open(f"{log_path}/acc.csv", "w"))
    print(*path_to_attacks.keys(), sep=',', file=open(f"{log_path}/s.csv", "w"))

    
    dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    logx.msg('Loaded train dataset')
    validloader = DataLoader(validDataset, batch_size=batch_size, shuffle=True)
    logx.msg('Loaded validation dataset')
    testloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)
    logx.msg('Loaded test dataset')
    testloaderSeparate = [DataLoader(testItem, batch_size=max_round, shuffle=True) for testItem in testSet]
    logx.msg('Loaded test dataset separately')
    k = trainDataset[0][0].shape[0]

    model = MLP(k*10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss(reduction='mean')

    # for query=key:=identity, l1 score ~= 0.6
    train_loss = []
    train_loss_hard = []
    valid_loss = []
    valid_loss_hard = []
    train_acc = []
    valid_acc = []
    test_acc = []
    best_testScore=1000000
    for i in range(epochs):
        losses_soft = AverageMeter('loss soft')
        losses_hard = AverageMeter('loss hard')
        losses_median_ref = AverageMeter('loss median')
        losses_mean_ref = AverageMeter('loss mean')

        for x, y, c in dataloader:
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
            ## give a different prior to avoid over fitting
#             beta = x.median(dim=-1, keepdim=True)[0]  # if (torch.rand(1)<0.5).item() else x.mean(dim=-1,keepdim=True)
            convex_weight = torch.rand(x.shape[0],x.shape[2],1) # bsz by n by 1
            convex_weight = F.normalize(convex_weight, p=1, dim=1)
            convex_weight = convex_weight.cuda()
            beta = torch.bmm(x,convex_weight)

            pred ,predW, predS = model.cuda().getALL(beta, x)
            
            predW=predW.unsqueeze(1)
            predS=predS.unsqueeze(1)
            
            gt_Center=c[:, :, [0]]
            target_loss = criterion(pred, gt_Center)
            
            gt_W=y[:, [0], :]
            gt_W = gt_W / gt_W.sum(2, keepdim=True)
            class_loss = criterion(predW, gt_W)
            loss = target_loss * (1-args.class_reg_scale)+\
                        class_loss * (args.class_reg_scale)

            pred_b = getBinaryPred(model, x, beta)
            loss_b = criterion(pred_b, gt_Center)

            losses_soft.update(loss.cpu().detach().numpy(), n = x.shape[0])
            losses_hard.update(loss_b.cpu().detach().numpy(), n = x.shape[0])

            loss_median_ref = criterion(beta, gt_Center)
            loss_mean_ref = criterion(x.mean(dim=-1, keepdim=True), gt_Center)

            losses_median_ref.update(loss_median_ref.cpu().detach().numpy(), n = x.shape[0])
            losses_mean_ref.update(loss_mean_ref.cpu().detach().numpy(), n =x.shape[0])

            loss.backward()
            optimizer.step()

        logx.msg(f'Model trained for {i}-th epoch')
        losses_soft_val, losses_hard_val = test(model, testloader)

        train_score = test_classes_l1(model.cpu(), dataloader)
        valid_score = test_classes_l1(model.cpu(), validloader)
        test_score = test_classes_l1(model.cpu(), testloader)
        logx.msg(f'Model tested for {i}-th epoch')
        logx.msg(
            f'{losses_soft.avg}|{losses_hard.avg}|{losses_soft_val.avg}|{losses_hard_val.avg}|{losses_median_ref.avg}|{losses_mean_ref.avg}\t \
        accuracy: {train_score.avg:.6f}, {valid_score.avg:.6f}, {test_score.avg:.6f}')


        
        test_acc_sep = [test_classes(model.cpu(), testItem).avg for testItem in
                        testloaderSeparate]
        print(*test_acc_sep, sep=',', file=open(f"{log_path}/acc.csv", "a"))
        
        test_acc_sep = [test_classes_l1(model.cpu(), testItem).avg for testItem in
                        testloaderSeparate]
        print(*test_acc_sep, sep=',', file=open(f"{log_path}/l1.csv", "a"))

        test_acc_sep = [test_scores_l1(model.cpu(), testItem).avg for testItem in
                        testloaderSeparate]
        print(*test_acc_sep, sep=',', file=open(f"{log_path}/s.csv", "a"))
        
        train_acc=test_classes(model.cpu(), dataloader)
        test_acc=test_classes(model.cpu(), testloader)

        metrics = {'loss(train)': losses_soft.avg, 'loss on class(train)': train_score.avg}
        logx.metric(phase='train', metrics=metrics, epoch=i)
        
        metrics = {'loss': losses_soft_val.avg, 
                   'loss on class': test_score.avg, 
                   'acc': test_acc.avg,
                   'loss(train)': losses_soft.avg, 
                   'loss on class(train)': train_score.avg, 
                   'acc(train)': train_acc.avg,
                   'loss _median': losses_median_ref.avg,
                   'loss _mean': losses_mean_ref.avg
                  }
        logx.metric(phase='val', metrics=metrics, epoch=i)
        
        save_dict = {'epoch': i + 1,
                     'state_dict': model.state_dict(),
                     'best_loss': test_score.avg,
                     'optimizer' : optimizer.state_dict()}
        logx.save_model(save_dict, metric=test_score.avg, epoch=i, higher_better=False)
        
        