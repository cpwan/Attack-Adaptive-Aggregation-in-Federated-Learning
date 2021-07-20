from __future__ import print_function
from runx.logx import logx



import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients_attackers import *
from server import Server
import importlib


def main(args):
       
    
    logx.msg('#####################')
    logx.msg('#####################')
    logx.msg('#####################')
    logx.msg(f'    Task:\t{args.dataset}\n\
    Aggregation Rule:\t{args.AR}\n\
    Data distribution:\t{args.loader_type}\n\
    Attacks:\t{args.attacks} ')
    logx.msg('#####################')
    logx.msg('#####################')
    logx.msg('#####################')

    torch.manual_seed(args.seed)

    device = args.device

    attacks = args.attacks

    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.attacks}/{args.AR}')

    #######################################
    ##### Preparing the task,         #####
    ##### setup the model and dataset #####
    #######################################
    
    # This imports `tasks.<args.dataset>` as `task`
    # For example, if `args.dataset` is 'mnist', then `task` refers to `tasks.mnist`
    # We expect a file named <args.dataset>.py under the ./tasks directory:
    # ./tasks/
    #     mnist.py
    #     cifar.py
    #     imagenet.py
    #     imdb.py
    # In the <args.dataset>.py, we expect 2 methods and 1 class: train_dataloader, test_dataloader, Net
    task = importlib.import_module(f'tasks.{args.dataset}')
    if args.train_on_testset:
        task.train_on_testset=True
    trainData = task.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                       store=False)
    testData = task.test_dataloader(args.test_batch_size)
    Net = task.Net
    criterion = F.cross_entropy
    
    logx.msg("Data distribution of clients:\n"+"\n".join([f'Client{i}:\t{len(trainData[i].dataset)}' for i in range(args.num_clients)]))
        
    #######################################
    ##### Initialize the server       #####
    #######################################
    # create server instance
    model0 = Net()
    server = Server(model0, testData, criterion, device)
    server.set_AR(args.AR)
    server.path_to_aggNet = args.path_to_aggNet
    
    #######################################
    ##### Initialize the clients      #####
    #######################################
    if args.save_model_weights:
        dataset_suffix = 'train' if not args.train_on_testset else 'test'
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}_n{args.num_clients}_{dataset_suffix}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        '''
        honest clients are labeled as 1, malicious clients are labeled as 0
        '''
        label = torch.ones(args.num_clients)
        for i in args.attacker_list_labelFlipping:
            label[i] = 0
        for i in args.attacker_list_labelFlippingDirectional:
            label[i] = 0
        for i in args.attacker_list_omniscient:
            label[i] = 0
        for i in args.attacker_list_backdoor:
            label[i] = 0
        for i in args.attacker_list_semanticBackdoor:
            label[i] = 0
        torch.save(label, f'{server.savePath}/label.pt')
    # create clients instance
    attacker_list_labelFlipping = args.attacker_list_labelFlipping
    attacker_list_omniscient = args.attacker_list_omniscient
    attacker_list_backdoor = args.attacker_list_backdoor
    attacker_list_labelFlippingDirectional = args.attacker_list_labelFlippingDirectional
    attacker_list_semanticBackdoor = args.attacker_list_semanticBackdoor
    for i in range(args.num_clients):        
        model = Net()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if i in attacker_list_labelFlipping:
            client_i = Attacker_LabelFlipping01swap(i, model, trainData[i], optimizer, criterion, device,
                                                    args.inner_epochs)
        elif i in attacker_list_labelFlippingDirectional:
            client_i = Attacker_LabelFlipping1to7(i, model, trainData[i], optimizer, criterion, device,
                                                  args.inner_epochs)
        elif i in attacker_list_omniscient:
            client_i = Attacker_Omniscient(i, model, trainData[i], optimizer, criterion, device, args.omniscient_scale,
                                           args.inner_epochs)
        elif i in attacker_list_backdoor:
            client_i = Attacker_Backdoor(i, model, trainData[i], optimizer, criterion, device, args.inner_epochs)
            
            if 'RANDOM' in args.attacks.upper():
                client_i.utils.setRandomTrigger(seed=args.attacks)
                logx.msg(client_i.utils.trigger_position)
                logx.msg(f'Client {i} is using a random backdoor with seed \"{args.attacks}\"')
            if 'CUSTOM' in args.attacks.upper():
                client_i.utils.setTrigger(*args.backdoor_trigger)
                logx.msg(client_i.utils.trigger_position)
                logx.msg(f'Client {i} is using a backdoor with hyperparameter \"{args.backdoor_trigger}\"')
            
        elif i in attacker_list_semanticBackdoor:
            client_i = Attacker_SemanticBackdoor(i, model, trainData[i], optimizer, criterion, device,
                                                 args.inner_epochs)
        else:
            client_i = Client(i, model, trainData[i], optimizer, criterion, device, args.inner_epochs)
        server.attach(client_i)
        
    #######################################
    ##### Start federated training   ######
    #######################################
    loss, accuracy = server.test()
    steps = 0
        # define which metrics to record
    metrics={'loss':None,'accuracy':None,'loss_attack':-1,'attack_success_rate':-1, 'rt_clients':-1,'rt_agg':-1}
    metrics['loss']=loss
    metrics['accuracy']=accuracy
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)
    
    def eval_asr():
        if 'BACKDOOR' in args.attacks.upper():
            if 'SEMANTIC' in args.attacks.upper():
                loss, accuracy, bdata, bpred = server.test_semanticBackdoor()
            else:
                loss, accuracy = server.test_backdoor()
        elif 'LABEL' in args.attacks.upper():
            accuracy = server.test_labelFlipping()
            loss = -1
        else:
            loss, accuracy = -1, -1 # not evaluated
        metrics['loss_attack']=loss
        metrics['attack_success_rate']=accuracy
        logx.metric(phase='val', metrics=metrics, epoch=steps)
        
    eval_asr()
        
    
    for j in range(args.epochs):
        steps = j + 1
        metrics={'loss':None,'accuracy':None,'loss_attack':-1,'attack_success_rate':-1, 'rt_clients':-1,'rt_agg':-1}

        logx.msg('\n\n########EPOCH %d ########' % j)
        logx.msg('###Model distribution###\n')
        server.distribute()
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)
        crt_avg, art = server.train(group) #this return the runtime of the client training and the aggregation
        metrics['rt_clients']=crt_avg
        metrics['rt_agg']=art
        
        
        writer.add_scalar('test/crt_avg', crt_avg, steps)
        writer.add_scalar('test/art', art, steps)

        loss, accuracy = server.test()
        
        metrics['loss']=loss
        metrics['accuracy']=accuracy
        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)

        eval_asr()
    
    writer.close()
