import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def LF_success_rate(pred, gt, attacked_classes, target_class):
    '''
    pred: bsz by 1, indices of class 
    gt: bsz by 1, indices of class 
    attacked_classes: list of int
    target_class: int
    
    Compute the fraction of attacked_classes' instance being classified as target_class
    
    return
        success_rate : fraction of successully attacked instance
        num_attacked_instance : number of attacked instance
    '''
    # A perfect attack expects the attacked_classes' instance to be predicted as target_class
    # We only care about the prediction on attacked_classes' instance. 
    # So, treat other classes' instance to be -1, which will never appear in pred.
    assert -1 not in pred, "pred should contain values from 0 to C-1 "
    attackers_target = torch.tensor(list(map(lambda x: target_class if (x in attacked_classes) else -1, gt)))
    
    num_attacked_instance = (attackers_target!=-1).sum().item()
    num_success_attack = pred.eq(attackers_target.view_as(pred)).sum().item()
    
    success_rate = num_success_attack/(num_attacked_instance+1e-16)
    return success_rate, num_attacked_instance