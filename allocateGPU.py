import os
import numpy as np
import subprocess
import torch


def allocate_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))



    print('Number of gpu avaliable:\t%d'% torch.cuda.device_count())
    currentGPU=torch.cuda.current_device()
    assert type(currentGPU)==int, 'GPU not available' 
    print('Current GPU:\t%d'%torch.cuda.current_device())
    print('GPU name: \t%s'%torch.cuda.get_device_name(currentGPU))
