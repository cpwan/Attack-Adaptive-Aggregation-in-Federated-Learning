import os

lim = 2
limit = str(lim)
os.environ["OMP_NUM_THREADS"] = limit  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = limit  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = limit  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = limit  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = limit  # export NUMEXPR_NUM_THREADS=1

import subprocess
import numpy as np
import torch


def allocate_gpu():
    allowedGPU = np.array([2,3])  # only the gpu in the list can be used
    gpu_to_use = allowedGPU[np.argmax([int(x.split()[2]) for x in np.array(
        subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                         stdout=subprocess.PIPE).stdout.readlines())[allowedGPU]])]

    #     gpu_to_use= np.argmax([int(x.split()[2]) for x in
    #     subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free",
    #     shell=True, stdout=subprocess.PIPE).stdout.readlines()])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)

    torch.set_num_threads(lim)

    print('Number of gpu avaliable:\t%d' % torch.cuda.device_count())
    currentGPU = torch.cuda.current_device()
    assert type(currentGPU) == int, 'GPU not available'
    print('Current GPU:\t%d' % torch.cuda.current_device())
    print('GPU name: \t%s' % torch.cuda.get_device_name(currentGPU))
