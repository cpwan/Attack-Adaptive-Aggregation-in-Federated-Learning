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

def getAvaliableGPU():
    allowed_GPU = np.array([5,6])  # only the gpu in the list can be used
    MEM=[int(x.split()[2]) for x in np.array(
        subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                         stdout=subprocess.PIPE).stdout.readlines())[allowed_GPU]]
    i=np.argmax(MEM)
    gpu_avaliable=MEM[i]
    gpu_to_use = allowed_GPU[i]      
    return gpu_to_use, gpu_avaliable

def allocate_gpu():
    gpu_to_use, gpu_avaliable = getAvaliableGPU()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)

    torch.set_num_threads(lim)

    print('Number of gpu avaliable:\t%d' % torch.cuda.device_count())
    currentGPU = torch.cuda.current_device()
    assert type(currentGPU) == int, 'GPU not available'
    print('Current GPU:\t%d ; %d (virtual)' % (gpu_to_use,torch.cuda.current_device()))
    print('GPU name: \t%s' % torch.cuda.get_device_name(currentGPU))
    print('GPU Memory avaliable:\t%d' % gpu_avaliable)
    
if __name__ == '__main__':
    '''
    If called directly, this module checks whether there are sufficient GPU memory to be allocated.
    
    Usage:
    `python allocateGPU.py $m$`
    where $m$ is the GPU memory to be allocated.
    What it does:
    If there is at least one GPU that has avaliable memory more than $m$,
    then this program returns 0. Otherwise, it returns 1.
    This behaviour is tailored for GNU parallel for running multiple jobs on GPU. 
    Specifically, doing `parallel --limit "python allocateGPU.py $m$ < somejobs.sh"`
    will run the jobs in somejobs.sh only if there are $m$ GPU avaliable.      
    
    '''
    import sys
    if (len(sys.argv)!=2):
        print("    Usage:\
    `python allocateGPU.py $m$`\
    where $m$ is the GPU memory to be allocated.")
        exit()
        
    m=int(sys.argv[1])
    
    gpu_to_use, gpu_avaliable = getAvaliableGPU()
#     print('Current GPU:\t%d | GPU Memory avaliable:\t%d' % (gpu_to_use,gpu_avaliable))
    
    if m < gpu_avaliable:
        sys.exit(0)
    else:
        sys.exit(1)
    
    