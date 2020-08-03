#!/usr/bin/env python
# coding: utf-8

# In[ ]:


root="./logs/netRes_experiments/cifar_dirichlet/"


# In[ ]:


import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    
    
    
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath) if 'csv' not in dname]
    tags = summary_iterators[0].Tags()['scalars']
    summary_iterators=[it for it in summary_iterators if it.Tags()['scalars'] == tags]
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps

dict1={}
def to_csv(dpath):
    dirs = os.listdir(dpath)
    print(dirs)
    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())

    np_values = np.array(values)

    for index, tag in enumerate(tags):
    
        df = pd.DataFrame(np_values[index], index=steps[:np_values[index].shape[0]], columns=dirs)
        file_name = tag.replace("/", "_") + '.csv'
        folder_path = os.path.join(dpath, 'csv')
        dict1[os.path.join(folder_path, file_name)]=df


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


# if __name__ == '__main__':
#     root="./logs/netRes_experiments/cifar_dirichlet/"
    

paths=sorted([os.path.join(root,dname) for dname in os.listdir(root)])
paths=[i for i in paths if not 'attack' in i.lower()]
#     paths=[i for i in path if 'csv' not in i]
for path in paths:
    print(path)
    try:
        to_csv(path)
    except:
        continue


# In[ ]:


len(dict1.keys())


# In[ ]:


# import os
# allfiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("./logs")) for f in fn]


# In[ ]:


allfiles=sorted(dict1.keys())


# In[ ]:


allcsv=[i for i in allfiles if 'test_accuracy.csv' in i]
len(allcsv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def getSortedColumnsDF(DF):
    order=['fedavg','median','baseline','aggNet_noMedian','aggNetResidual','aggNetBlocks','aggNetBlocksMultiple']
    order=dict(zip(order,range(len(order))))
    DF=DF.reindex([v for i,v in sorted(zip([order[i] for i in DF.columns],DF.columns))], axis=1)
    return DF


# In[ ]:


allcsv=[i for i in allfiles if 'test_accuracy.csv' in i]


# In[ ]:


import pandas as pd
j=0
concatAll=pd.DataFrame()
for i in sorted(allcsv):
    DF=dict1[i]
    DF=getSortedColumnsDF(DF)
    title=i.split('/')[4]
    fig=DF.plot(title=title)
    fig.set(xlabel='commnication rounds', ylabel='Accuracy')
    fig.get_figure().savefig(f"./results/acc/{title}.svg")
    concatAll[title]=DF.iloc[-1]
concatAll.to_csv(f"./results/acc/summary.csv")


# In[ ]:


concatAll


# In[ ]:


allcsv=[i for i in allfiles if 'test_backdoor_success_rate.csv' in i]


# In[ ]:


import pandas as pd
j=0
concatAll=pd.DataFrame()
for i in sorted(allcsv):
    DF=dict1[i]
    DF=getSortedColumnsDF(DF)
    title=i.split('/')[4]
    fig=DF.plot(title=title)
    fig.set(xlabel='commnication rounds', ylabel='Attack success rate')
    fig.get_figure().savefig(f"./results/asr/{title}.svg")
    concatAll[title]=DF.iloc[-1]
concatAll.to_csv(f"./results/asr/summary.csv")


# In[ ]:





# In[ ]:


# import matplotlib.pyplot as plt

# def plot(xdata,ydata,xlabel,ylabel,title,savepath):
#     _,ax=plt.subplots(figsize=(6,6))
#     plt.scatter(xdata,ydata)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     sup=max(ax.get_xlim()[1],ax.get_ylim()[1])
#     inf=min(ax.get_xlim()[0],ax.get_ylim()[0])
#     plt.plot([inf, sup], [inf, sup], ls = "--",c =".3")
#     plt.savefig(savepath)
# plot(a,b,"x","y","title","./results/test.svg")


# In[ ]:


# a=[1,2,3]
# # b=[4,5,6]


# In[ ]:


# import pandas as pd
# a=list(np.random.rand(20))
# b=list(np.random.rand(20))
# c=list(np.random.rand(20))
# di={'a':a,'b':b,'c':c}


# In[ ]:


# df=pd.DataFrame.from_dict({'a':a})


# In[ ]:


# df.index=a


# In[ ]:





# In[ ]:


# files=os.listdir('./results/aggResult/training')


# In[ ]:





# In[ ]:


# l1=[]
# for f in files:
#     key=f.split('_')[2]
#     item=pd.read_csv('./results/aggResult/training/'+f)
#     l1.append((key,item))
    


# In[ ]:


# [i[0] for i in l1]


# In[ ]:


# def getSorted(L):
#     order=['random','fedavg','median','baseline','aggNet','aggNetRes','aggNetBlock','aggNetBlockMultiple']
#     order=dict(zip(order,range(len(order))))
#     sortedL=[v for i,v in sorted(zip([order[k] for k,i in L],L))]
#     return sortedL


# In[ ]:





# In[ ]:


# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
# for (key,item) in getSorted(l1):
#     df=item
#     x=df.Step
#     y=df.Value
#     start_index=x.where(x==0).last_valid_index()
#     plt.plot(x[start_index:start_index+80],y[start_index:start_index+80],label=key)
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('Average BCE loss')
# plt.savefig('./results/aggResult/convergence-BCE.svg')


# In[ ]:





# In[ ]:





# In[ ]:




