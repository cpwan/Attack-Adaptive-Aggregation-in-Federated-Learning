from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torchtext
from torchtext.experimental.vocab import vocab_from_file_object
from transformers import MobileBertTokenizer, MobileBertModel

from dataloader import *


class BERTGRUSentiment(nn.Module):
    '''
    Model retrieved from
    https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb

    '''

    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        output = output.squeeze(1)
        return output


def Net():
    bert = MobileBertModel.from_pretrained('google/mobilebert-uncased')

    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    model = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    return model


##################################################
#                                                #
#                                                #
#     Helper class for generating dataset        #
#                                                #
#                                                #
##################################################
class IMDB(torch.utils.data.Dataset):
    def __init__(self, train=True):
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

        def customTokenizer(sentence: str) -> list:
            # keeps only the first 510 character
            max_len = 510
            tokens = tokenizer.tokenize(sentence)
            tokens = ['<cls>'] + \
                     tokens[:max_len] + \
                     ['<sep>'] + \
                     ['<pad>'] * max(0, max_len - len(tokens))
            return tokens

        def getVocab(tokenizer):
            tokenizer.save_pretrained("./")

            token_old = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            token_new = ['<pad>', '<unk>', '<cls>', '<sep>', '<mask>']

            fin = open("vocab.txt", "rt")
            data = fin.read()
            for old, new in zip(token_old, token_new):
                data = data.replace(old, new)
            fin.close()

            fin = open("vocab_adapted.txt", "wt")
            fin.write(data)
            fin.close()

            f = open('vocab_adapted.txt', 'r')
            v = vocab_from_file_object(f)
            return v

        trainData, testData = torchtext.experimental.datasets.IMDB(vocab=getVocab(tokenizer), tokenizer=customTokenizer,
                                                                   data_select=('train', 'test'))
        self.dataset = trainData if train == True else testData

        self.targets = [sample[0] for sample in self.dataset.data]  # for later use when partitioning the data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        sample = (self.dataset[i][1], self.dataset[i][0].float())
        return sample


def getDataset(train=True):
    dataset = IMDB(train)

    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset(train=True)
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_data = getDataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    return test_loader


if __name__ == '__main__':
    print("#Initialize a network")
    net = Net()
    batch_size = 10
    y = net(torch.randint(30522, (batch_size, 512)))
    print(f"Output shape of the network with batchsize {batch_size}:\t {y.size()}")

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0]
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
