from __future__ import print_function

import collections
import pickle

import torch
import torch.nn as nn
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors
import torchtext.experimental.vocab
from torchtext.experimental.datasets.raw.text_classification import RawTextIterableDataset
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

from dataloader import *


# When importing this as a package, several NLP related instance will be initialized, which may take some time.

######### utilies for text tasks ###########
class Tokenizer:
    def __init__(self, tokenize_fn='basic_english', lower=True, max_length=None):
        self.tokenize_fn = torchtext.data.utils.get_tokenizer(tokenize_fn)
        self.lower = lower
        self.max_length = max_length

    def tokenize(self, s):
        tokens = self.tokenize_fn(s)
        if self.lower:
            tokens = [token.lower() for token in tokens]
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
            paddedTokens = ['<pad>'] * self.max_length
            paddedTokens[:len(tokens)] = tokens
            tokens = paddedTokens
        return tokens


def build_vocab_from_data(raw_data, tokenizer, **vocab_kwargs):
    token_freqs = collections.Counter()
    for label, text in raw_data:
        tokens = tokenizer.tokenize(text)
        token_freqs.update(tokens)
    vocab = torchtext.vocab.Vocab(token_freqs, **vocab_kwargs)
    return vocab


def process_raw_data(raw_data, tokenizer, vocab):
    raw_data = [(label, text) for (label, text) in raw_data]
    text_transform = sequential_transforms(tokenizer.tokenize,
                                           vocab_func(vocab),
                                           totensor(dtype=torch.long))
    label_transform = sequential_transforms(totensor(dtype=torch.long))
    transforms = (label_transform, text_transform)
    dataset = TextClassificationDataset(raw_data,
                                        vocab,
                                        transforms)
    return dataset


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        print(labels)
        print(text)
        labels = torch.LongTensor(labels)
        lengths = torch.LongTensor([len(x) for x in text])
        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)
        return labels, text, lengths


def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.GRU):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                r, z, n = p.chunk(3)
                nn.init.xavier_uniform_(r)
                nn.init.xavier_uniform_(z)
                nn.init.xavier_uniform_(n)
            elif 'weight_hh' in n:
                r, z, n = p.chunk(3)
                nn.init.orthogonal_(r)
                nn.init.orthogonal_(z)
                nn.init.orthogonal_(n)
            elif 'bias' in n:
                r, z, n = p.chunk(3)
                nn.init.zeros_(r)
                nn.init.zeros_(z)
                nn.init.zeros_(n)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_pretrained_embedding(initial_embedding, pretrained_vectors, vocab, unk_token):
    pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).detach()
    unk_tokens = []
    pretrained_embedding = pretrained_vectors.lookup_vectors(vocab.get_itos())
    isUnk = torch.sum(pretrained_vectors.lookup_vectors(vocab.get_itos()) != 0, dim=1) == 0
    for idx, token in enumerate(vocab.get_itos()):
        if isUnk[idx]:
            unk_tokens.append(token)
    return pretrained_embedding, unk_tokens


## hard coded global variable ##

max_length = 200
max_size = 25000

tokenizer = Tokenizer(max_length=max_length)

raw_train_data, raw_test_data = torchtext.experimental.datasets.raw.IMDB()
raw_train_data = RawTextIterableDataset(list(raw_train_data))
raw_test_data = RawTextIterableDataset(list(raw_test_data)[::5])


def loadVocab():
    try:
        vocab = torchtext.experimental.vocab.vocab_from_file_object(file_like_object=open("vocab.txt", "r"))
    except Exception as e:
        print(e)
        print("Initialize a vocab")
        vocab = build_vocab_from_data(raw_train_data, tokenizer, max_size=max_size)
        print(*vocab.itos, sep="\n", file=open("vocab.txt", 'w'))
        vocab = torchtext.experimental.vocab.vocab_from_file_object(file_like_object=open("vocab.txt", "r"))

    return vocab


vocab = loadVocab()


######### model for text tasks ###########

class GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, text):
        # text = [seq len, batch size]
        # lengths = [batch size]
        text = text.permute(1, 0)
        embedded = self.embedding(text)

        # embedded = [seq len, batch size, emb dim]
        output, hidden = self.gru(embedded)

        # output = [seq_len, batch size, n directions * hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        prediction = self.fc(hidden.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction


def loadPretrainEmbedding(embedding):
    try:
        embedding = torch.load("fasttext.embedding")
    except Exception as e:
        print(e)
        print("Initialize the embedding")
        fasttext = torchtext.experimental.vectors.FastText(language='en', unk_tensor=None, root='.data',
                                                           validate_file=True)

        unk_token = '<unk>'
        pad_token = '<pad>'
        pad_idx = vocab[pad_token]

        embedding, unk_tokens = get_pretrained_embedding(embedding, fasttext, vocab, unk_token)
        torch.save(embedding, f="fasttext.embedding")
        print("FastText embedding has been saved to \"fasttext.embedding\"")
    return embedding


def Net():
    input_dim = 25002  # hard code for IMDb
    emb_dim = 300
    hid_dim = 256
    output_dim = 2
    pad_token = '<pad>'
    pad_idx = vocab[pad_token]
    model = GRU(input_dim, emb_dim, hid_dim, output_dim, pad_idx)
    model.apply(initialize_parameters)

    pretrained_embedding = loadPretrainEmbedding(model.embedding)

    model.embedding.weight.data.copy_(pretrained_embedding)
    model.embedding.weight.require_grad = False

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
        trainData = process_raw_data(raw_train_data, tokenizer, vocab)
        testData = process_raw_data(raw_test_data, tokenizer, vocab)
        self.dataset = trainData if train == True else testData

        self.targets = [sample[0] for sample in self.dataset.data]  # for later use when partitioning the data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        sample = (self.dataset[i][1], self.dataset[i][0])
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
    y = net(torch.randint(25002, (batch_size, 500)))
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
