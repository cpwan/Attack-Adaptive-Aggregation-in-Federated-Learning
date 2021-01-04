#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import math
import random
import time
from functools import reduce

import numpy as np
import torch

from utils import utils

'''
Attack-Resistant Federated Learning with Residual-based Reweighting

retrieved from https://github.com/fushuhao6/Attack-Resistant-Federated-Learning/blob/master/FedAvg/averaging.py

Reference:
Fu, Shuhao, et al. "Attack-Resistant Federated Learning with Residual-based Reweighting." arXiv preprint arXiv:1912.11464 (2019).

'''

eps = np.finfo(float).eps

useGPU=True


def average_weights(w):
    cur_time = time.time()
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    print('model aggregation "average" took {}s'.format(time.time() - cur_time))
    return w_avg


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights


def reweight_algorithm_restricted(y, LAMBDA, thresh):
    if useGPU:
        y=y.cuda()
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    if useGPU:
        restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
            torch.cuda.FloatTensor)
    else:
        restricted_y = y * (reweight >= thresh).type(torch.FloatTensor) + line_y * (reweight < thresh).type(
            torch.FloatTensor)
    return reweight_regulized, restricted_y


def gaussian_reweight_algorithm_restricted(y, sig, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)

    reweight = gaussian_zero_mean(e, sig=sig)
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation
    if useGPU:
        restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
            torch.cuda.FloatTensor)
    else:
        restricted_y = y * (reweight >= thresh).type(torch.FloatTensor) + line_y * (reweight < thresh).type(
            torch.FloatTensor)
    return reweight_regulized, restricted_y


def theilsen_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = theilsen(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation
    if useGPU:
        restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
            torch.cuda.FloatTensor)
    else:
        restricted_y = y * (reweight >= thresh).type(torch.FloatTensor) + line_y * (reweight < thresh).type(
            torch.FloatTensor)
    return reweight_regulized, restricted_y


def median_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    y_median = median_opt(y).unsqueeze(1).repeat(1, num_models)
    residual = y - y_median
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation
    if useGPU:
        restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + y_median * (reweight < thresh).type(
            torch.cuda.FloatTensor)
    else:
        restricted_y = y * (reweight >= thresh).type(torch.FloatTensor) + y_median * (reweight < thresh).type(
            torch.FloatTensor)
    return reweight_regulized, restricted_y


def simple_reweight(y, LAMBDA, thresh, alpha):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    # e = 1 / (residual.abs() + eps)
    # e_max = e.max(dim=-1)[0].unsqueeze(1).repeat(1, num_models)
    # reweight = e / e_max
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = 1 / e * torch.max(-K, torch.min(K, e))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std

    # sorted idx (remove alpha)
    sort_ids = e.abs().sort()[1].sort()[1]
    # remove_ids = sort_ids >= int((1 - alpha) * num_models)
    remove_ids = [i for i in sort_ids if i.item() >= int((1 - alpha) * num_models)]
    remove_ids = remove_ids * (reweight < thresh)
    
    if useGPU:
        keep_ids = (1 - remove_ids).type(torch.cuda.FloatTensor)
        remove_ids = remove_ids.type(torch.cuda.FloatTensor)
    else:
        keep_ids = (1 - remove_ids).type(torch.FloatTensor)
        remove_ids = remove_ids.type(torch.FloatTensor)
    
    restricted_y = y * keep_ids + line_y * remove_ids
    reweight_regulized = reweight_regulized * keep_ids
    return reweight_regulized, restricted_y


def is_valid_model(w):
    if isinstance(w, list):
        w_keys = list(range(len(w)))
    else:
        w_keys = w.keys()
    for k in w_keys:
        params = w[k]
        if torch.isnan(params).any():
            return False
        if torch.isinf(params).any():
            return False
    return True


def get_valid_models(w_locals):
    w, invalid_model_idx = [], []
    for i in range(len(w_locals)):
        if is_valid_model(w_locals[i]):
            w.append(w_locals[i])
        else:
            invalid_model_idx.append(i)
    return w, invalid_model_idx


def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)
    if useGPU:
        reweight_sum=reweight_sum.cuda()
        
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum.cpu())

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def IRLS_other_split_restricted(w_locals, LAMBDA=2, thresh=0.1, mode='median'):
    if mode == 'median':
        reweight_algorithm = median_reweight_algorithm_restricted
    elif mode == 'theilsen':
        reweight_algorithm = theilsen_reweight_algorithm_restricted
    elif mode == 'gaussian':
        reweight_algorithm = gaussian_reweight_algorithm_restricted  # in gaussian reweight algorithm, lambda is sigma

    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm(transposed_y_list, LAMBDA, thresh)
            print(reweight.sum(dim=0))
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm(y, LAMBDA, thresh)
                print(reweight.sum(dim=0))
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


def simple_IRLS(w, LAMBDA=2, thresh=0.03, alpha=1 / 11.0):
    SHARD_SIZE = 50000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = simple_reweight(transposed_y_list, LAMBDA, thresh, alpha)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = simple_reweight(y, LAMBDA, thresh, alpha)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum  )
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, (reweight / reweight.max()).to(torch.device("cpu"))


def random_select(size, thresh=0.5):
    assert thresh < 1.0
    a = []
    while len(a) < 3:
        for i in range(size):
            if random.uniform(0, 1) > thresh:
                a.append(i)
    return a


def theilsen(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yy = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyj = yy
    yyi = yyj.transpose(-1, -2)
    
    if useGPU:
        xx = torch.cuda.FloatTensor(range(num_models))
    else:
        xx = torch.FloatTensor(range(num_models))
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    if useGPU:
        diag = torch.cuda.FloatTensor([float('Inf')] * num_models)
    else:
        diag = torch.FloatTensor([float('Inf')] * num_models)
    inf_lower = torch.tril(diag.repeat(num_models, 1), diagonal=0).repeat(total_num, 1, 1)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + inf_lower
    slopes, _ = torch.flatten(slopes, 1, 2).sort()
    raw_slopes = slopes[:, :int(num_models * (num_models - 1) / 2)]
    slopes = median_opt(raw_slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    if useGPU:
        xx_median = torch.cuda.FloatTensor(xx_median)
    else:
        xx_median = torch.FloatTensor(xx_median)
    intercepts = yy_median - slopes * xx_median
    return slopes, intercepts


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


def gaussian_zero_mean(x, sig=1):
    return torch.exp(- x * x / (2 * sig * sig))


class Net():
    def __init__(self, LAMBDA=2, thresh=0.1):
        self.LAMBDA = LAMBDA
        self.thresh = thresh

    def main(self, deltas: list):
        '''
        deltas: a list of state_dicts

        return 
            Delta: robustly aggregated state_dict

        '''
        Delta = deltas[0]
        param_trainable = utils.getFloatSubModules(deltas[0])

        param_nontrainable = [param for param in deltas[0].keys() if param not in param_trainable]
        for param in param_nontrainable:
            for i in range(len(deltas)):
                del deltas[i][param]
        #         print(utils.getFloatSubModules(deltas[0]))
        
        rDelta, w = IRLS_aggregation_split_restricted(deltas, self.LAMBDA, self.thresh)
        Delta.update(rDelta)
        print(w)
        return Delta
