
from __future__ import with_statement # Required in 2.5
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

import torch.nn.functional as F
from torch.nn import Linear

import time

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

def count_weight_by_class(data, result, val_mask):
    acc_list = []

    def get_result(pred, y, num_class):
        ls = []
        for i in range(num_class):
            TP = float(pred[(pred == y) & (pred == i)].shape[0]) + 1e-6
            TN = float(pred[(pred != i) & (y != i)].shape[0]) + 1e-6
            FP = float(pred[(pred == i) & (y != i)].shape[0]) + 1e-6
            FN = float(pred[(pred != i) & (y == i)].shape[0]) + 1e-6
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
            ls.append(f1)
        return ls

    def top_k(q, k):
        for i in range(q.shape[1]):
            rank = np.argsort(-q[:, i])
            q[rank, i] = -np.array(range(1, k + 1))
        return q

    for i in range(len(result)):
        acc = get_result(
                np.argmax(result[i], axis=1)[val_mask], data.y[val_mask].numpy(),
                data.num_class)
        acc_list.append(acc)
    acc_list = np.array(acc_list)
    weight = top_k(acc_list, len(acc_list))
    return F.softmax(torch.tensor(weight, dtype=torch.float), dim=0)


def count_weight_by_model(data, result, val_mask):
    acc_list = []

    def get_result(pred, y, num_class):
        acc = len(pred[pred == y]) / pred.shape[0]
        return acc

    for i in range(len(result)):
        acc = get_result(
                np.argmax(result[i], axis=1)[val_mask], data.y[val_mask].numpy(),
                data.num_class)
        acc_list.append(acc)
    acc_list = np.array(acc_list)
    rank = np.argsort(-acc_list)
    print(acc_list)
    acc_list[rank] = -(np.array(range(1, len(result) + 1)))
    return F.softmax(torch.tensor(acc_list, dtype=torch.float),dim=-1).view(-1, 1)


class ans_ensemble_model(torch.nn.Module):
    def __init__(self, num_model=3, num_class=2, weight=None):
        super(ans_ensemble_model, self).__init__()
        self.fuse_weight = torch.nn.Parameter(weight, requires_grad=True)
        self.fuse_weight2 = torch.nn.Parameter(torch.FloatTensor(num_class),requires_grad=True)
        self.fuse_weight2.data.fill_(0)
    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        y = self.fuse_weight
        y = F.softmax(y, dim=0)
        x = torch.mul(x, y)
        x = torch.sum(x, dim=1)
        #x = x + self.fuse_weight2
        return F.log_softmax(x,dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ans_ensemble_model1(torch.nn.Module):
    def __init__(self, num_model=3, num_class=2, weight=None):
        super(ans_ensemble_model1, self).__init__()
        self.fuse_weight = torch.nn.Parameter(weight, requires_grad=True)
        self.fuse_weight2 = torch.nn.Parameter(torch.FloatTensor(num_class),requires_grad=True)
        self.fuse_weight2.data.fill_(0)
        self.num_class = num_class
        #print(self.fuse_weight)
    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        y = self.fuse_weight
        y = F.softmax(y, dim=0)
        y = y.repeat(1, self.num_class)
        x = torch.mul(x, y)
        x = torch.sum(x, dim=1)
        x = x + self.fuse_weight2
        return F.log_softmax(x,dim=-1)

    def __repr__(self):
        return self.__class__.__name__


def get_mean_result(data, result1, result2, val_mask1, val_mask2, num_model):
    data = data.to('cpu')
    valid_results = []
    acc1 = []
    for pred in result1:
        pred = np.argmax(pred, axis=1)
        pred = torch.tensor(pred)
        acc = pred[val_mask1].eq(
                data.y[val_mask1]).sum().item() / val_mask1.sum().item()
        acc1.append(acc)
    print(acc1)
    max_acc1 = np.max(acc1)
    for i in range(len(result1)):
        if acc1[i] < max_acc1 * 0.95:
            print("get_mean_result skip result1@{}".format(i))
            continue
        valid_results.append(result1[i])
    acc2 = []
    for pred in result2:
        pred = np.argmax(pred, axis=1)
        pred = torch.tensor(pred)
        acc = pred[val_mask2].eq(
                data.y[val_mask2]).sum().item() / val_mask2.sum().item()
        acc2.append(acc)
    print(acc2)
    max_acc2 = np.max(acc2)
    for i in range(len(result2)):
        if acc2[i] < max_acc2 * 0.95:
            print("get_mean_result skip result2@{}".format(i))
            continue
        valid_results.append(result2[i])
    pred1 = np.array(valid_results).mean(axis=0)
    return pred1


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
def rf_train(train_x,train_y,test_x):
    rf1 = RandomForestClassifier(n_jobs=4, n_estimators=600, random_state=123)
    rf1.fit(train_x, train_y)
    y_pred = rf1.predict_proba(test_x)
    return y_pred

def get_rf_result(data, result1, result2, val_mask_1, val_mask_2, num_model,time_control):
    if (time_control.isTimeToStop() == True):
            return None
    data = data.to('cpu')
    result1 = np.hstack(result1)
    result2 = np.hstack(result2)
    train_y1 = data.y[val_mask_1].numpy()
    train_y2 = data.y[val_mask_2].numpy()
    train_x1 = result1[val_mask_1]
    train_x2 = result2[val_mask_2]
    y_pred1 = None
    y_pred2 = None
    try:
        with time_limit(time_control.get_remain_time()):
            y_pred1 = rf_train(train_x1,train_y1,result1)
            #elapsed = (time.time() - start)
            #print("随机森林ensemble训练时间", elapsed)
    except TimeoutException:
        print( "Timed out!")
        return None
    
    try:
        with time_limit(time_control.get_remain_time()):
            y_pred2 = rf_train(train_x2,train_y2,result2)
            #elapsed = (time.time() - start)
            #print("随机森林ensemble训练时间", elapsed)
    except TimeoutException:
        print( "Timed out!")
        return None
    
    #rf1 = RandomForestClassifier(n_jobs=4, n_estimators=500)
    #rf1.fit(train_x1, train_y1)
    #y_pred1 = rf1.predict_proba(result1)

    #rf2 = RandomForestClassifier(n_jobs=4, n_estimators=500)
    #rf2.fit(train_x2, train_y2)
    #y_pred2 = rf2.predict_proba(result2)
    
    y_pred1 = torch.tensor(y_pred1)
    y_pred2 = torch.tensor(y_pred2)
    y_pred1 = F.log_softmax(y_pred1, dim=-1)
    y_pred2 = F.log_softmax(y_pred2, dim=-1)
    y_pred = (y_pred1 + y_pred2)/2
    return y_pred.numpy()


def get_linear_reslut(data, result1, result2, val_mask_1, val_mask_2,
                                            num_model, device,clock):
    #print(result1[:,1,:])
    if (clock.isTimeToStop() == True):
            return None
    data = data.to('cpu')
    weight1 = count_weight_by_class(data, result1, val_mask_1)
    weight2 = count_weight_by_class(data, result2, val_mask_2)

    result1 = torch.tensor(result1, dtype=torch.float)
    result1 = result1.permute(1, 0, 2)
    #print(result1.shape)
    result2 = torch.tensor(result2, dtype=torch.float)
    result2 = result2.permute(1, 0, 2)
    train_x1 = result1[val_mask_1]
    train_x2 = result2[val_mask_2]
    train_y1 = data.y[val_mask_1]
    train_y2 = data.y[val_mask_2]
    test_x1 = result1[data.test_mask]
    test_x2 = result2[data.test_mask]
    model1 = ans_ensemble_model(num_model, int(max(data.y)) + 1, weight1)
    result1 = result1.to(device)
    result2 = result2.to(device)
    model2 = ans_ensemble_model(num_model, int(max(data.y)) + 1, weight2)
    train_x1 = train_x1.to(device)
    model1 = model1.to(device)
    train_y1 = train_y1.to(device)
    test_x1 = test_x1.to(device)
    optimizer = torch.optim.LBFGS(model1.parameters(),lr=0.5)
    min_loss = float('inf')
    for epoch in range(1, 30):
        def fun():
            model1.train()
            optimizer.zero_grad()
            out = model1(train_x1)
            loss = F.nll_loss(out, train_y1, reduction='mean')
            loss.backward()
            return loss
        optimizer.step(fun)
        if (clock.isTimeToStop() == True):
            return None
    model1.eval()
    out1 = model1(result1)
    train_x2 = train_x2.to(device)
    model2 = model2.to(device)
    train_y2 = train_y2.to(device)
    test_x2 = test_x2.to(device)
    optimizer = torch.optim.LBFGS(model2.parameters(),lr=0.5)
    min_loss = float('inf')
    for epoch in range(1, 30):
        def fun():
            model2.train()
            optimizer.zero_grad()
            out = model2(train_x2)
            loss = F.nll_loss(out, train_y2, reduction='mean')
            loss.backward()
            return loss
        if (clock.isTimeToStop() == True):
            return None
        optimizer.step(fun)
    model2.eval()
    out2 = model2(result2)
    out = (out1 + out2) / 2
    out = out.detach().cpu().numpy()
    return out


def get_linear_reslut1(data, result1, result2, val_mask_1, val_mask_2, num_model, device,clock):
    #print(result1[:,1,:])
    if (clock.isTimeToStop() == True):
            return None
    data = data.to('cpu')
    weight1 = count_weight_by_model(data, result1, val_mask_1)
    weight2 = count_weight_by_model(data, result2, val_mask_2)
    result1 = torch.tensor(result1, dtype=torch.float)
    result1 = result1.permute(1, 0, 2)
    #print(result1.shape)
    result2 = torch.tensor(result2, dtype=torch.float)
    result2 = result2.permute(1, 0, 2)
    train_x1 = result1[val_mask_1]
    train_x2 = result2[val_mask_2]
    train_y1 = data.y[val_mask_1]
    train_y2 = data.y[val_mask_2]
    test_x1 = result1[data.test_mask]
    test_x2 = result2[data.test_mask]
    model1 = ans_ensemble_model1(num_model, int(max(data.y)) + 1, weight1)
    result1 = result1.to(device)
    result2 = result2.to(device)
    model2 = ans_ensemble_model1(num_model, int(max(data.y)) + 1, weight2)
    train_x1 = train_x1.to(device)
    model1 = model1.to(device)
    train_y1 = train_y1.to(device)
    test_x1 = test_x1.to(device)
    optimizer = torch.optim.LBFGS(model1.parameters(),lr=0.5)
    min_loss = float('inf')
    for epoch in range(1, 30):
        def fun():
            model1.train()
            optimizer.zero_grad()
            out = model1(train_x1)
            loss = F.nll_loss(out, train_y1, reduction='mean')
            loss.backward()
            return loss
        optimizer.step(fun)
        if (clock.isTimeToStop() == True):
            return None
    model1.eval()
    out1 = model1(result1)
    train_x2 = train_x2.to(device)
    model2 = model2.to(device)
    train_y2 = train_y2.to(device)
    test_x2 = test_x2.to(device)
    optimizer = torch.optim.LBFGS(model2.parameters(),lr=0.5)
    min_loss = float('inf')
    for epoch in range(1, 30):
        def fun():
            model2.train()
            optimizer.zero_grad()
            out = model2(train_x2)
            loss = F.nll_loss(out, train_y2, reduction='mean')
            loss.backward()
            return loss
        optimizer.step(fun)
        if (clock.isTimeToStop() == True):
            return None
    model2.eval()
    out2 = model2(result2)
    out = (out1 + out2) / 2
    out = out.detach().cpu().numpy()
    return out


def get_ensemble_result(data, result1, result2, val_mask1, val_mask2,num_model, device,clock):
    pred1 = get_mean_result(data, result1, result2, val_mask1, val_mask2,num_model)
    pred3 = get_linear_reslut(data, result1, result2, val_mask1, val_mask2,num_model, device,clock)
    pred4 = get_linear_reslut1(data, result1, result2, val_mask1, val_mask2,num_model, device,clock)
    pred2 = get_rf_result(data, result1, result2, val_mask1, val_mask2,num_model,clock)
    pred1 = (pred1 - pred1.mean()) / pred1.std()
    if(pred2 is not None):
        pred2 = (pred2 - pred2.mean()) / pred2.std()
    else:
        pred2 = 0
    if(pred3 is not None):
        pred3 = (pred3 - pred3.mean()) / pred3.std()
    else:
        pred3 = 0
    if(pred4 is not None):
        pred4 = (pred4 - pred4.mean()) / pred4.std()
    else:
        pred4 = 0
    pred = pred1 + pred2 + pred3 + pred4
    print("牛顿法")
    pred = np.argmax(pred, axis=1)
    #pred1 = np.argmax(pred1, axis=1)
    ##pred2 = np.argmax(pred2, axis=1)
    #pred3 = np.argmax(pred3, axis=1)
    #pred4 = np.argmax(pred4, axis=1)
    #pred = torch.tensor(pred)
    #pred1 = torch.tensor(pred1)
    #pred2 = torch.tensor(pred2)
    #pred3 = torch.tensor(pred3)
    #pred4 = torch.tensor(pred4)
    return pred, None, None, None, None


def evalue_result(pred, y, data):
    correct = float(pred[data.train_mask].eq(
            data.y[data.train_mask]).sum().item())
    acc2 = correct / (data.train_mask.sum().item() + 0)
    print('*gcn 测试集Accuracy: {:.4f}'.format(acc2))
    correct = float(pred[data.test_mask].eq(y[data.test_mask]).sum().item())
    acc2 = correct / (data.test_mask.sum().item() + 0)
    print('*gcn 测试集Accuracy: {:.4f}'.format(acc2))
    print()





