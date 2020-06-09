from __future__ import with_statement # Required in 2.5
import torch
import torch.nn.functional as F
from sub_models import GatedGNN
from sub_models import GatedGCN
from sub_models_di import GatedGNN_di
from sub_models_di import GatedGIN_di
from sub_models import GatedGIN
from sub_models import Lin2_APPNP
from sub_models import GCN_APPNP
from sub_models import StopEarly
from sub_models import GIN
from sub_models import GCN
from sub_models import GAT
from sub_models import ARMA
from sub_models import GMM
from sub_models import GraphGNN
from sub_models import MF
from sub_models import AGNN
from sub_models import SAGE
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
import time
import gc



import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

def random_weight(num):
    weight = torch.rand(num)
    weight = weight + 0.5
    weight = weight.view(weight.shape[0], 1)
    #print(weight)
    return weight


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
def label_train_and_predict(data, sample_mask, val_mask, node_norm,time_control):
    data = data.to('cpu')
    x = data.feature.cpu().numpy()
    train_y = data.y[sample_mask.cpu()].numpy()
    train_x = x[sample_mask.cpu(), :]
    start = time.time()
    try:
        with time_limit(time_control.get_remain_time()):
            y_pred = rf_train(train_x,train_y,x)
            elapsed = (time.time() - start)
            print("label训练时间", elapsed)
            y_pred = torch.tensor(y_pred, dtype=torch.float)
            y_pred = F.log_softmax(y_pred, dim=-1).numpy()
            return y_pred
    except TimeoutException:
        print( "Timed out!")
        return None

def gcn_train(data, aggr, sample_mask, val_mask, node_norm, device, time_control,hidden):
    num_class = int(max(data.y)) + 1
    model = GCN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred

def gated_train(data, aggr, sample_mask, val_mask, node_norm, device, time_control,hidden,conv_aggr):
    num_class = int(max(data.y)) + 1
    model = GatedGCN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,num_layers=2,aggr=conv_aggr)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def graphnn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GraphGNN(features_num=data.x.size()[1],num_class=num_class,hidden=64,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def graphnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GraphGNN_di(features_num=data.x.size()[1],num_class=num_class,hidden=64,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred

def mf_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = MF(features_num=data.x.size()[1],num_class=num_class,hidden=32,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred


def ggnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,hidden,conv_aggr):
    num_class = int(max(data.y)) + 1
    model = GatedGNN_di(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,aggr=conv_aggr)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])
            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,hidden):
    num_class = int(max(data.y)) + 1
    model = GatedGIN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])
            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,hidden):
    num_class = int(max(data.y)) + 1
    model = GatedGIN_di(features_num=data.x.size()[1],num_class=num_class,hidden=hidden)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])
            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred



def mean_train(data, aggr, sample_mask, val_mask, node_norm, device,
                             time_control):
    num_class = int(max(data.y)) + 1
    model = GatedGCN(features_num=data.x.size()[1],num_class=num_class,hidden=64,num_layers=2,aggr='mean',res=True)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred


def gmm_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GMM(features_num=data.x.size()[1],num_class=num_class,hidden=48,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred


def arma_train(data, aggr, sample_mask, val_mask, node_norm, device,
                             time_control):
    num_class = int(max(data.y)) + 1
    model = ARMA(features_num=data.x.size()[1], num_class=num_class)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred


def gin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GIN(features_num=data.x.size()[1],num_class=num_class,hidden=64,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def gin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GIN_di(features_num=data.x.size()[1],num_class=num_class,hidden=48,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred

def gat_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = int(max(data.y)) + 1
    model = GAT(features_num=data.x.size()[1],num_class=num_class,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred


def appnp_lin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control):
    num_class = data.num_class
    model = Lin2_APPNP(features_num=data.x.size()[1],num_class=num_class,hidden=64,K=10,alpha=0.1,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred
def appnp_gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,hidden):
    num_class = data.num_class
    model = GCN_APPNP(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,K=5,alpha=0.2,num_layers=1)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    print(aggr, "开始训练")
    t = StopEarly(2)
    try:
        for epoch in range(1, 501):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = None
            if (node_norm != None):
                loss = F.nll_loss((out[sample_mask]),data.y[sample_mask],reduction='none')
                loss = (loss * node_norm).mean()
            else:
                loss = F.nll_loss((out[sample_mask]), data.y[sample_mask])

            if (epoch % 25 == 0):
                model.eval()
                out = model(data)
                _, pred = out.max(dim=1)
                correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                acc3 = correct / (val_mask.sum().item())
                if (time_control.isTimeToStop() == True):
                    if (epoch <= 150):
                        return None
                    return out.detach().cpu().numpy()
                if (t.isTimeToStop(1 - acc3, model, epoch) == True):
                    end_epoch = epoch
                    print("早停", end_epoch)
                    break
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.detach().cpu().numpy()
            del (model)
            del (out)
            del (optimizer)
            gc.collect()
            torch.cuda.empty_cache()
    except:
        del (model)
        del (optimizer)
        torch.cuda.empty_cache()
        return None
    return pred

def ggnn_test(data, device):
    num_class = int(max(data.y)) + 1
    model = GatedGNN(features_num=data.x.size()[1],num_class=num_class)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
        gc.collect()
    return elapsed
def ggnn_di_test(data, device,hidden,aggr):
    num_class = int(max(data.y)) + 1
    model = GatedGNN_di(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,aggr=aggr)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
        gc.collect()
    return elapsed
def ggin_test(data, device,hidden):
    num_class = int(max(data.y)) + 1
    model = GatedGIN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
        gc.collect()
    return elapsed
def ggin_di_test(data, device,hidden):
    num_class = int(max(data.y)) + 1
    model = GatedGIN_di(features_num=data.x.size()[1],num_class=num_class,hidden=hidden)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
        gc.collect()
    return elapsed
def mean_test(data, device):
    num_class = int(max(data.y)) + 1
    model = GatedGCN(features_num=data.x.size()[1],num_class=num_class,hidden=48)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
        gc.collect()
    return elapsed


def gin_test(data, device):
    num_class = int(max(data.y)) + 1
    model = GIN(features_num=data.x.size()[1],num_class=num_class, hidden=64,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def gin_di_test(data, device):
    num_class = int(max(data.y)) + 1
    model = GIN_di(features_num=data.x.size()[1],num_class=num_class, hidden=64,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def gat_test(data, device):
    num_class = int(max(data.y)) + 1
    model = GAT(features_num=data.x.size()[1],num_class=num_class)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def gcn_test(data, device,hidden=32):
    num_class = int(max(data.y)) + 1
    model = GCN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def appnp_lin_test(data, device):
    num_class = data.num_class
    print(2)
    model = Lin2_APPNP(features_num=data.x.size()[1],num_class=num_class,hidden=128,K=10,alpha=0.1,num_layers=2)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    #start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed
def appnp_gcn_test(data, device,hidden):
    print(1)
    num_class = data.num_class
    model = GCN_APPNP(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,K=5,alpha=0.2,num_layers=1)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    #start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed



def arma_test(data, device):
    num_class = data.num_class
    model = ARMA(features_num=data.x.size()[1], num_class=num_class)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def gmm_test(data, device):
    num_class = data.num_class
    model = GMM(features_num=data.x.size()[1], num_class=num_class, hidden=48)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def graphnn_test(data, device):
    num_class = data.num_class
    model = GraphGNN(features_num=data.x.size()[1],num_class=num_class,hidden=48)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed

def graphnn_di_test(data, device):
    num_class = data.num_class
    model = GraphGNN_di(features_num=data.x.size()[1],num_class=num_class,hidden=48)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed



def mf_test(data, device):
    num_class = data.num_class
    model = MF(features_num=data.x.size()[1], num_class=num_class, hidden=32)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def agnn_test(data, device):
    num_class = data.num_class
    model = AGNN(features_num=data.x.size()[1], num_class=num_class, hidden=16)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed


def sage_test(data, device):
    num_class = data.num_class
    model = SAGE(features_num=data.x.size()[1], num_class=num_class, hidden=64)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed
def gated_test(data, device,hidden,conv_aggr):
    num_class = data.num_class
    model = GatedGCN(features_num=data.x.size()[1],num_class=num_class,hidden=hidden,num_layers=2,aggr=conv_aggr)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=0.0005)
    start = time.clock()
    for epoch in range(1, 2):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = None
        node_norm = None
        if (node_norm != None):
            loss = F.nll_loss((out[data.train_mask]),data.y[data.train_mask],reduction='none')
            loss = (loss * node_norm).mean()
        else:
            loss = F.nll_loss((out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    elapsed = (time.clock() - start)
    with torch.no_grad():
        del (model)
    return elapsed






