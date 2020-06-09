import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from sklearn.model_selection import train_test_split
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree, add_remaining_self_loops, remove_self_loops, is_undirected
from torch_geometric.utils import to_undirected, sort_edge_index
import gc
from sub_models import Label_Extract
from sub_models import Feature_Extract
import model_train as mt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import os


def check_label(data, device):
    x = data.x_label.numpy()
    y = np.ones(data.x.shape[0], dtype=int)
    y[data.train_mask] = 1
    y[data.test_mask] = 0
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=0)
    rf1 = RandomForestClassifier(n_jobs=4, n_estimators=400, random_state=123)
    print('Predicting...')
    rf1.fit(X_train, y_train)
    y_prob = rf1.predict(X_test)
    acc = len(y_test[y_test == y_prob]) / len(y_test)
    print("验证集训练集预测值", acc)
    if (acc < 0.7):
        return True
    else:
        return False


def get_feature(data, label_is_work, is_direct, has_feature, edge_expand,device):
    if (has_feature):
        newX = data.x.detach().clone()
        if (data.x.shape[1] > 128):
            pca = PCA(n_components=128)
            newX = pca.fit_transform(data.x.detach().numpy())
            newX = torch.tensor(newX, dtype=torch.float, device=device)

        model = Feature_Extract()
        data = data.to(device)
        model = model.to(device)
        feature = model(newX, data, has_feature)
        feature = feature.detach()
        del (model)
        data.feature = feature
    else:
        newX = torch.ones((data.x.shape[0], 2), dtype=torch.float, device=device)
        model = Feature_Extract()
        data = data.to(device)
        model = model.to(device)
        feature = model(newX, data, has_feature)
        feature = feature.detach()
        del (model)
        data.feature = feature
        data = data.to('cpu')
        motifs = count_motifs(data, is_direct)
        if (motifs is not None):
            print("增加motifs")
            data.feature = torch.cat(
                            (data.feature, torch.tensor(motifs, dtype=torch.float)), dim=1)
            del (motifs)
    if (label_is_work):
        print("特征提取 label and feature")
        data.feature = torch.cat((data.feature, data.x_label), dim=1)
    else:
        print("特征提取 feature")
    del (data.label)
    del (data.x_label)
    gc.collect()
    return data


def count_motifs(data, is_direct):
    if (data.edge_index.shape[1] > 500000):
            return None
    edge_index = data.edge_index.clone()
    if (is_direct == True):
            edge_index = to_undirected(data.edge_index)
    edge_index, _ = sort_edge_index(edge_index)
    edge_index = edge_index.numpy()
    k = pd.DataFrame(edge_index.T).reset_index(drop=True)
    k.rename(columns={0: data.x.shape[0], 1: edge_index.shape[1]}, inplace=True)
    try:
        name = 'graph' + str(data.x.shape[0]) + str(data.x.shape[1]) + str(
                        data.edge_index.shape[0]) + str(data.edge_index.shape[1])
        name1 = name + '.in'
        name2 = name + '.out'
        path = './' + name + '.in'
        k.to_csv(path, sep=' ', index=False)
        os.system('./orca 4 ' + name1 + ' ' + name2)
        k = pd.read_csv('./' + name2, sep=' ', header=None)
    except:
        return None
    return k.to_numpy()
def add_direct_edge(data):
    data = data.to('cpu')
    edge_index = data.edge_index.detach().clone()
    edge_index, _ = sort_edge_index(edge_index)
    edge_index = edge_index.numpy()
    k = pd.DataFrame(edge_index.T).reset_index(drop=True)
    k.rename(columns={0: data.x.shape[0], 1: edge_index.shape[1]}, inplace=True)
    k['1'] = data.edge_weight.detach().numpy()
    #print(k)
    #return
    try:
        name = 'edge' + str(data.x.shape[0]) + str(data.x.shape[1]) + str(
                        data.edge_index.shape[0]) + str(data.edge_index.shape[1])
        name1 = name + '.in'
        name2 = name + '.out'
        path = './' + name + '.in'
        k.to_csv(path, sep=' ', index=False)
        os.system('./edge 4 ' + name1 + ' ' + name2)
        k = pd.read_csv('./' + name2, sep=' ', header=None)
    except:
        return None,None
    k = k.to_numpy()
    return k[:,0:2],k[:,2]

def gnn_test_time(data, aggr, device):
    if (aggr == 'ggnn'):
        return mt.ggnn_test(data, device)
    elif (aggr == 'ggnn_di'):
        return mt.ggnn_di_test(data, device)
    elif (aggr == 'gated_add_di'):
        return mt.ggnn_di_test(data, device,48,'add')
    elif (aggr == 'gated_mean_di'):
        return mt.ggnn_di_test(data, device,48,'mean')
    elif (aggr == 'appnp_gcn'):
        return mt.appnp_gcn_test(data, device)
    elif (aggr == 'appnp_gcn128'):
        return mt.appnp_gcn_test(data, device,128)
    elif (aggr == 'appnp_gcn48'):
        return mt.appnp_gcn_test(data, device,48)
    elif (aggr == 'appnp_gcn96'):
        return mt.appnp_gcn_test(data, device,96)
    elif (aggr == 'arma'):
        return mt.arma_test(data, device)
    elif (aggr == 'gmm'):
        return mt.gmm_test(data, device)
    elif (aggr == 'mean'):
        return mt.mean_test(data, device)
    elif (aggr == 'graphnn'):
        return mt.graphnn_test(data, device)
    elif (aggr == 'graphnn_di'):
        return mt.graphnn_di_test(data, device)
    elif (aggr == 'mf'):
        return mt.mf_test(data, device)
    elif (aggr == 'agnn'):
        return mt.agnn_test(data, device)
    elif (aggr == 'sage'):
        return mt.sage_test(data, device)
    elif (aggr == 'ggin'):
        return mt.ggin_test(data, device)
    elif (aggr == 'gat'):
        return mt.gat_test(data, device)
    elif (aggr == 'ggin32'):
        return mt.ggin_test(data, device, 32)
    elif (aggr == 'ggin24'):
        return mt.ggin_test(data, device, 24)
    elif (aggr == 'ggin48'):
        return mt.ggin_test(data, device, 48)
    elif (aggr == 'ggin64'):
        return mt.ggin_test(data, device, 64)
    elif (aggr == 'ggin96'):
        return mt.ggin_test(data, device,96)
    elif (aggr == 'ggin_di'):
        return mt.ggin_di_test(data, device,12)
    
    elif (aggr == 'ggin_di24'):
        return mt.ggin_di_test(data, device,24)
    
    elif (aggr == 'ggin_di32'):
        return mt.ggin_di_test(data, device,32)
    
    elif (aggr == 'ggin_di48'):
        return mt.ggin_di_test(data, device,48)
        
    elif (aggr == 'ggin_di56'):
        return mt.ggin_di_test(data, device,56)
    elif (aggr == 'ggin_di64'):
        return mt.ggin_di_test(data, device,64)
    
    elif (aggr == 'gcn16'):
        return mt.gcn_test(data, device,16)
    elif (aggr == 'gcn156'):
        return mt.gcn_test(data, device,156)
    elif (aggr == 'gcn64'):
        return mt.gcn_test(data, device,64)
    elif (aggr == 'gcn128'):
        return mt.gcn_test(data, device,128)
    elif (aggr == 'gcn48'):
        return mt.gcn_test(data, device,48)
    elif (aggr == 'gated_mean32'):
        return mt.gated_test(data, device,32,'mean')
    elif (aggr == 'gated_mean48'):
        return mt.gated_test(data, device,48,'mean')
    elif (aggr == 'gated_mean56'):
        return mt.gated_test(data, device,56,'mean')
    elif (aggr == 'gated_add56'):
        return mt.gated_test(data, device,56,'add')
    elif (aggr == 'gated_add48'):
        return mt.gated_test(data, device,48,'add')
    elif (aggr == 'gated_add32'):
        return mt.gated_test(data, device,32,'add')
    else:
        print(aggr,"not exist")
def gnn_train_and_predict(data,aggr,sample_mask,val_mask,node_norm,device,time_control=None):
    if (aggr == 'ggnn'):
        return mt.ggnn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'ggnn_di'):
        return mt.ggnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'ggnn_add_di'):
        return mt.ggnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48,'add')
    elif (aggr == 'ggnn_mean_di'):
        return mt.ggnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48,'mean')
    elif (aggr == 'gin'):
        return mt.gin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'gin_di'):
        return mt.gin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'gat'):
        return mt.gat_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'gcn'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,12)
    elif (aggr == 'gcn16'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,16)
    elif (aggr == 'gcn32'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,32)
    elif (aggr == 'gcn64'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,64)
    elif (aggr == 'gcn128'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,128)
    elif (aggr == 'gcn48'):
        return mt.gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48)
    elif (aggr == 'appnp_lin'):
        return mt.appnp_lin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'appnp_gcn128'):
        return mt.appnp_gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,128)
    elif (aggr == 'appnp_gcn48'):
        return mt.appnp_gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48)
    elif (aggr == 'appnp_gcn96'):
        return mt.appnp_gcn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,96)
    elif (aggr == 'arma'):
        return mt.arma_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'gmm'):
        return mt.gmm_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'mean'):
        return mt.mean_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'graphnn'):
        return mt.graphnn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'graphnn_di'):
        return mt.graphnn_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'mf'):
        return mt.mf_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'agnn'):
        return mt.agnn_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'sage'):
        return mt.sage_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control)
    elif (aggr == 'ggin'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48)
    elif (aggr == 'ggin32'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,32)
    elif (aggr == 'ggin24'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,24)
    elif (aggr == 'ggin48'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48)
    elif (aggr == 'ggin64'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,64)
    elif (aggr == 'ggin96'):
        return mt.ggin_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,96)
    elif (aggr == 'ggin_di24'):
        return mt.ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,24)
    elif (aggr == 'ggin_di32'):
        return mt.ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,32)
    elif (aggr == 'ggin_di48'):
        return mt.ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48)
    elif (aggr == 'ggin_di56'):
        return mt.ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,56)
    elif (aggr == 'ggin_di64'):
        return mt.ggin_di_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,64)
    elif (aggr == 'gated_mean32'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,32,'mean')
    elif (aggr == 'gated_mean48'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48,'mean')
    elif (aggr == 'gated_mean56'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,56,'mean')
    elif (aggr == 'gated_add32'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,32,'add')
    elif (aggr == 'gated_add48'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,48,'add')
    elif (aggr == 'gated_add56'):
        return mt.gated_train(data, aggr, sample_mask, val_mask, node_norm, device,time_control,56,'add')
    else:
        print(aggr,"not exist")
def train_one_model(data,aggr,sample_mask,val_mask,node_norm,device,time_control=None):
    if (aggr == 'feature'):
        return mt.label_train_and_predict(data, sample_mask, val_mask, node_norm,time_control)
    else:
        return gnn_train_and_predict(data, aggr, sample_mask, val_mask, node_norm,device, time_control)


def train_two_model(data,aggr,sample_mask_1,val_mask_1,sample_mask_2,val_mask_2,
                    node_norm1=None,node_norm2=None,device='cpu',time_control=None):
    pred1 = train_one_model(data, aggr, sample_mask_1, val_mask_1, node_norm1,device, time_control)
    if (time_control.isTimeToStop() == True):
        return pred1, None
    pred2 = train_one_model(data, aggr, sample_mask_2, val_mask_2, node_norm2,device, time_control)
    return pred1, pred2


def get_sample_retio(data):
    #print("计算采样比例")
    train_num = data.train_mask[data.train_mask == True].shape[0]
    node_num = data.train_mask.shape[0]
    retio = (train_num - 1000) / train_num
    retio = np.max((retio, 0.6666))
    #print("采样比例:",retio)
    return retio


def sample_by_label(x_mask,labels,weight=0.5,top=True,random=False,bootstamp=False):
    if (random == False):
        index = torch.tensor(range(x_mask.shape[0]))[x_mask].numpy()
        label = labels[x_mask].numpy()
        df = pd.DataFrame()
        df['index'] = index
        df['label'] = label
        class_list = list(df['label'].unique())
        sample_list = []

        def typicalsamling(group, sample_list):
            name = int(group.name)
            df = None
            if (bootstamp == False):
                if (top == True):
                    df = group.sample(frac=1, replace=False,random_state=1337).head(int(len(group) * weight))
                else:
                    df = group.sample(frac=1, replace=False,random_state=1337).tail(int(len(group) * weight))
            else:
                df = group.sample(frac=1, replace=True)
            sample_list += list(df['index'].values)
            return df

        result = df.groupby(['label']).apply(typicalsamling, sample_list)
        if (bootstamp == True):
            sample_mask = sample_list
            p = pd.DataFrame()

            p['num'] = [1] * len(sample_mask)
            p['index'] = sample_mask
            p = p.groupby('index').sum()

            sample_mask = torch.zeros(x_mask.shape[0], dtype=torch.bool)
            val_mask = x_mask.clone()
            norm_weight = torch.ones(x_mask.shape[0], dtype=torch.float)

            sample_mask[p.index] = True
            val_mask[p.index] = False

            norm_weight[p.index] = torch.tensor(p['num'].values, dtype=torch.float)
            return sample_mask, val_mask, norm_weight
        sample_list = np.array(sample_list)
        sample_mask = torch.zeros(x_mask.shape[0], dtype=torch.bool)
        sample_mask[sample_list] = True
        val_mask = x_mask.clone().detach()
        val_mask[sample_list] = False
        return sample_mask, val_mask
    else:
        index = torch.tensor(range(x_mask.shape[0]))[x_mask].numpy()
        label = labels[x_mask].numpy()
        df = pd.DataFrame()
        df['index'] = index
        df['label'] = label
        sample_list = df.sample(frac=weight)
        sample_list = np.array(sample_list)
        sample_mask = torch.zeros(x_mask.shape[0], dtype=torch.bool)
        sample_mask[sample_list] = True
        val_mask = x_mask.clone().detach()
        val_mask[sample_list] = False
        return sample_mask, val_mask


def get_sample(data, retio):
    sample_mask1, val_mask1 = sample_by_label(data.train_mask, data.y, retio,True, False)
    sample_mask2, val_mask2 = sample_by_label(data.train_mask, data.y, retio,False, False)

    node_norm1 = torch.ones(data.x.shape[0], dtype=torch.float)
    node_norm1 = node_norm1 / 1.2
    node_norm1[val_mask2] = 1.2

    node_norm2 = torch.ones(data.x.shape[0], dtype=torch.float)
    node_norm2 = node_norm2 / 1.2
    node_norm2[val_mask1] = 1.2
    return sample_mask1, val_mask1, sample_mask2, val_mask2, node_norm1, node_norm2


def count_label(data, is_direct, device):
    for i in range(2):
        data = data.to(device)
        model = Label_Extract()
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model = model.to(device)
                x = model(data, is_direct)
                data.x_label = x
                data = data.to('cpu')
                return data
        except:
            data = data.to('cpu')
            model = model.to('cpu')
            torch.cuda.empty_cache()
    with torch.no_grad():
        data = data.to('cpu')
        model = model.to('cpu')
        x = model(data, is_direct)
        data.x_label = x
    return data


def get_rank_two(result):
    pred = torch.tensor(result)
    max1 = pred.max(dim=1)[1]
    pred = pred - pred.max(dim=1)[0].view(-1, 1)
    pred[pred == 0] = -10000
    max2 = pred.max(dim=1)[1]
    pred = torch.cat((max1, max2), dim=-1)
    return pred


def get_simiraly(best,x,y):
    pre = len(x[(x==y)&(best!=y)])
    return pre


def choose_model(data, aggr_list, train_mask, val_mask, device, clock):
    result_list = []
    model_list = []
    for i in range(0, len(aggr_list)):
        aggr = aggr_list[i]
        pred = train_one_model(data, aggr, train_mask, val_mask, None, device,
                                                     clock)
        if (pred is not None):
            model_list.append(aggr)
            result_list.append(pred)
        if (clock.isTimeToStop() == True):
            print("时间耗尽")
            break
    result_list = np.array(result_list)
    model_list = np.array(model_list)
    if(len(result_list)==0):
        return result_list,model_list
    rank = choose_model_by_result(data, result_list, val_mask, aggr_list)
    return result_list[rank], model_list[rank]


def choose_model_by_result(data, result1, val_mask, aggr_list):
    data = data.to('cpu')
    val_acc = []
    for e in result1:
        pred = np.argmax(e, axis=1)
        pred = torch.tensor(pred)
        correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
        acc2 = correct / (val_mask.sum().item() + 0)
        #print('*gcn 测试集Accuracy: {:.4f}'.format(acc2))
        val_acc.append(acc2)
    val_acc = np.array(val_acc)
    best_index = np.argsort(-val_acc)
    #best_ans = get_rank_two(result1[best_index[0]])
    best_ans = np.argmax(result1[best_index[0]],axis=1)
    sim_list = []
    y = data.y[val_mask].numpy()
    for e in result1:
        pred = np.argmax(e,axis=1)
        sim = get_simiraly(best_ans[val_mask], pred[val_mask],y)
        #print(sim)
        sim_list.append(sim)
    sim_list = np.array(sim_list)
    best_index2 = np.argsort(-sim_list)
    score1 = np.ones(len(best_index))
    score1[best_index] = range(len(best_index))
    score2 = np.ones(len(best_index2))
    score2[best_index2] = range(len(best_index2))
    score2[best_index[0]] = -1
    #print(score2)
    #score = score1 + score2 * 1.1
    score = score2+score1
    print("model selection")
    print(aggr_list)
    print("val acc:", val_acc)
    print("sim: ", sim_list)
    print("val score:", score1)
    print("sim score:", score2)
    print("merge score:", score)
    return np.argsort(score)







