import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.data import Data
from ensemble import get_ensemble_result

from ensemble import evalue_result
import time
import model_utils as mu
import os
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)
def init():
    path = os.path.abspath(__file__)
    path = path.split('/')
    new_path = '/'
    for i in range(len(path)-1):
        new_path = new_path+path[i]+'/'
    print(new_path)
    os.system('g++ -O2 -std=c++11 -o orca '+new_path+'orca.cpp')
    os.system('g++ -O2 -std=c++11 -o edge '+new_path+'edge.cpp')
init()
class time_control:
    def __init__(self,start_time,remain_time):
        self.start_time = start_time
        self.remain_time = remain_time
        self.end_time = start_time+remain_time-3
    def isTimeToStop(self):
        now = time.time()
        #print(now)
        if(now>=self.end_time):
            return True
        else:
            return False
    def get_remain_time(self):
        now = time.time()
        remain = self.end_time-now
        remain = np.max((remain,0))
        return int(remain)
class Model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
    def normalize(self, data):
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        return data
    def generate_pyg_data(self, data):

        x = data['fea_table']
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
            self.has_feature=False
        else:
            x = x.drop('node_index', axis=1).to_numpy()
            if(x.max()==x.min()):
                x = data['fea_table']
                x = x.to_numpy()
                x = x[:,0]
                x = x.reshape(x.shape[0])
                x = np.array(pd.get_dummies(x))
                self.has_feature=False
            else:
                self.has_feature=True
        x = torch.tensor(x, dtype=torch.float)
        df = data['edge_file']
        edge_index_weight = df[['src_idx', 'dst_idx','edge_weight']].to_numpy()
        edge_index_weight = np.array(sorted(edge_index_weight, key=lambda d: d[0]))
        edge_index=edge_index_weight[:,0:2]
        edge_weight=edge_index_weight[:,2:3]
        edge_weight = edge_weight.reshape(edge_weight.shape[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        #计算label
        y = data.y.numpy()
        y = np.array(pd.get_dummies(y))
        y = torch.tensor(y,dtype=torch.float)
        y[~data.train_mask]=torch.zeros(y.shape[1])
        yy = torch.tensor((~data.train_mask).clone().detach().numpy(),dtype=torch.float).view(-1,1)
        y = torch.cat((y,yy),dim=1)
        data.label=y
        return data
    def pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred
    def pre_choose(self,data):
        if(data.is_directed()==True):
            if(self.has_feature==True):
                aggr_list = ['feature','gated_add_di','gated_mean_di','gated_add_di','gated_mean_di']#,'gated_mean56','gated_add56']
            else:
                if(self.label_is_work):
                    aggr_list = ['feature','feature','feature','ggin_di48','ggin_di32','ggin_di64']
                else:
                    aggr_list = ['ggin_di48','ggin_di24','ggin_di56','ggin_di32','ggin_di64']
        else:
            if(self.has_feature==True):
                aggr_list =['feature','gated_add48','gated_mean48','gcn64','appnp_gcn48','gat']
               #aggr_list =['feature','feature','feature','feature','feature','feature']
            else:
                if(self.label_is_work):
                    aggr_list = ['feature','ggin48','ggin24','ggin64','ggin32','ggin96']
                else:
                    aggr_list = ['ggin48','ggin24','ggin64','ggin32','ggin96']
        return aggr_list
    def check_time(self,data):
        aggr_list = self.pre_choose(data)
        #time_list = []
        #for aggr in aggr_list:
        #    if(aggr!=aggr_list[0]):
        #        t = mu.gnn_test_time(data,aggr,self.device)
        #        time_list.append(t)
        #    else:
        #        t=0
        #        time_list.append(t)
        #aggr_list = np.array(aggr_list)
        #time_list = np.array(time_list)
        #rank_list = np.argsort(time_list)
        #aggr_list = aggr_list[rank_list]
        return aggr_list
    def get_edge(self,data):
        edge_index,edge_weight = mu.add_direct_edge(data)
        #edge_index,edge_weight = None,None
        if(edge_index is not None and edge_weight is not None):
            print("添加有向边")
            edge_weight = edge_weight.reshape(edge_weight.shape[0])
            edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        else:
            edge_index = data.edge_index[[1,0],:]
            edge_weight = data.edge_weight.clone()
        data.edge_index_re,data.edge_weight_re = edge_index,edge_weight
        return data
    def choose_model(self,data,train_mask1,val_mask1):
        ans_list = []
    def pre(self,data):
        data.is_direct = data.is_directed()
        data.num_class = int(max(data.y))+1
        self.is_direct = data.is_direct
        if(self.is_direct==True):
            data = self.get_edge(data)
        self.edge_expand = False
    def get_feature(self,data):
        return data
    def get_sample_retio(self,data):
        #print("计算采样比例")
        train_num = data.train_mask[data.train_mask==True].shape[0]
        #node_num = data.train_mask.shape[0]
        #retio = (train_num-1000)/train_num
        #retio = np.max((retio,0.6666))
        sample_node = int(train_num*2/3)
        if(train_num-sample_node>=1200):
            sample_node=train_num-1200
        #
        retio = sample_node/train_num
        print("采样比例:",retio)
        return retio
    def get_ans(self,data,time_budget):
        self.pre(data)
        retio = self.get_sample_retio(data)
        data=data.to('cpu')
        data = mu.count_label(data,self.is_direct,self.device)#label提取
        self.label_is_work = mu.check_label(data,self.device) #判断标签会不会泄漏
        aggr_list = self.check_time(data)
        
        data=data.to('cpu')
        #self.label_is_work=False
        data = mu.get_feature(data,self.label_is_work,self.is_direct,self.has_feature,self.edge_expand,self.device) #特征提取
        result1 = []
        result2 = []
        
        data=data.to('cpu')
        sample_mask1,val_mask1,sample_mask2,val_mask2,node_norm1,node_norm2 = mu.get_sample(data,retio)
        data.to(self.device)
        elapsed = (time.time() - self.start)
    
        #aggr_list = np.hstack((aggr_list,aggr_list))
        print(aggr_list)
        #result1.append(mu.train_one_model(data,'feature',sample_mask1,val_mask1,node_norm2,self.device,time_control=None))
        #result2.append(mu.train_one_model(data,'feature',sample_mask2,val_mask2,node_norm2,self.device,time_control=None))
        #pred = mu.feature_train(data,sample_mask1,val_mask1,node_norm1,self.device,clock)
        print(self.start,time.time())
        print("已用时间",elapsed)
        print("剩余时间:",time_budget-elapsed)
        #开始训练 40%选择模型 %50 训练模型 %10ensemble
        remain_time = time_budget-elapsed
        print(remain_time)
        print(remain_time*0.4)
        node_norm1=None
        node_norm2=None
        
        clock = time_control(time.time(),remain_time*0.4)
        choose_ans_list,model_list = mu.choose_model(data,aggr_list,sample_mask1,val_mask1,self.device,clock)
        #print("模型选择完毕")
        print(model_list)
        if(len(model_list)==0):
            return np.random.randint(0,int(data.y.max()+1),data.y.shape[0])
        best_model = model_list[0]
        result1.append(choose_ans_list[0])
        print(self.start,time.time())
        elapsed = (time.time() - self.start)
        
        print("已用时间",elapsed)
        print("剩余时间:",time_budget-elapsed)

        remain_time = time_budget-elapsed
        clock = time_control(time.time(),remain_time-20)
        result2.append(mu.train_one_model(data,best_model,sample_mask2,val_mask2,node_norm2,self.device,clock))
        for i in range(1,len(model_list)):
            if(clock.isTimeToStop()==True):
                print(time.time())
                print(clock.end_time)
                print("时间耗尽")
                break
            aggr = model_list[i]
            pred1 = mu.train_one_model(data,aggr,sample_mask2,val_mask2,node_norm2,self.device,clock)
            if(pred1 is not None):
                result1.append(choose_ans_list[i])
                result2.append(pred1)
            if(clock.isTimeToStop()==True):
                print(time.time())
                print(clock.end_time)
                print("时间耗尽")
                break
            
        data = data.to(self.device)
        #return result1,result2,val_mask1,val_mask2
        result1 = np.array(result1)
        result2 = np.array(result2)
        #print(result1)
        #print(result2)
        #model_count = np.max((int(len(result1)*0.8),2))
        #model_count = np.min((model_count,len(result1)))
        #result1 = result1[range(model_count)]
        #result2 = result2[range(model_count)]
        #print("剩余数量",model_count)
        num_model = len(result1)
        
        
        elapsed = (time.time() - self.start)
        
        print("已用时间",elapsed)
        print("剩余时间:",time_budget-elapsed)

        remain_time = time_budget-elapsed
        clock = time_control(time.time(),remain_time-2)
        pred = None
        if(num_model!=0):
            pred,pred1,pred2,pred3,pred4 = get_ensemble_result(data,result1,result2,val_mask1,val_mask2,num_model,self.device,clock)
        else:
            choose_ans_list = np.array(choose_ans_list)
            if(len(choose_ans_list)!=0):
                pred = choose_ans_list.mean(axis=0)
            else:
                pred = np.random.randint(0,int(data.y.max()+1),data.y.shape[0])
        elapsed = (time.time() - self.start)
        print(self.start,time.time())
        print("已用时间",elapsed)
        print("剩余时间:",time_budget-elapsed)
        return pred
    def train_predict(self, data,time_budget,n_class,schema):
        #num_model=4
        #aggr_list=['add','appnp','gcn','mean']
        #hidden=48
        #ime_budget=200
        self.start = time.time()
        print(self.start,time.time())
        data = self.generate_pyg_data(data)
        pred = self.get_ans(data,time_budget)
        
        #return pred,pred1,pred2,pred3,pred4,data
        #result1,result2,val_mask1,val_mask2 = self.get_ans(data,time_budget)
        #return result1,result2,val_mask1,val_mask2,data
        #return pred,pred1,pred2,pred3,pred4,data
        #pred = pred.max(dim=1)[1]
        data = data.to('cpu')
        pred = pred[data.test_mask]
        return pred.flatten()