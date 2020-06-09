import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import AGNNConv
from torch_geometric.nn import GMMConv
from torch_geometric.nn import GraphConv, SAGEConv
from torch.nn import Parameter, ModuleList
from torch_geometric.data import Data
import random

from torch_geometric.utils import remove_self_loops, degree, add_self_loops

from torch.nn import ReLU, Sequential
from torch_geometric.nn import GINConv, TopKPooling, GraphConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(1234)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class GraphGNN(torch.nn.Module):        #已调试 待测concat
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(GraphGNN, self).__init__()
        self.first_lin = Linear(features_num, hidden)
        self.conv1 = GraphConv(hidden, hidden, aggr='add')
        self.conv2 = GraphConv(hidden, hidden, aggr='add')
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.out = Linear(hidden, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        first_x = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.fuse_weight[0] * first_x
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.fuse_weight[1] * first_x
        x = self.out(x)
        return F.log_softmax(x, dim=1)


class MF(torch.nn.Module):    #已调试 待测concat
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(MF, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(MFConv(hidden, hidden, 5))
        self.out = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        i = 0
        x_first = x
        for conv in self.convs:
            h = x
            x = conv(x, edge_index)
            x = F.dropout(x, p=0.2, training=self.training)
            x = x + self.fuse_weight[i] * x_first
            i += 1
        x = self.out(x)
        return F.log_softmax(x, dim=-1)


class AGNN(torch.nn.Module):    #待精调 比gat好使
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(AGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.out = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):    #已精调
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.out = Linear(hidden * 3, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        xx = x
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.2, training=self.training)
        xx = torch.cat([xx, x], dim=1)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.2, training=self.training)
        xx = torch.cat([xx, x], dim=1)
        x = self.out(xx)
        return F.log_softmax(x, dim=-1)


class SAGE(torch.nn.Module):    #已精调
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = Linear(hidden * 3, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        xx = x
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.2, training=self.training)
        xx = torch.cat([xx, x], dim=1)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.2, training=self.training)
        xx = torch.cat([xx, x], dim=1)
        x = self.out(xx)
        return F.log_softmax(x, dim=-1)


class GMM(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(GMM, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GMMConv(hidden, hidden, 1, 1))
        self.out = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        #xx = None
        i = 0
        for conv in self.convs:
            h = x
            x_first = h
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=0.2, training=self.training)
            x = x + self.fuse_weight[i] * x_first
            i += 1
        x = self.out(x)
        return F.log_softmax(x, dim=-1)


class FeatureBlock(MessagePassing):
    def __init__(self, aggr='add', **kwargs):
        super(FeatureBlock, self).__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,self.out_channels, self.num_layers)


class Feature_Extract(torch.nn.Module):
    def __init__(self):
        super(Feature_Extract, self).__init__()
        self.conv1 = FeatureBlock('add')

    def forward(self, x, data, has_feature):
        if (has_feature):
            edge_index, edge_weight = data.edge_index, data.edge_weight
            edge_index, norm = GCNConv.norm(edge_index,x.shape[0],edge_weight,dtype=x.dtype)
            x1 = self.conv1(x, edge_index, edge_weight=norm)
            x2 = self.conv1(x1, edge_index, edge_weight=norm)
            return torch.cat([x, x1, x2], dim=1)
        else:
            edge_index, edge_weight = data.edge_index, data.edge_weight
            x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
            x2 = self.conv1(x1, edge_index, edge_weight=edge_weight)
            return torch.cat([x, x1, x2], dim=1)


class GatedConv(MessagePassing):
    def __init__(self, aggr='add',  **kwargs):
        super(GatedConv, self).__init__(aggr=aggr, **kwargs)
    def forward(self, x, edge_index, edge_weight=None):
        h = x
        h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        return h
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,self.out_channels, self.num_layers)
class GatedGIN(torch.nn.Module): #已精调
    def __init__(self,num_layers=2,hidden=32,features_num=32,num_class=2):
        super(GatedGIN, self).__init__()
        self.conv1 = GatedConv(aggr='add')
        self.conv2 = GatedConv(aggr='add')
        self.out = Linear(hidden, num_class)
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.out = Linear(hidden, num_class)
        
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.first_lin(x)
        x = F.dropout(x, p=0.5, training=self.training)
        h = x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.rnn(x, h)
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x,edge_index,edge_weight=edge_weight)
        x = self.rnn(x, h)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)
class GatedGNN(torch.nn.Module): #已精调
    def __init__(self,num_layers=2,hidden=32,features_num=32,num_class=2):
        super(GatedGNN, self).__init__()
        self.conv1 = GatedConv(aggr='add')
        self.conv2 = GatedConv(aggr='add')
        self.out = Linear(hidden, num_class)
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.out = Linear(hidden, num_class)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.first_lin(x)
        x = F.dropout(x, p=0.5, training=self.training)
        h = x
        x_first = x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.rnn(x, h)
        h = x
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x,edge_index,edge_weight=edge_weight)
        x = self.rnn(x, h)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)
class ARMA(torch.nn.Module):#有问题
    def __init__(self, features_num=32, num_class=2):
        super(ARMA, self).__init__()
        self.conv1 = ARMAConv(features_num,32,num_stacks=1,num_layers=1,shared_weights=True,dropout=0.25)

        self.conv2 = ARMAConv(32,num_class,num_stacks=1,num_layers=1,shared_weights=True,dropout=0.25,act=None)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def label_norm(edge_index, num_nodes, edge_weight=None, improved=False,dtype=None):
    if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,device=edge_index.device)
    edge_index,edge_weight= remove_self_loops(edge_index,edge_weight)
    row, col = edge_index
    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    in_deg = in_deg+1
    out_deg = out_deg+1
    deg_inv_sqrt = 1/(in_deg.sqrt())
    deg_out_sqrt = 1/(out_deg.sqrt())
    return edge_index,deg_inv_sqrt[col]*edge_weight*deg_out_sqrt[row]
class Label_Extract(torch.nn.Module):
    def __init__(self):
        super(Label_Extract, self).__init__()
        self.conv1 = FeatureBlock('add')
        self.conv2 = FeatureBlock('add')
    def forward(self, data,is_direct):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        label = data.label
        edge_weight = (edge_weight-edge_weight.min())/(edge_weight.max()-edge_weight.min()+1)
        edge_index, norm = label_norm(edge_index, x.shape[0],None, dtype=x.dtype)
        edge_index2 = data.edge_index[[1,0],:]
        x = label
        xx = None
        x1 = self.conv1(x, edge_index,norm)
        x2 = self.conv1(x, edge_index2,norm)
        x3 = self.conv1(x1,edge_index2,norm)
        x4 = self.conv1(x2,edge_index,norm)
        #x5 = self.conv1(x2,edge_index,norm)
        #x6 = self.conv1(x2,edge_index2,norm)
        re = self.conv1(torch.ones((x1.shape[0],1),dtype = torch.float), edge_index2,norm*norm)
        #x1 = self.conv1(x, edge_index,norm*norm)
        x3 = x3 - label*re
        print(x3.shape)
        return torch.cat([x3-x1,x3,x1],dim=1)
    
class GAT(torch.nn.Module):#已调试
    def __init__(self, num_layers=3, hidden=64, features_num=32, num_class=2):
        super(GAT, self).__init__()
        self.first_lin = Linear(features_num, 4)
        self.conv1 = GATConv(features_num, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_class, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        #x = self.first_lin(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__

class GatedGCN_directed(torch.nn.Module):
    def __init__(self,num_layers=2,hidden=32,aggr='add',features_num=32,num_class=2,res=False,node_num=0):
        super(GatedGCN_directed, self).__init__()
        print(num_layers, aggr, hidden, res)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GatedBlock(hidden, aggr))
        self.res = res
        self.rnn = torch.nn.GRUCell(hidden * 2, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.rnn.reset_parameters()
        self.out = Linear(hidden, num_class)

    def reset_parameters(self):
        self.first_lin.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2 = data.edge_index[[1, 0], :]
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        if (self.res == True):
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x1 = conv(x, edge_index, edge_weight=edge_weight)
                x2 = conv(x, edge_index2, edge_weight=edge_weight)
                x = torch.cat([x1, x2], dim=1)
                x = self.rnn(x, h)
                i += 1
        else:
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x, h)
                if (i != 0):
                    x = x + self.fuse_weight[i] * x_first
                i += 1
        x = self.out(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ensemble_linear(torch.nn.Module):
    def __init__(self, features_num=32, num_class=2):
        super(ensemble_linear, self).__init__()
        self.lin1 = Linear(features_num, features_num)
        self.lin2 = Linear(features_num, num_class)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
class GatedBlock(torch.nn.Module):
    def __init__(self, hidden=32,aggr='add'):
        super(GatedBlock, self).__init__()
        self.nn1 = Linear(hidden,hidden)
        self.conv1 = GatedConv(aggr=aggr)
    def forward(self, x,edge_index, edge_weight):
        #x = self.bn2(x)
        
        x = self.conv1(x, edge_index,edge_weight=edge_weight)
        x = self.nn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        return x
    def __repr__(self):
        return self.__class__.__name__
class GatedGCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=32,  features_num=32, num_class=2,res=True,aggr='add'):
        super(GatedGCN, self).__init__()
        print(num_layers,aggr,hidden,res)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GatedBlock(hidden, aggr))
            #self.convs.append(GCNConv(hidden,hidden))
        #print(self.convs)
        self.res = res
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers), requires_grad=True)
        self.fuse_weight.data.fill_(float(1)/(num_layers+1))
        
        #print(self.fuse_weight)
        self.rnn.reset_parameters()
        self.out = Linear(hidden,num_class)
        self.reset_parameters
    def reset_parameters(self):
        self.lin2.reset_parameters()
        self.first_lin.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        
        #x = self.bn1(x)
        x = F.dropout(x, p=0.7, training=self.training)
        
        #x = self.bn2(x)
        if(self.res==True):
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x,h)
                x = x+self.fuse_weight[i]*x_first
                i += 1
        else:
            i=0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x,h)
                if(i!=0):
                    x = x+self.fuse_weight[i]*x_first
                i += 1
        x = self.out(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name_
    
class Lin2_APPNP(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=48,  features_num=16, num_class=2,K=20,alpha=0.1):
        super(Lin2_APPNP, self).__init__()
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.ppnp=APPNP(K=K, alpha=alpha)
        

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x=F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x=self.lin2(x)
        x=self.ppnp(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
class GCN_APPNP(torch.nn.Module):
    def __init__(self,num_layers=1,hidden=48,features_num=16,num_class=2,K=5,alpha=0.2):
        super(GCN_APPNP, self).__init__()
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
        self.ppnp = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = self.ppnp(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class StopEarly:
    def __init__(self, pat=3):
        self.last_val_loss = 3000000000
        self.times = 0
        self.pat = pat
        self.min_val_loss = 3000000000
        self.bestModel = None
        self.best_epoch = 0

    def copy(self, dirs):
        new_dir = {}
        for key in dirs.keys():
            new_dir[key] = dirs[key].clone()
        return new_dir

    def isTimeToStop(self, val_loss, model, epoch):
        if (epoch % 50 == 0):
            if (val_loss >= self.last_val_loss):
                self.times += 1
        if (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.bestModel = self.copy(model.state_dict())
            self.best_epoch = epoch
        self.last_val_loss = val_loss
        if (self.times >= self.pat):
            print(self.best_epoch)
            model.load_state_dict(self.bestModel)
            return True
        if (epoch == 500):
            print("正常结束")
            print(self.best_epoch)
            model.load_state_dict(self.bestModel)
            return True
        return False


class GIN(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=32, features_num=32, num_class=2):
        super(GIN, self).__init__()

        self.lin1 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.conv1 = GINConv(self.lin1)
        self.conv2 = GINConv(self.lin3)
        self.lin2 = Linear(hidden, num_class)

        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (3))
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_first = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = x + self.fuse_weight[0] * x_first
        x_first = x
        x = F.relu(self.conv2(x, edge_index))    #, edge_attr=edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.fuse_weight[1] * x_first
        #x=x+first_x
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class MFConv(MessagePassing):
    def __init__(self,in_channels,out_channels,max_degree=5,root_weight=False,bias=True,**kwargs):
        super(MFConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree
        self.root_weight = root_weight

        self.rel_lins = torch.nn.ModuleList([
                Linear(in_channels, out_channels, bias=bias)
                for _ in range(max_degree + 1)
        ])

        if root_weight:
            self.root_lins = torch.nn.ModuleList([
                    Linear(in_channels, out_channels, bias=False)
                    for _ in range(max_degree + 1)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        if self.root_weight:
            for lin in self.root_lins:
                lin.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)

        deg = degree(edge_index[1 if self.flow == 'source_to_target' else 0],
                                 x.size(0),
                                 dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        if not self.root_weight:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        h = self.propagate(edge_index, x=x)

        out = x.new_empty(list(x.size())[:-1] + [self.out_channels])

        for i in deg.unique().tolist():
            idx = (deg == i).nonzero().view(-1)

            r = self.rel_lins[i](h.index_select(0, idx))
            if self.root_weight:
                r = r + self.root_lins[i](x.index_select(0, idx))

            out.index_copy_(0, idx, r)

        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,self.out_channels)
    
    
    
'''
class GatedBlock(torch.nn.Module):#被代替
    def __init__(self, hidden=32, aggr='add'):
        super(GatedBlock, self).__init__()
        self.nn1 = Linear(hidden, hidden)
        self.conv1 = GatedConv(hidden, aggr=aggr)

    def reset_parameters(self):
        self.nn1.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        #x = self.bn2(x)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.nn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print(x.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__
'''

class GatedGCN_inplace(torch.nn.Module): #被代替
    def __init__(self,num_layers=2,hidden=32,aggr='add',features_num=32,num_class=2,res=True):
        super(GatedGCN_inplace, self).__init__()
        print(num_layers, aggr, hidden, res)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GatedBlock(hidden, aggr))
        self.res = res
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.lin = Linear(hidden, hidden)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.rnn.reset_parameters()
        self.out = Linear(hidden, num_class)
        self.reset_parameters()

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.rnn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))

        x = F.dropout(x, p=0.5, training=self.training)

        if (self.res == True):
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x, h)
                x = x + self.fuse_weight[i] * x_first
                i += 1
        else:
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x, h)
                if (i != 0):
                    x = x + x_first
                i += 1
        x = self.out(x)
        return F.log_softmax(x, dim=-1)

    def embedding(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))

        x = F.dropout(x, p=0.5, training=self.training)

        if (self.res == True):
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x, h)
                x = x + x_first
                i += 1
        else:
            i = 0
            for conv in self.convs:
                x_first = x
                h = x
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = self.rnn(x, h)
                if (i != 0):
                    x = x + x_first
                i += 1
        return x

    def __repr__(self):
        return self.__class__.__name_





