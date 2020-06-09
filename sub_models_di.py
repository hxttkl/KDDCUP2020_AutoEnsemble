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


class GraphGNN_di(torch.nn.Module):        #已调试 待测concat
    def __init__(self, num_layers=2, hidden=32, features_num=32, num_class=2):
        super(GraphGNN_di, self).__init__()
        print("有向图graphgnn")
        self.first_lin = Linear(features_num, hidden)
        self.conv1 = GraphConv(hidden, hidden, aggr='add')
        self.conv1_di = GraphConv(hidden, hidden, aggr='add')
        self.conv2 = GraphConv(hidden, hidden, aggr='add')
        self.conv2_di = GraphConv(hidden, hidden, aggr='add')
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.out = Linear(hidden*2, num_class)
        self.con_lin1 = Linear(hidden*2,hidden)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index_re, edge_weight_re =  data.edge_index_re, data.edge_weight_re
        if(data.edge_index_re is None):
            edge_index_re = edge_index[[1,0],:]
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = F.relu(self.conv1_di(x, edge_index_re))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x = torch.cat([x1,x2],dim=1)
        x = self.con_lin1(x)
        
        x1 = F.relu(self.conv2(x, edge_index))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = F.relu(self.conv2_di(x, edge_index_re))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x = torch.cat([x1,x2],dim=1)
        
        x = self.out(x)
        return F.log_softmax(x, dim=-1)
    
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
class GatedGIN_di(torch.nn.Module): #已精调
    def __init__(self,num_layers=2,hidden=32,features_num=32,num_class=2):
        super(GatedGIN_di, self).__init__()
        
        print("有向图ggin")
        self.conv1 = GatedConv(aggr='add')
        self.conv2 = GatedConv(aggr='add')
        self.out = Linear(hidden, num_class)
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.out = Linear(hidden, num_class)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.con_lin1 = Linear(hidden*2,hidden)
        self.con_lin2 = Linear(hidden*2,hidden)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index_re, edge_weight_re =  data.edge_index_re, data.edge_weight_re
        if(data.edge_index_re is None):
            edge_index_re = edge_index[[1,0],:]
        x = self.first_lin(x)
        x = F.dropout(x, p=0.5, training=self.training)
        h = x
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x2 = self.conv1(x, edge_index_re, edge_weight=edge_weight_re)
        x = torch.cat([x1,x2],dim=1)
        x = self.con_lin1(x)
        x = self.rnn(x, h)
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x1 = self.conv2(x,edge_index,edge_weight=edge_weight)
        x2 = self.conv2(x,edge_index_re,edge_weight=edge_weight_re)
        x = torch.cat([x1,x2],dim=1)
        x = self.con_lin2(x)
        x = self.rnn(x, h)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)
    
class GatedGNN_di(torch.nn.Module): #已精调
    def __init__(self,num_layers=2,hidden=32,features_num=32,num_class=2,aggr='add'):
        super(GatedGNN_di, self).__init__()
        print("有向图ggnn")
        self.conv1 = GatedConv(aggr = aggr)
        self.conv2 = GatedConv(aggr = aggr)
        self.out = Linear(hidden, num_class)
        self.rnn = torch.nn.GRUCell(hidden, hidden, bias=True)
        self.first_lin = Linear(features_num, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.out = Linear(hidden, num_class)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(num_layers),requires_grad=True)
        self.fuse_weight.data.fill_(float(1) / (num_layers + 1))
        self.con_lin1 = Linear(hidden*2,hidden)
        self.con_lin2 = Linear(hidden*2,hidden)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index_re, edge_weight_re =  data.edge_index_re, data.edge_weight_re
        if(data.edge_index_re is None):
            edge_index_re = edge_index[[1,0],:]
        x = self.first_lin(x)
        x = F.dropout(x, p=0.5, training=self.training)
        h = x
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x2 = self.conv1(x, edge_index_re, edge_weight=edge_weight_re)
        x = torch.cat([x1,x2],dim=1)
        x = self.con_lin1(x)
        x = self.rnn(x, h)
        h = x
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x1 = self.conv2(x,edge_index,edge_weight=edge_weight)
        x2 = self.conv2(x,edge_index_re,edge_weight=edge_weight_re)
        x = torch.cat([x1,x2],dim=1)
        x = self.con_lin2(x)
        x = self.rnn(x, h)
        x = self.out(x)
        return F.log_softmax(x, dim=-1)






