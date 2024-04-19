"""IGKT modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F

# from ..relgraphconv import RelGraphConv
from dgl.nn.pytorch import RelGraphConv, SAGEConv

def uniform(size, tensor):
    # 将张量中的值初始化均匀分布
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class IGKT(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # RGCN, GAT, LSTM, 
    '''
    in_feats：输入特征的维度
    latent_dim：隐藏层的维度列表
    num_relations：关系类型的数量
    num_bases：每个关系类型的基础数量？？？什么意思
    regression: 标志模型是否用于回归任务
    edge_dropout: 边的dropout比率
    force_undirected: 是否强制图是无向的
    ide_features: 是否有侧边信息
    n_side_features: 侧边信息的数量
    multiply_by: 用于缩放输出的系数
    '''
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGKT, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList() # 创建一个moduleList来存储图卷积层
        print(in_feats, latent_dim, num_relations, num_bases)
        self.convs.append(gconv(in_feats, latent_dim[0], num_relations, num_bases=num_bases, self_loop=True,))
        for i in range(0, len(latent_dim)-1): # 循环创建图卷积层，并添加到ModuleList中
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases, self_loop=True,))
        
        # 根据是否有辅助特征，调整线性层的输入维度
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, block):
        # block参数，代表传入的图数据
        # 对于输入的图块block应用边的dropout，防止过拟合
        block = edge_drop(block, self.edge_dropout, self.training)

        concat_states = [] # 初始化一个空列表，用于存储每层图卷积后的节点特征
        x = block.ndata['x'].type(th.float32) # one hot feature to emb vector : this part fix errors
        # 从图块block中提取节点数据x，并执行图卷积操作。
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block, x, block.edata['etype'], 
                             norm=block.edata['edge_mask'].unsqueeze(1)))
            concat_states.append(x) # 将当前层的输出x添加到状态列表中
        concat_states = th.cat(concat_states, 1)  # 沿着第一个维度（特征维度）拼接所有层的输出。
        
        # 从图块节点中提取标签nlabel，并创建一个布尔数组users和items,表示哪些节点是用户节点，哪些是物品节点。
        users = block.ndata['nlabel'][:, 0] == 1 
        items = block.ndata['nlabel'][:, 1] == 1
        # 根据用户和物品的标签，从concat_states中选择相应的特征，并将他们拼接
        x = th.cat([concat_states[users], concat_states[items]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)

    # 重写类的字符串表示方法，返回类的名称
    def __repr__(self):
        return self.__class__.__name__

# 定义了一个函数来实现边的drop_out
def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    # 如果不是训练状态，不应用边的dropout
    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges() # 返回途中所有表的源节点（src）和目标节点（_,这里用下划线表示，因为我们只关心源节点）
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    # 创建新的张量，与图中边的数量相同，用dropout填充，意味着每个边以dropout的概率被丢弃
    to_drop = th.bernoulli(to_drop).to(th.bool) 
    # 根据给定的概率值进行伯努利实验，将to_drop张量中的每个元素转换为0或1，其中1的概率是drop_out
    graph.edata['edge_mask'][to_drop] = 0
    # graph.edata['edge_mask'] 是图中边数据（edge data）的一个字段，用于存储与边相关的额外信息。
    # edge_mask 中对应于 to_drop 张量中值为 True 的位置设置为 0。这意味着这些边将被"丢弃"，即在后续的计算中不被考虑。

    return graph




class IGKT_TS(IGKT):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # RGCN, GAT, LSTM, 
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGKT, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        print(in_feats, latent_dim, num_relations, num_bases)
        self.convs.append(gconv(in_feats, latent_dim[0], num_relations, num_bases=num_bases, self_loop=True,))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases, self_loop=True,))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        self.lin2 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, block):
        block = edge_drop(block, self.edge_dropout, self.training)

        concat_states = []
        x = block.ndata['x'].type(th.float32) # one hot feature to emb vector : this part fix errors
        
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block, x, block.edata['etype'], 
                             norm=block.edata['ts'].unsqueeze(1))) # 时间戳作为归一化因子
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1) # 沿着特征维度拼接所有的输出
        
        users = block.ndata['nlabel'][:, 0] == 1
        items = block.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
