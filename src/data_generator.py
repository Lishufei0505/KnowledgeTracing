
import pickle
import numpy as np
import pandas as pd

import torch as th
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import coo_matrix

import dgl # 用于处理图数据

import config

def one_hot(idx, length):
    '''
    接受一个索引列表和长度length, 返回一个one-hot编码的张量
    '''
    x = th.zeros([len(idx), length], dtype=th.int32) 
    # len(idx) 是idx列表的长度
    # length 是 one-hont编码向量的长度（通常由idx的最大值加一确定）
    x[th.arange(len(idx)), idx] = 1.0 
    # arange(len(idx))  [0, len()idx-1] 在每一行，和idx索引指定的列设置值为1.0 ，其余位置是0
    return x  

#######################
# Subgraph Extraction  子图提取 
# 该函数接受 DGL 图、用户节点索引、物品节点索引、用户邻居和物品邻居，返回一个包含子图的 DGL 图。
# 子图中包含了节点标签、边标签和时间戳等信息。
#######################
def get_subgraph_label(graph:dgl.graph,
                       u_node_idx:th.tensor, i_node_idx:th.tensor,
                       u_neighbors:th.tensor, i_neighbors:th.tensor,
                       )->dgl.graph:
    # u_neighbors:th.tensor 邻居节点索引的张量
    nodes = th.cat([u_node_idx, i_node_idx, u_neighbors, i_neighbors], dim=0,) 
    # 将用户节点索引、物品节点索引和他们的邻居节点索引拼接起来，形成一个包含所有节点索引的张量
    nodes = nodes.type(th.int32)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 
    # 根据提供的节点索引nodes从原始图graph中提取子图
    node_labels = [0,1] + [2]*len(u_neighbors) + [3]*len(i_neighbors)
    # 创建一个节点标签列表，其中用户节点和物品节点分别标记0和1，用户的邻居节点标记为2，物品的邻居节点标记为3
    subgraph.ndata['nlabel'] = one_hot(node_labels, 4) 
    # 将节点标签转换为one-hot编码的向量
    subgraph.ndata['x'] = subgraph.ndata['nlabel']
    # 将节点标签设置为节点特征？？？？
    # set edge mask to zero as to remove links between target nodes in training process
    subgraph = dgl.add_self_loop(subgraph) # 为子图中的每个节点添加自环
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.float32)
    # 为子图的每条边创建一个edge_mask特征，初始值为1，表示所有边都是激活的。
    target_edges = subgraph.edge_ids([0, 1], [1, 0], return_uv=False)
    # 获取子图中用户节点和物品节点之间的边的ID

    # normalized timestamp
    timestamps = subgraph.edata['ts'] # 从子图的边数据中提取时间戳
    standard_ts = timestamps[target_edges.to(th.long)[0]] # 从目标边的时间戳中选择一个作为标准化的时间戳基准
    timestamps = timestamps - standard_ts.item() # 所有的时间戳减去标准化的时间戳基准
    # 对时间戳进行归一化处理，使其值在[min(timestamps)，standard_ts]范围内
    timestamps = th.clamp(1 - (timestamps - th.min(timestamps)) / (th.max(timestamps)-th.min(timestamps) + 1e-5), min=th.min(timestamps), max=standard_ts)
    subgraph.edata['ts'] = timestamps + 1e-9
    # 将归一化后的时间戳加回一个很小的值，以避免除以零的错误。
    rating, ts = subgraph.edata['label'].unsqueeze(1), subgraph.edata['ts'].unsqueeze(1)
    # 将标签和时间戳转化为二维张量，以便与其他特征拼接
    subgraph.edata['efeat'] = th.cat([rating, ts], dim=1)
    subgraph.remove_edges(target_edges)
    # 从子图中移除目标边，即用户节点和物品节点之间的边。
    return subgraph




'''
用于创建一个图序列数据集
定义了初始化方法，用于加载用户序列、物品序列、部分矩阵、标签矩阵等
并构建图结构
'''
class KT_Sequence_Graph(Dataset):
    def __init__(self, user_groups, item_groups, df, part_matrix, tag_coo_matrix, seq_len):
        '''
        user_groups: 用户组
        item_groups: 物品组
        df : dataFrame 二维表格
        part_matrix: 部分矩阵
        tag_coo_matrix: 标签坐标矩阵
        seq_len: 序列长度
        '''
        self.user_seq_dict = {} # 创建空自字典，用于存储用户交互物品的序列数据
        self.seq_len = seq_len # 序列长度是 seq_len
        self.user_ids = [] # 用户ID列表
        self.part_matrix, self.tag_coo_matrix = part_matrix, tag_coo_matrix,

        self.user_id_set = set() # 用户ID 集合

        # 初始化空列表，用于存储用户行为序列中的用户ID、内容ID 和 正确性标记
        uids = [] 
        eids = []
        correctness = []

        # get user seqs 获取用户的交互序列
        for user_id in user_groups.index:
            self.user_id_set.add(user_id)
            # 遍历user_groups中的每个用户ID,构建用户的行为序列。
            # 序列中包含内容ID、部分、标签内容ID、标签滞后、问题结束时间、回答正确性、问题开始时间、用户答案和时间戳。
            c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = user_groups[user_id] 
            # 一个用户的数据中包含多个属性

            n = len(c_id)
            uids.extend([user_id]*n)
            eids.extend(c_id) # 将列表中的每一个元素添加到另一个列表末尾，append会将列表作为一个整体添加到末尾
            correctness.extend(ans_c)

            # 如果内容ID的长度小于2，则跳过当前循环，过滤掉交互小于2的。
            if len(c_id) < 2:
                continue
            
            # 如果交互大于序列长度seq_len，则将序列分割成多个子序列
            if len(c_id) > self.seq_len:
                initial = len(c_id) % self.seq_len # initial 表示余数
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0") # 将用户id与0拼接后添加到用户id列表
                    self.user_seq_dict[f"{user_id}_0"] = ( # 用户的交互序列添加到字典
                        c_id[:initial], part[:initial], t_c_id[:initial], t_lag[:initial], 
                        q_et[:initial], ans_c[:initial], q_he[:initial], u_ans[:initial],
                        ts[:initial]
                    )
                chunks = len(c_id)//self.seq_len #  整除，表示可以切分多少个长度为seq_len的子序列
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.user_seq_dict[f"{user_id}_{c+1}"] = (
                        c_id[start:end], part[start:end], t_c_id[start:end], t_lag[start:end], 
                        q_et[start:end], ans_c[start:end], q_he[start:end], u_ans[start:end],
                        ts[start:end]
                    )
            else: # 如果交互的长度没有超过seq_len
                self.user_ids.append(f"{user_id}") # 直接将原始的用户id添加到用户
                # 将完整的c_id和相关列表存储在self.user_seq_dict字典中，键是用户ID
                self.user_seq_dict[f"{user_id}"] = (c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts)
        
        self.item_seq_dict = {} # 物品的用户字典，多少个用户和该物品相关联？
        for user_seq_id in self.user_ids:  # 遍历用户id列表，用户id是原始用户id或者增加索组成的字符串
            user_seq = self.user_seq_dict[user_seq_id] # 每个用户的交互序列
            target_cid = user_seq[0] # c_id 
            target_cid = target_cid[-1] # 最后一个元素
            # 通过 目标 cid作为索引，从item_group中获取相关的物品信息
            u_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = item_groups[target_cid] 
            # uIf 可能是与商品交互过的用户列表
            n = self.seq_len #*2
            if n > len(u_id):
                n =len(u_id)
            indices = np.random.choice(len(u_id), n, replace=False)  # 从uid列表中随机选择n个不重复的索引
            # 使用随机索引从u_id列表中获取相应的元素，作为字典中键的值
            self.item_seq_dict[user_seq_id] = u_id[indices]

        # build user-exe matrix 构建用户执行矩阵
        uids = df['user_id'] # 所有用户的唯一标识符
        eids = df['content_id'] # 所有内容的唯一标识符
        correctness = df['answered_correctly'] # 用户对某些内容回答是否正确
        ts = df['timestamp']
        ts=(ts-ts.min())/(ts.max()-ts.min()) # 时间戳归一化
        num_user = max(uids)+1 
        print(num_user)
        print(num_user+config.TOTAL_EXE)

        uids += config.TOTAL_EXE # 将config.TOTAL_EXE当成一个单独的元素添加到列表中
        num_nodes = num_user+config.TOTAL_EXE

        src_nodes = np.concatenate((uids, eids)) # 将两个数组拼接起来构建图的源节点
        dst_nodes = np.concatenate((eids, uids)) # 构建目标节点
        correctness = np.concatenate((correctness, correctness)) # 创建无向边
        ts = np.concatenate((ts, ts)) # 边中带有时间信息
        print(len(src_nodes), len(dst_nodes), len(correctness))
        # 创建稀疏矩阵coo_matrx（值，（行索引，列索引），形状（， ））
        # 矩阵表示用户之间的交互，其中correctness作为数据，src_nodes 和 dst_nodes 分别作为行列索引
        user_exe_matrix = coo_matrix((correctness, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        
        # build graph 
        self.graph = dgl.from_scipy(sp_mat=user_exe_matrix, idtype=th.int32) # 从稀疏矩阵创建图
        self.graph.ndata['node_id'] = th.tensor(list(range(num_nodes)), dtype=th.int32) # 为每一个node添加node_id属性
        self.graph.edata['label'] = th.tensor(correctness, dtype=th.float32) # 为每条边添加属性 正确性作为边的标签信息
        self.graph.edata['etype'] = th.tensor(correctness, dtype=th.int32) # 正确性作为边的类型
        self.graph.edata['ts'] = th.tensor(ts, dtype=th.float32) # 为每条边添加时间属性
        ts_max = self.graph.edata['ts'].max() # 记录时间的最大值，用于后序标准化和其他时间计算

        src, dst, etypes = [], [], []

        print('--------------------------- node degree before')
        print(self.graph.in_degrees().float().mean()) # 打印当前节点入度的平均值
        print('--------------------------- node degree after')
        
        print('start part') 
        # 部分矩阵相关
        for i in self.part_matrix.keys():
            for j in self.part_matrix[i].keys():
                if i==j: continue
                part = self.part_matrix[i][j]
                if part > 0:
                    src.append(i)
                    dst.append(j) 
                    etypes.append(2) # 源节点和目标节点存在某种联系，边的类型是2

        print('start tag')
        # 标签的矩阵
        # 根据tag_coo_matrix 决定添加边的类型
        for i in self.tag_coo_matrix.keys():
            for j in self.tag_coo_matrix[i].keys():
                if i==j: continue
                tag_coo = self.tag_coo_matrix[i][j]
                if tag_coo == 1: 
                    src.append(i)
                    dst.append(j)
                    etypes.append(3)
                elif tag_coo == 2:
                    src.append(i)
                    dst.append(j)
                    etypes.append(4)
                elif tag_coo >= 3:
                    src.append(i)
                    dst.append(j)
                    etypes.append(5)

        # 先创建空列表，再往图中添加       
        print('start adding edges')
        n_edges =  len(etypes) # 首先计算边的总总数
        edata = {
            'etype': th.tensor(np.array(etypes), dtype=th.int32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
            'ts': th.tensor(np.array([ts_max]*n_edges), dtype=th.float32),
        }

        self.graph.add_edges(src, dst, data=edata) # dgl 的一个函数，向有向图中添加多条边。
        '''
        u 是源节点（source nodes）的列表。
        v 是目标节点（destination nodes）的列表。
        data 是一个可选参数，可以是一个字典，包含边的属性，如特征或标签。
        weight 是一个可选参数，用于指定边的权重，通常用于加权图。
        '''
        print(self.graph.in_degrees().float().mean())

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_seq_id = self.user_ids[index] # 根据索引获取用户的序列id
        c_id, p, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = self.user_seq_dict[user_seq_id] # 根据ID获取用户的行为序列数据
        u_id = self.item_seq_dict[user_seq_id] #获取与用户信息相关联的物品
        seq_len = len(c_id) 

        #build graph
        # content_ids = content_ids[1:]
        # parts = parts[1:]
        label = ans_c[1:] - 1
        label = np.clip(label, 0, 1)
        
        answer_correct = ans_c[:-1] # 表示除了最后一个元素之外，对每个内容都回答正确
        # ques_had_explian = ques_had_explian[1:]
        user_answer = u_ans[:-1] # 除了最后一个元素的所有元素，表示用户的答案

        target_item_id = c_id[-1] # 获取最后一个元素作为目标物品ID
        label = label[-1] 

        #build graph
        # 根据用户序列ID计算用户节点的索引，并设置物品节点的索引为目标项目id
        u_idx, i_idx = int(user_seq_id.split('_')[0])+config.TOTAL_EXE, target_item_id
        # 计算用户节点和物品节点的邻居索引
        u_neighbors, i_neighbors = u_id+config.TOTAL_EXE, c_id[:-1]
        
        # 过滤掉用户节点和物品节点中的目标索引，避免自环。
        u_neighbors = u_neighbors[u_neighbors!=i_idx]
        i_neighbors = i_neighbors[i_neighbors!=u_idx]

        # 创建子图
        subgraph = get_subgraph_label(graph = self.graph,
                                      u_node_idx=th.tensor([u_idx]), 
                                      i_node_idx=th.tensor([i_idx]), 
                                      u_neighbors=th.tensor(u_neighbors), 
                                      i_neighbors=th.tensor(i_neighbors),           
                                    )
        

        return subgraph, th.tensor(label, dtype=th.float32)

# 将多个图对象打包成一个批次图G ，以便DataLoader使用
def collate_data(data):
    # 将data中的元组解包成两个数组连连个列表
    g_list, label_list = map(list, zip(*data)) #map 函数将一个函数应用到一个可迭代对象的每个元素上
    # list 函数应用到 zip(*data) 的结果上。这意味着 list 函数将被应用到 zip(*data) 生成的每个元组上，将它们转换成列表。
    g = dgl.batch(g_list) # 将g_list中的多个图像打包成一个批次图g，用于批量处理
    g_label = th.stack(label_list) # stack 函数将 label_list 中的所有标签堆叠成一个一维的张量 g_label。
    # 这个张量包含了所有图对象对应的标签。
    return g, g_label

# 定义了一个名为 get_dataloader 的函数，它负责加载数据、创建图数据集对象，并为训练和测试准备 DataLoader。
# 用于加载训练和测试数据集，并创建相应的DataLoader对象
# 接受数据路径、批量大小、工作线程和序列长度作为参数
def get_dataloader(data_path='ednet', batch_size=128, num_workers=8, seq_len=64):
    with open(f'./data/{data_path}/part_matrix.pkl', 'rb') as pick: # 部分信息
        # 使用picke模块加载文件，序列化的对象.plk
        part_matrix = pickle.load(pick)
    with open(f"data/{data_path}/tag_coo_matrix.pkl", 'rb') as pick: # 标签信息
        tag_coo_matrix = pickle.load(pick)


    train_df = pd.read_csv(f'data/{data_path}/train_df.csv') # 读取测试和训练数据的csv文件
    test_df = pd.read_csv(f'data/{data_path}/test_df.csv')

    # 加载训练用户组的信息
    with open(f"data/{data_path}/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    # 用户物品组
    with open(f"data/{data_path}/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    # 使用训练用户组、训练物品组、训练数据框 train_df 和之前加载的 part_matrix、tag_coo_matrix 
    # 创建 KT_Sequence_Graph 对象，它是图数据集的一个实例。
    train_seq_graph = KT_Sequence_Graph(train_user_group, train_item_group, df=train_df, 
                                           part_matrix=part_matrix, tag_coo_matrix=tag_coo_matrix, seq_len=seq_len)
    # 创建dataLoader对象，用于训练图数据集
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)
    
    # 加载验证用户组和物品组的信息
    with open(f"data/{data_path}/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open(f"data/{data_path}/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(val_user_group, val_item_group, df=test_df, 
                                          part_matrix=part_matrix, tag_coo_matrix=tag_coo_matrix, seq_len=seq_len)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader


if __name__=="__main__":
    train_loader, _ = get_dataloader(data_path='ednet',
                                     batch_size=32, 
                                     num_workers=8,
                                     seq_len=32
                                     )
    for subg, label in train_loader:
        print(subg)
        print(subg.edata['ts'])
        print(label)
        break

