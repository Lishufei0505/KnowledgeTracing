import math, copy

import dgl
import pandas as pd
import numpy as np
import pickle as pkl

import torch as th
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import get_logger, get_args_from_yaml
from data_generator import get_dataloader
from data_generator_assist import get_dataloader_assist
import config


from models.igmc import IGMC
from models.igkt import IGKT_TS
from models.igakt import IGAKT


def evaluate(model, loader, device):
    '''
    model: 要评估的模型
    loader: 包含测试数据的DataLoader
    device: 指定模型或数据应该运行的设备
    '''
    # Evaluate AUC, ACC
    model.eval()
    val_labels = [] # 初始化空列表用于存储真实标签
    val_preds = [] # 用于存储模型的预测结果
    for batch in loader: # 遍历loader中的所有批次数据
        with th.no_grad():
            # 对输入数据进行预测，将结果存储在preds变量中
            preds = model(batch[0].to(device)) # 获取当前批次的输入数据，并将其发送到指定的devide
        labels = batch[1].to(device) # 将真实标签发送到device
        val_labels.extend(labels.cpu().tolist()) # 将真实标签从设备内从移动到cpu内存，并转换为Python列表，添加到val_labels列表中
        val_preds.extend(preds.cpu().tolist())
    # 因为评估函数在cpu中运行，所以利用.cpu方法将pytorch张量从GPU移动到CPU
    # 评估函数的输入是python列表，不能是tensor，评估函数需要一个可迭代的连续的数据结构来计算性能指标。
    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, val_acc


def train_epoch(model, optimizer, loader, device, logger, log_interval):
    '''
    logger: 用于记录训练日志的对象
    log_interval: 多久记录一次
    '''
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []
    mse_loss_fn = nn.MSELoss().to(device)
    bce_loss_fn = nn.BCELoss().to(device)

    # enumerate 将一个可迭代对象转换成一个枚举对象，一个包含索引和值的yuanzu序列
    for iter_idx, batch in enumerate(loader, start=1): # 指定索引从1开始
        t_start = time.time()

        inputs = batch[0].to(device) # 输入数据
        labels = batch[1].to(device) 
        preds = model(inputs) 
        loss = mse_loss_fn(preds, labels) + bce_loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)


NUM_WORKER = 16
def train(args:EasyDict, train_loader, test_loader, logger):
    # args 中包含训练参数的EasyDict对象


    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    ### prepare data and set model
    in_feats = (args.hop+1)*2  
    if args.model_type == 'IGMC':
        model = IGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'IGKT_TS':
        model = IGKT_TS(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.model_type == 'IGAKT':
        model = IGAKT(in_nfeats=in_feats,
                     in_efeats=2, 
                     latent_dim=args.latent_dims,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))
        
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    
    best_epoch = 0
    best_auc, best_acc = 0, 0

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    # 定义训练周期的范围
    epochs = list(range(1, args.train_epochs+1))

    eval_func_map = { # 定义评估函数的映射
        'IGMC': evaluate,
    }
    eval_func = eval_func_map.get(args.model_type, evaluate) 
    # .get查找与特定键（args.model_type）关联的值，如果没有提供默认值，evaluate
    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
    
        train_loss = train_epoch(model, optimizer, train_loader, 
                                 args.device, logger, 
                                 log_interval=args.log_interval
                                 )
        test_auc, test_acc = eval_func(model, test_loader, args.device)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'test_auc': test_auc,
            'test_acc': test_acc,

        }
        logger.info('=== Epoch {}, train loss {:.6f}, test auc {:.6f}, test acc {:.6f} ==='.format(*eval_info.values()))

        # lr_decay_step是一个整数，表示每隔多少个周期调整一下学习率
        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_auc < test_auc:
            logger.info(f'new best test auc {test_auc:.6f} acc {test_acc:.6f} ===')
            best_epoch = epoch_idx
            best_auc = test_auc
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            # 创建并存储当前模型状态的深拷贝
            # model.state_dict（）获取当前模型的参数状态
            # best_state 变量用于在训练结束时保存最佳模型

        
    th.save(best_state, f'./parameters/{args.key}_{args.data_name}_{best_auc:.4f}.pt')
    logger.info(f"Training ends. The best testing auc is {best_auc:.6f} acc {best_acc:.6f} at epoch {best_epoch}")
    return test_auc
    
import yaml
from collections import defaultdict
from datetime import datetime

DATALOADER_MAP = {
    'assist':get_dataloader_assist,
    'ednet':get_dataloader,
}

def main():
    while 1:
        with open('./train_configs/train_list.yaml') as f:
            # yaml.load 加载yaml文件内容，如果文件中是一个字典，就会被解析成字典，是序列则解析成列表
            # yaml.FullLoader确保在加载的过程中进行完整的安全检查
            files = yaml.load(f, Loader=yaml.FullLoader) 
        file_list = files['files']
        for f in file_list:
            date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            args = get_args_from_yaml(f)
            logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
            logger.info('train args')
            for k,v in args.items(): # 遍历args字典中的所有键值对
                logger.info(f'{k}: {v}') # 记录每个参数的名称和值到日志中

            best_lr = None 
            sub_args = args # 将args字典复制到sub_args，以便在训练过程中修改参数
            best_rmse_list = [] # 空列表，用于存储每次训练的最佳RMSE值

            dataloader_manager = DATALOADER_MAP.get(sub_args.dataset)
            train_loader, test_loader = dataloader_manager(batch_size=sub_args.batch_size, 
                                                                num_workers=NUM_WORKER,
                                                                seq_len=sub_args.max_seq
                                                           )

            for lr in args.train_lrs:
                sub_args['train_lr'] = lr
                best_rmse = train(sub_args, train_loader, test_loader, logger=logger)
                best_rmse_list.append(best_rmse)
            
            logger.info(f"**********The final best testing RMSE is {min(best_rmse_list):.6f} at lr {best_lr}********")
            logger.info(f"**********The mean testing RMSE is {np.mean(best_rmse_list):.6f}, {np.std(best_rmse_list)} ********")
        
        
if __name__ == '__main__':
    main()