"""
pre-process tag, part 
"""
from collections import defaultdict
import time
import pickle
import numpy as np
import pandas as pd
from copy import copy

from scipy.sparse import coo_matrix, save_npz

import config


ASSIST_TAGS = 102 # 标签的最大数量

def tag_to_vector(tag):
    tag_lsit = [0]*ASSIST_TAGS # 创建一个长度为ASSIST_TAGS的列表，初始化为0
    if isinstance(tag, float): # 如果传入的tag是浮点数类型，直接返回一个全0数据
        return np.array(tag_lsit)
    tag_lsit[tag]=1 # 对应位置的元素设置为1，创建一个稀疏向量
    return np.array(tag_lsit) # 转换为numpy数组并返回


if __name__=="__main__":
    ques_path = "data/assist/questions.csv"
    # question_id,bundle_id,correct_answer,part,tags

    df = pd.read_csv(ques_path)
    df['tag_vector'] = df['tags'].apply(tag_to_vector)
    tag_matrix = df['tag_vector'].tolist() # 将向量转化为列表
    tag_matrix = np.stack(tag_matrix, axis=0) # 将列表中的每一行向量堆叠成一个numpy数组
    tag_coo_matrix = np.matmul(tag_matrix, tag_matrix.T) # 计算matrix与转置的乘积，得到共线矩阵。
    print(tag_coo_matrix.sum()) 

    n_items = 3162

    item_tag_coo_matrix = defaultdict(dict) # 创建默认字典，用于存储项目标签的共现矩阵
    for i in range(n_items):
        for j in range(n_items):
            v = tag_coo_matrix[i][j]
            if v != 0:
                item_tag_coo_matrix[i][j]=v    

    with open('./data/assist/tag_coo_matrix.pkl', 'wb') as pick:
        pickle.dump(item_tag_coo_matrix, pick)

    parts = df['part'].tolist()

    # 存储项目部分的共现矩阵
    item_part_matrix=defaultdict(dict)
    for i in range(n_items):
        for j in range(n_items):
            if parts[i]==parts[j]:
                item_part_matrix[i][j]=1

    with open('./data/assist/part_matrix.pkl', 'wb') as pick:
        pickle.dump(item_part_matrix, pick)
