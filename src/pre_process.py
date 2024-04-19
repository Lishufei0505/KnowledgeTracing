"""
pre-processing ednet
对 ednet数据集进行预处理
"""


import time # 用于序列化时间
import pickle # 用于序列化python对象
import numpy as np # 用于数值计算
import pandas as pd # 用于数据处理 
from utils import get_time_lag 

from copy import copy

# 定义了一个字典类型，指定了不同数据字段的数据类型
data_type ={
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8"
}

def group_seq(df, groupby_key, cols, save_path):
    '''
    将DataFrame按照某个键值分组, 并将每组中的数据序列化保存到文件中。
    接收四个参数
    df(DataFrame): pandas中提供的二维表格型数据结构
    groupby_key: 用于分组的列名
    cols: 用于保留的列名列表
    save_path: 保存序列化数据的文件路径
    '''
    # build group : user_id(item_id) - seq
    # save as file
    # 创建cols副本，避免在后续操作中修改原始列表
    cols = copy(cols) # 创建的副本cols是局部变量，不会影响传递的参数，但最好不要用同一个名字
    cols.remove(groupby_key) # 从副本中移除groupby_key, 因为其用于分组，不作为序列化数据的一部分
    print(cols)
    group = df.groupby(groupby_key).apply(lambda df: tuple([df[c].values for c in cols]))
    # df.groupby 对DataFrame中的二维数据按键分组
    # apply 对每一个分组执行特定的函数
    # 遍历cols的每个列名c, 获取分组的列数据，并用.values转化为Numpy数组
    # 将每个分组的列数据打包成一个元组
    # lambda def 创建匿名函数，被groupby方法的apply()函数调用
 
    with open(save_path, 'wb') as pick: # 以二进制写入模式打开文件，pick是文件对象的别名
        pickle.dump(group, pick) # 利用pickle.dump函数将group字典序列化，写入之前打开的pick文件
    del group, df # 删除group和df变量，释放内存。
    return

def pre_process(train_path, ques_path, row_start=30e6, num_rows=30e6, split_ratio=0.8):
    '''
    用于预处理训练集和验证数据集
    主要用于数据清洗、特征提取、数据划分和序列化
    '''
    print("Start pre-process")
    t_s = time.time() # 记录初始时间
    # Features列表，包含用于从数据集中提取的特征列名
    Features = ["timestamp", "user_id", "content_id", "content_type_id", "task_container_id", "user_answer", 
                "answered_correctly", "prior_question_elapsed_time", "prior_question_had_explanation"]
    df = pd.read_csv(train_path)[Features] # 读取训练数据文件，并选择Features列表中的列
    df.index = df.index.astype('uint32')  # 将DataFrame的索引转换为无符号的32位整数

    # shift prior elapsed_time and had_explanation to make current elapsed_time and had_explanation
    ''' 数据清洗：规范化数据范围、处理缺失值、转换单位之间、确保数据类型的正确'''
    df = df[df.content_type_id == 0].reset_index() # 仅保留content_type_id为0的行
    # 过滤操作可能会导致索引不连续或者不完整，reset_index() 会创建一个新的连续索引
    df["prior_question_elapsed_time"].fillna(0, inplace=True)  # 将prior_question_elapsed_time列的缺失值nan替换成0
    df["prior_question_elapsed_time"] /= 1000 # convert to sec 时间从毫秒转换为秒
    df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].clip(0, 300) 
    # 将 "prior_question_elapsed_time" 列中的值限制在 0 到 300 秒之间。任何小于 0 的值都会被设置为 0，任何大于 300 的值都会被设置为 300。这可以用于数据规范化或确保时间值在合理的范围内。
    df["prior_question_had_explanation"].fillna(False, inplace=True) # 没有提供解释的情况
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].astype('int8')
    # 符号的 8 位整数类型。这通常是为了确保数据类型的一致性，并减少内存使用。
    
    # get time_lag feature
    print("Start compute time_lag")
    time_dict = get_time_lag(df)
    # with open("time_dict.pkl.zip", 'wb') as pick:
    #     pickle.dump(time_dict, pick)
    print("Complete compute time_lag")
    print("====================")

    df = df.sort_values(by=["timestamp"]) # 按照时间戳排序
    # train_df.drop("timestamp", axis=1, inplace=True)
    # train_df.drop("viretual_time_stamp", axis=1, inplace=True)

    print("Start merge dataframe")
    # merge with question dataframe to get part feature
    ques_df = pd.read_csv(ques_path)[["question_id", "part"]]  # 读取固定的两列
    df = df.merge(ques_df, how='left', left_on='content_id', right_on='question_id') 
    # df中的content_id列与ques_df中的question_id列将用于匹配和合并数据
    df.drop(["question_id"], axis=1, inplace=True) # 删除 question_id 列
    # axis = 0 表示行， axis = 1 表示列
    df["part"] = df["part"].astype('uint8')
    print(df.head(10)) # 打印前10条数据
    print("Complete merge dataframe")
    print("====================")

    # plus 1 for cat feature which starts from 0
    # 这些特征都从0开始，全部+1
    df["content_id"] += 1
    df["task_container_id"] += 1
    df["answered_correctly"] += 1
    df["prior_question_had_explanation"] += 1
    df["user_answer"] += 1

    Train_features = ["user_id", "content_id", "part", "task_container_id", "time_lag", "prior_question_elapsed_time",
                      "answered_correctly", "prior_question_had_explanation", "user_answer", "timestamp"]

    if num_rows == -1: # 如果行数 == -1，设置成dataFrame的函数 
        num_rows = df.shape[0]
    row_start = 0
    num_rows = df.shape[0]
    # df = df.iloc[int(row_start):int(row_start+num_rows)]
    val_df = df[int(num_rows*split_ratio):] # 后半段作为验证集
    train_df = df[:int(num_rows*split_ratio)] # 前半段作为训练集

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    # Check data balance 检查数据的平衡性
    # 用于验证集中有多少独特的用户是训练集中没出现过的
    num_new_user = val_df[~val_df["user_id"].isin(train_df["user_id"])]["user_id"].nunique()
    #  val_df[~val_df["user_id"].isin(train_df["user_id"])] 筛选出训练集中不存在的用户ID对应的行
    num_new_content = val_df[~val_df["content_id"].isin(train_df["content_id"])]["content_id"].nunique()
    train_content_id = train_df["content_id"].nunique()
    train_part = train_df["part"].nunique() # 统计唯一值的数量
    train_correct = train_df["answered_correctly"].mean()-1 # 所有问题的平均正确率
    val_correct = val_df["answered_correctly"].mean()-1
    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}/ Number of part {}".format(train_content_id, train_part))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")

    print("Start train and Val grouping")

    df.to_csv('data/test_df.csv')
    train_df.to_csv('data/train_df.csv')


    # 利用group_seq 对训练集和验证集中的数据进行分组，将结果就行序列化  同一个用户 和 同一个物品 两种
    group_seq(df=train_df, groupby_key="user_id", cols=Train_features, save_path="data/train_user_group.pkl.zip")
    group_seq(df=train_df, groupby_key="content_id", cols=Train_features, save_path="data/train_item_group.pkl.zip")

    group_seq(df=val_df, groupby_key="user_id", cols=Train_features, save_path="data/val_user_group.pkl.zip")
    group_seq(df=df, groupby_key="content_id", cols=Train_features, save_path="data/val_item_group.pkl.zip")
    
    print("Complete pre-process, execution time {:.2f} s".format(time.time()-t_s))

if __name__=="__main__":
    train_path = "data/train_30m.csv"
    ques_path = "data/questions.csv"
    # be aware that appropriate range of data is required to ensure all questions 
    # are in the training set, or LB score will be much lower than CV score
    # Recommend to user all of the data.
    pre_process(train_path, ques_path, 0, -1, 0.8)