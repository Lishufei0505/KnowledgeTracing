'''
    从Google Drive下载指定的数据集文件
    并将他们保存到本地目录
'''
import gdown # 用于从Goole Drive下载文件的Python库
import argparse # 参数解析器模块，用于构建命令行参数和选项
from itertools import chain # 将多个迭代器连接成一个长的迭代器
import os 

# 定义 assist_files_dict 字典，包含了 assist 数据集的文件路径和对应的Google Drive文件ID
assist_files_dict ={
    'assist/assist.csv' : '1YWuiE2wYhepN7P6Jo51mproW7_sEANrO',
    'assist/questions.csv' : '1JaZDZC0JmOqdS5g1WJt24_sZDt_art-N',
}

ednet_files_dict = {
    'ednet/train_30m.csv' : '1XlTBPBFYEzzy4dUhAYFC78mKGnXmz9X_',
    'ednet/questions.csv' : '1drZV1NkJDuufGkIUDJdeeyW7O61XnWkm',
}

# 检查该脚本是否作为主程序运行
if __name__ == '__main__':
    print ("kaishizhixing !!!")
    parser = argparse.ArgumentParser() # 用于解析命令行参数
    parser.add_argument("-d","--dataset", type=str, default="all") # 添加命令行参数
    parser.add_argument("-p","--path", type=str, default="./data")
    args = parser.parse_args() # 解析命令行参数

    # 循环创建3个子目录，用于存放下载的数据文件
    for p in ['assist', 'ednet']: # 'junyi'没用到直接删除
        path = f'{args.path}/{p}'  # f-string 格式化字符串变量
        # os.makedirs() 函数会在指定路径下创建目录，如果目录已存在则不会抛出错误
        os.makedirs(path, exist_ok=True)
    print("makedir done!!!")
    if args.dataset == 'all':
        # chain 将多个可迭代对象连接成一个可迭代对象
        # dict.item（）字典中的键值对
        # dict() 将可迭代对象转化为字典
        dataset_files_dict = dict(chain(assist_files_dict.items(), ednet_files_dict.items()))
    if args.dataset == 'assist':
        dataset_files_dict = assist_files_dict
    if args.dataset == 'ednet':
        dataset_files_dict = ednet_files_dict
    for output, url  in dataset_files_dict.items():
        # gdown.download 需要有两个参数
        # 函数的第一个参数是一个格式化字符串，它构建了一个指向 Google Drive 上特定文件的 URL。url 变量的值被插入到 URL 模板中。
        # 第二个参数是下载文件后保存的本地路径，其中 args.path 是用户通过命令行参数指定的目录路径，
        # output 是文件名。quiet=False 参数表示在下载过程中显示输出信息，例如下载进度。
        gdown.download(f'https://drive.google.com/uc?id={url}', f'{args.path}/{output}', quiet=False)
        print("done!")