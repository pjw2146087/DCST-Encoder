import numpy as np
import time
import pandas as pd
import glob
from datetime import datetime
import os
def split_dataset(data, train_ratio, val_ratio, test_ratio, random_seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "划分比例之和必须等于1.0"
    gnss = data['gnss']
    adj = data['adj']
    num_samples = len(gnss)
    
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(int(time.time()))
    
    # 计算各个子集的样本数量
    num_test = int(test_ratio * num_samples)
    num_train = int(train_ratio * num_samples)
    
    num_val = num_samples - num_test - num_train
    
    # 创建打乱后的索引
    indices = np.random.permutation(num_samples)
    
    # 划分数据集
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    # 将划分后的索引降序排列
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)
    # 使用索引获取数据
    train_data_gnss = gnss.iloc[train_indices]
    val_data_gnss = gnss.iloc[val_indices]
    test_data_gnss = gnss.iloc[test_indices] 
    
    train_data_adj = adj[train_indices][:, train_indices]
    train_data_adj = np.argwhere(train_data_adj > 0)
    val_data_adj = adj[val_indices][:, val_indices]
    val_data_adj = np.argwhere(val_data_adj > 0)
    test_data_adj = adj[test_indices][:, test_indices]
    test_data_adj = np.argwhere(test_data_adj > 0)
    train_data = dict(
        gnss = train_data_gnss,
        adj = train_data_adj,
    )
    val_data = dict(
        gnss = val_data_gnss,
        adj = val_data_adj,
    )
    test_data = dict(
        gnss = test_data_gnss,
        adj = test_data_adj,
    )
    if test_ratio == 0:
        return train_data, val_data
    else:
        return train_data, val_data, test_data
def split_index(data, train_ratio, val_ratio, test_ratio, random_seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "划分比例之和必须等于1.0"
    gnss = data['gnss']
    adj = data['adj']
    num_samples = len(gnss)

    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(int(time.time()))

    # 计算各个子集的样本数量
    num_test = int(test_ratio * num_samples)
    num_train = int(train_ratio * num_samples)

    num_val = num_samples - num_test - num_train

    # 创建打乱后的索引
    indices = np.random.permutation(num_samples)

    # 划分数据集
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    # 将划分后的索引降序排列
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)
    if test_ratio == 0:
        return train_indices, val_indices
    else:
        return train_indices, val_indices, test_indices
    
    
def split_files(path,train_ratio, val_ratio, test_ratio, random_seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "划分比例之和必须等于1.0"
    path_gnss = path['gnss']
    path_adj = path['adj']
    gnssfilenames=sorted(glob.glob(os.path.join(path_gnss, "*.xlsx"))) if path_gnss is not None else None
    adjfilenames=sorted(glob.glob(os.path.join(path_adj, "*.npy"))) if path_adj is not None else None
    num_samples = len(gnssfilenames)
    
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(int(time.time()))
    
    # 计算各个子集的样本数量
    num_test = int(test_ratio * num_samples)
    num_train = int(train_ratio * num_samples)
    num_val = num_samples - num_test - num_train
    
    
    # 创建打乱后的索引
    indices = np.random.permutation(num_samples)
    
    # 划分数据集
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    # 使用索引获取数据
    train_filenames = dict(
        gnss = np.array(gnssfilenames)[train_indices].tolist(),
        adj = np.array(adjfilenames)[train_indices].tolist() if path_adj is not None else None,
    )
    val_filenames = dict(
        gnss = np.array(gnssfilenames)[val_indices].tolist(),
        adj = np.array(adjfilenames)[val_indices].tolist() if path_adj is not None else None,
    )
    test_filenames = dict(
        gnss = np.array(gnssfilenames)[test_indices].tolist(),
        adj = np.array(adjfilenames)[test_indices].tolist() if path_adj is not None else None,
    )
    if test_ratio == 0:
        return train_filenames, val_filenames
    else:
        return train_filenames, val_filenames, test_filenames
def get_coco(data,description = None,version = None,year = None,contributor = None,date_created = None,
             licenses = None,categories_info = None):
    #定义默认参数
    if description is None:
        description = "This is a coco gnss data"
    if version is None:
        version = "1.0"
    if year is None:
        # 获取当前时间
        current_time = datetime.now()

        # 提取年份
        current_year = current_time.year
        
        year = current_year
        date_created = current_time
    if contributor is None:
        contributor = 'Hero'
        
    # 定义info部分
    info = {
        "description": description,
        "version": version,
        "year": year,
        "contributor": contributor,
        "date_created": str(date_created)
    }
    # 定义一些常量
    if licenses is None:
        
        license_id = 1
        license_name = "Example License"

        # 创建licenses列表
        licenses = [
            {
                "id": license_id,
                "name": license_name,
            }
        ]

    # 定义要生成的元素数量
    if categories_info is None:
        categories_info = [
            {"id": 1, "name": "field"},
            {"id": 2, "name": "road"}
        ]
        
    num_trajectories = len(data['gnss']) if data['gnss'] is not None else 0
    num_adjs = len(data['adj'])  if data['adj'] is not None else 0
    # 创建trajectories列表
    trajectories = []
    for i in range(0, num_trajectories):
        trajectories.append({
            "id": i,
            "index": data['index'][i].tolist() if data['index'] is not None else None,
            "file_name": data['gnss'][i] if data['gnss'] is not None else None,
            "license": license_id,
        })

    # 创建adjs列表
    adjs = []
    for i in range(0, num_adjs):
        adjs.append({
            "id": i,
            "index": data['index'][i].tolist() if data['index'] is not None else None,
            "file_name": data['adj'][i] if data['gnss'] is not None else None,
            "license": license_id,
        })

    # 创建categories列表
    categories = []
    for category in categories_info:
        categories.append({
            "id": category["id"],
            "name": category["name"],
            "supercategory": ""
        })

    # 定义 COCO 数据集 JSON 数据
    coco_data = {
        "info": info,
        "licenses": licenses,
        "trajectories": trajectories,
        "adjs": adjs,
        "categories": categories
    }
    return coco_data