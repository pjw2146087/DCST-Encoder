import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time
import numpy as np
from models.Pretrain import Pretrain_Serial
from dataset import GraphDataset
from utils.utils import WarmupCosineLR
from fieldroaddatapipeline.dataloader import FieldRoadDataLoader
from torch_geometric.data import Data

train_path=dict(
    gnss="../dataset/dataset_high/wheat_large/sampled_wheat_43",#换数据集在这换名字
    adj="../dataset/dataset_high/wheat_large/sampled_wheat_adj",#换数据集在这换名字
    json="../dataset/dataset_high/wheat_large/Non-Identically_Distributed_Coco/sampled_wheat_43_train.json"#换数据集在这换名字
)
train_dataset = GraphDataset(train_path,mode='train', num_workers=0,max_len=1000,drop_rate = 0)
valid_path=dict(
    gnss="../dataset/dataset_high/wheat_large/sampled_wheat_43",#换数据集在这换名字
    adj="../dataset/dataset_high/wheat_large/sampled_wheat_adj",#换数据集在这换名字
    json="../dataset/dataset_high/wheat_large/Non-Identically_Distributed_Coco/sampled_wheat_43_valid.json"#换数据集在这换名字
)
valid_dataset = GraphDataset(valid_path,mode='valid',num_workers=0,max_len=1000,drop_rate = 0)
test_path=dict(
    gnss="../dataset/dataset_high/wheat_large/sampled_wheat_43",#换数据集在这换名字
    adj="../dataset/dataset_high/wheat_large/sampled_wheat_adj",#换数据集在这换名字
    json="../dataset/dataset_high/wheat_large/Non-Identically_Distributed_Coco/sampled_wheat_43_valid.json"#换数据集在这换名字
)
test_dataset = GraphDataset(test_path,mode='test',num_workers=0,max_len=1000,drop_rate = 0)

# Create data loaders using PyTorch DataLoader
train_loader = FieldRoadDataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
valid_loader = FieldRoadDataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=True)
test_loader = FieldRoadDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
####################################超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.autograd.set_detect_anomaly(True)
total_epochs =1000
# Set the random seed if needed
#torch.manual_seed(2023)
# Initialize your model, optimizer, and LR scheduler
# 创建模型
model = Pretrain_Serial(
    img_size=43,
    patch_size=1,
    in_chans=1,
    num_classes=2,
    embed_dim=180,
    num_heads=9,
    depth=1,
    decoder_embed_dim=180, 
    decoder_depth=1, 
    decoder_num_heads=9,
    norm_pix_loss=False,
).to(device)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.0005, 
    weight_decay=0.0005
)
scheduler = WarmupCosineLR(
    optimizer,                    
    warmup_start_lr=0.0001,                    
    end_lr=0.00005,                     
    warmup_epochs=5,                     
    total_epochs=total_epochs
)

##########################################训练
train_losses = []
valid_losses = []
best_valid_loss = 1000
total_train_time=0
total_valid_time=0
total_test_time=0
for pass_num in range(total_epochs):
    model.train()
    epoch_start_time = time.time()
    train_loss_total = 0.0
    num_samples = 0
    for batch_id, (points, labels,adjs,trace_id) in enumerate(train_loader()):
        points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
        labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
        adjs = adjs.clone().detach().to(torch.float32).squeeze(0).to(device)
        # 找到邻接矩阵中非零元素的索引
        rows, cols = torch.nonzero(adjs, as_tuple=True)
        # 按照源节点和目标节点的顺序构建新的张量
        edge_index = torch.stack([rows, cols]).to(device)
        data = Data(x=points, edge_index=edge_index, y=labels)
        loss= model.train_step(data, mask_ratio_image=0.65, mask_ratio_graph=0.5,optimizer=optimizer,loss_config=None)
        trajectory_length = points.shape[0]
        train_loss_total += loss * trajectory_length
        num_samples += trajectory_length
    avg_train_loss = train_loss_total / num_samples
    train_losses.append(avg_train_loss)
    train_end_time = time.time()
    epoch_train_time = train_end_time - epoch_start_time
    total_train_time += epoch_train_time
    train_fps = num_samples / epoch_train_time
    scheduler.step()
    with torch.no_grad():
        model.eval()
        valid_loss_total = 0.0
        num_samples = 0
        for batch_id, (points, labels,adjs,trace_id) in enumerate(valid_loader()):
            points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
            labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
            adjs = adjs.clone().detach().to(torch.float32).squeeze(0).to(device)
            # 找到邻接矩阵中非零元素的索引
            rows, cols = torch.nonzero(adjs, as_tuple=True)
            # 按照源节点和目标节点的顺序构建新的张量
            edge_index = torch.stack([rows, cols]).to(device)
            data = Data(x=points, edge_index=edge_index, y=labels)
            loss = model.valid_step(data, mask_ratio_image=0.65, mask_ratio_graph=0.5,loss_config=None)
            trajectory_length = points.shape[0]
            valid_loss_total += loss * trajectory_length
            num_samples += trajectory_length
        avg_valid_loss = valid_loss_total / num_samples
        valid_losses.append(avg_valid_loss)
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), './weights/pre_model.pt')
            final_loss = True
        valid_end_time = time.time()
        epoch_valid_time = valid_end_time - train_end_time
        total_valid_time += epoch_valid_time
        valid_fps = num_samples / epoch_valid_time

        test_loss_total = 0.0
        num_samples = 0
        for batch_id, (points, labels,adjs,trace_id) in enumerate(test_loader()):
            points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
            labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
            adjs = adjs.clone().detach().to(torch.float32).squeeze(0).to(device)
            # 找到邻接矩阵中非零元素的索引
            rows, cols = torch.nonzero(adjs, as_tuple=True)
            # 按照源节点和目标节点的顺序构建新的张量
            edge_index = torch.stack([rows, cols]).to(device)
            data = Data(x=points, edge_index=edge_index, y=labels)
            loss = model.test_step(data, mask_ratio_image=0.65, mask_ratio_graph=0.5,loss_config=None)
            trajectory_length = points.shape[0]
            test_loss_total += loss * trajectory_length
            num_samples += trajectory_length
        avg_test_loss = test_loss_total / num_samples
        test_end_time = time.time()
        epoch_test_time = test_end_time - valid_end_time
        total_test_time += epoch_test_time
        test_fps = num_samples / epoch_test_time

        if final_loss is not None:
            final_pretrain_result = str(dict(loss=avg_test_loss))
# Calculate average training and evaluation times
avg_train_time = total_train_time / total_epochs
avg_valid_time = total_valid_time / total_epochs
avg_test_time = total_test_time / total_epochs
