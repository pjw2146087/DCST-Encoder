import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from models.Encoder import DC_encoder
from dataset import GraphDataset
from fieldroaddatapipeline.dataloader import FieldRoadDataLoader
import time
from collections import defaultdict
#如果已经按照说明在其他数据集上训练模型，使用并跟换上述数据集类型
# Define paths for the test dataset
'''test_path=dict(
    gnss="../dataset/dataset_high/wheat_large/sampled_wheat_43",
    adj="../dataset/dataset_high/wheat_large/sampled_wheat_adj",
    json="../dataset/dataset_high/wheat_large/Non-Identically_Distributed_Coco/sampled_wheat_43_test.json"
) '''
#如果已经按照说明在其他数据集上训练模型，使用上面的test_path
test_path=dict(
    gnss="test samples/sampled_wheat_43",
    adj="test samples/sampled_wheat_adj",
    json="test samples/Non-Identically_Distributed_Coco/sampled_wheat_43_test.json"
)
# Create the test dataset and dataloader
test_dataset = GraphDataset(test_path, mode='predict', num_workers=0, max_len=1000, drop_rate=0)
test_loader = FieldRoadDataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DC_encoder(
    img_size=43, 
    patch_size=1, 
    in_chans=1, 
    num_classes=2, 
    embed_dim=180, 
    depth=1,               
    num_heads=9, 
    mlp_ratio=4., 
    qkv_bias=True,              
    drop_rate=0.6, 
    attn_drop_rate=0.6, 
    drop_path_rate=0.4
).to(device)

# Load the trained model weights
model.load_state_dict(torch.load('./weights/model.pt'), strict=False)
# Start testing
model.eval()
all_predictions = []
all_labels = []
total_test_time = 0
trace_results = defaultdict(list)
num_samples = 0
with torch.no_grad():
    for batch_id, (points, labels, adjs, trace_id, coordinates) in enumerate(test_loader()):
        points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
        labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
        adjs = adjs.clone().detach().to(torch.float32).squeeze(0).to(device)
        coordinates = coordinates.clone().detach().to(torch.float64).squeeze().to(device)
        trace_id = trace_id[0]
        # 找到邻接矩阵中非零元素的索引
        rows, cols = torch.nonzero(adjs, as_tuple=True)
        # 按照源节点和目标节点的顺序构建新的张量
        edge_index = torch.stack([rows, cols]).to(device)
        data = Data(x=points, edge_index=edge_index, y=labels)
        predicts = model.test_step(data)
        predicts = torch.softmax(predicts, dim=1)
        trace_results[trace_id].append((points.cpu().numpy(), predicts.cpu().numpy(), labels.cpu().numpy(), coordinates.cpu().numpy()))

# Summarize results per trace_id
for trace_id, results in trace_results.items():
    points, predicts, labels, coordinates = zip(*results)
    
    points = np.concatenate(points, axis=0)
    predicts = np.concatenate(predicts, axis=0)
    labels = np.concatenate(labels, axis=0)
    coordinates = np.concatenate(coordinates, axis=0)
    prelabels = np.argmax(predicts, axis=1)
    class_result = classification_report(labels, prelabels, digits=4)
