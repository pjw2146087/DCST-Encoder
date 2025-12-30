from  . import Dataset
from  ..datareader import FieldRoadDataReader
# from ..dataaugmenter import Compose,DataBalancer,Gaussian_Noise,Uniform_Noise,Pulse_Noise,get_statistics_dict
import numpy as np
import pandas as pd
__all__ = [
    "TraceDataset"
]
class TraceDataset(Dataset):
    def __init__(self, path, mode='train', num_workers = 0,max_len = 5000, drop_rate = 0.01):
        assert mode in ['train', 'valid', 'test'], 'mode is one of train, eval ,test.'
        self.mode = mode
        self.data = []
        self.adjs = {}

        if mode == 'train' or mode == 'valid':
            fileiter=FieldRoadDataReader(path,dataset_format = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate)
            for cropped_points,cropped_labels,trace_id,cropped_adjs in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,labels,adj in zip(cropped_points,cropped_labels,cropped_adjs): 
                    self.data.append((points, labels, trace_id, adj))
                
        else:
            fileiter=FieldRoadDataReader(path,dataset_format = 'Trace',num_workers = num_workers,max_len = max_len, drop_rate = drop_rate)
            for cropped_points,trace_id,cropped_adjs in fileiter:
                self.adjs[str(trace_id)] = cropped_adjs
                for points,adj in zip(cropped_points,cropped_adjs): 
                    self.data.append((points,trace_id, adj))

    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            points, labels, trace_id , adj= self.data[index]
            return points, labels, adj
        else:
            points, trace_id ,adj= self.data[index]
            return points, adj ,trace_id

    def __getadj__(self, traid):
        return self.adjs[str(traid)]

    def __len__(self):
        return len(self.data)
    
