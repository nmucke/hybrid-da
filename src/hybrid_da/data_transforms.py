import torch
import torch.nn as nn
import pdb

class MinMaxScaler(nn.Module):
    
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.min_val = 1e12
        self.max_val = -1e12

    def fit(self, features: torch.Tensor):
        batch_min = features.min()
        batch_max = features.max()

        if batch_min < self.min_val:
            self.min_val = batch_min
        if batch_max > self.max_val:
            self.max_val = batch_max
        
    def transform(self, data: torch.Tensor):

        lol = (data - self.min_val) / (self.max_val - self.min_val)
        # check if any of the values are smaller than 0.0 or larger than 1.0                               
        #assert torch.all(data[key] <= 1.0)

        return lol
    
    def inverse_transform(self, data: torch.Tensor):
        return data * (self.max_val - self.min_val) + self.min_val
    
