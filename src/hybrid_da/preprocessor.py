import pdb
import torch
import torch.nn as nn
from hybrid_da.data_transforms import MinMaxScaler

class Preprocessor(nn.Module):

    def __init__(
        self,
        transform_type: str = "min_max",
        no_transform_vars: list = [],
    ) -> None:
        super().__init__()

        self.transform_type = transform_type
        self.no_transform_vars = no_transform_vars

        self.first_call = True

    
    def _get_data_shapes(self, data: dict) -> dict:
        shapes = {}
        for key, value in data.items():
            shapes[key] = value.shape

        return shapes


    def fit(self, data: dict) -> dict:
        if self.first_call:
            self.first_call = False

            self.data_shapes = self._get_data_shapes(data)

            self.scalers = {}
            for key, value in data.items():
                if key not in self.no_transform_vars:
                    self.scalers[key] = MinMaxScaler()

        for key, value in data.items():
            if key not in self.no_transform_vars:
                self.scalers[key].fit(value)

    def transform(self, data: dict) -> dict:
        for key, value in data.items():
            if key not in self.no_transform_vars:
                data[key] = self.scalers[key].transform(value)

        return data
    
    def inverse_transform(self, data: dict) -> dict:
        for key, value in data.items():
            if key not in self.no_transform_vars:
                data[key] = self.scalers[key].inverse_transform(value)

        return data
    
