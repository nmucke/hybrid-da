import torch
import yaml

class SpatialEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 

class CoordinateEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return

class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        variable_config: dict,
    ) -> None:
        super().__init__()

        self.variable_config = variable_config



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 
