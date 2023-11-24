import torch
import yaml

# open yml file
with open("configs/variable_names.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


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

class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        variable_config: dict,
    ) -> None:
        super().__init__()

        self.variable_config = variable_config
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 
