from pathlib import Path
from typing import Any

import numpy as np
import torch
import pdb
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import yaml

torch.set_default_dtype(torch.float32)

# open yml file
with open("configs/variables.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

'''
X_VARIABLE_NAME = config['variable_names']['x']
Y_VARIABLE_NAME = config['variable_names']['y']
Z_VARIABLE_NAME = config['variable_names']['z']
POROSITY_VARIABLE_NAME = config['variable_names']['porosity']
PERMEABILITY_VARIABLE_NAME = config['variable_names']['permeability']
YOUNGS_MODULUS_VARIABLE_NAME = config['variable_names']['youngs_modulus']
POISSONS_RATIO_VARIABLE_NAME = config['variable_names']['poissons_ratio']
INJECTION_RATE_VARIABLE_NAME = config['variable_names']['injection_rate']
PRESSURE_VARIABLE_NAME = config['variable_names']['pressure']
DISPLACEMENT_X_VARIABLE_NAME = config['variable_names']['displacement_x']
DISPLACEMENT_Y_VARIABLE_NAME = config['variable_names']['displacement_y']
DISPLACEMENT_Z_VARIABLE_NAME = config['variable_names']['displacement_z']
CO2_MOLAR_FRACTION_VARIABLE_NAME = config['variable_names']['co2_molar_fraction']
'''

def get_xarray_data(data_path: Path) -> xr.core.dataset.Dataset:
    return xr.load_dataset(data_path, engine='netcdf4')

def get_x_y_z(data: xr.core.dataset.Dataset) -> torch.Tensor:

    x = torch.tensor(data[X_VARIABLE_NAME].values, dtype=torch.get_default_dtype())
    y = torch.tensor(data[Y_VARIABLE_NAME].values, dtype=torch.get_default_dtype())
    z = torch.tensor(data[Z_VARIABLE_NAME].values, dtype=torch.get_default_dtype())

    x_y_z = torch.stack([x, y, z], dim=0)

    return x_y_z

def get_variable(
    data: xr.core.dataset.Dataset,
    variable_names: str,
) -> torch.Tensor:
    return torch.tensor(data[variable_names].values, dtype=torch.get_default_dtype())

def get_porosity(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[POROSITY_VARIABLE_NAME].values)

def get_permeability(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[PERMEABILITY_VARIABLE_NAME].values)

def get_youngs_modulus(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[YOUNGS_MODULUS_VARIABLE_NAME].values)

def get_poissons_ratio(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[POISSONS_RATIO_VARIABLE_NAME].values)

def get_injection_rate(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[INJECTION_RATE_VARIABLE_NAME].values)

def get_pressure(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[PRESSURE_VARIABLE_NAME].values)

def get_displacement_x(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[DISPLACEMENT_X_VARIABLE_NAME].values)

def get_displacement_y(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[DISPLACEMENT_Y_VARIABLE_NAME].values)

def get_displacement_z(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[DISPLACEMENT_Z_VARIABLE_NAME].values)

def get_co2_molar_fraction(data: xr.core.dataset.Dataset) -> torch.Tensor:
    return torch.Tensor(data[CO2_MOLAR_FRACTION_VARIABLE_NAME].values)


class XarrayDataset(Dataset[Any]):
    """
    A torch dataset that loads data from an xarray dataset.

    From a path to a folder it loads the xarray correspoonding to the id of the individual xarray. 

    The items it returns are two dictionaries, one for the features and one for the targets.

    Features consits of the following variables:
        - porosity (Nx, Ny, Nz)
        - permebility (Nx, Ny, Nz)
        - youngs_modulus (Nx, Ny, Nz)
        - poissons_ratio (Nx, Ny, Nz)
        - injection_rate (Nt)
        - time (Nt)
        - x_y_z (Nx, Ny, Nz)

    Target consits of the following variables:
        - pressure (Nx, Ny, Nz, Nt)
        - displacement_x (Nx, Ny, Nz, Nt)
        - displacement_y (Nx, Ny, Nz, Nt)
        - displacement_z (Nx, Ny, Nz, Nt)
        - co2_molar_fraction (Nx, Ny, Nz, Nt)
    """

    def __init__(
        self,
        data_folder: Path,
        variable_config: dict,
        num_samples: int = 10,  
    ):
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.variable_config = variable_config

    def __len__(self):
        return self.num_samples
    
    def _get_features(self, data: xr.core.dataset.Dataset) -> dict:

        for name, variable_name in self.variable_config['features'].items():
            globals()[name] = get_variable(data, variable_name)

        x_y_z = torch.stack([x, y, z], dim=0)

        time = torch.arange(0, injection_rate.shape[0])

        x[x.isnan()] = 0
        y[y.isnan()] = 0
        z[z.isnan()] = 0
        porosity[porosity.isnan()] = 0
        permeability[permeability.isnan()] = 0
        youngs_modulus[youngs_modulus.isnan()] = 0
        poissons_ratio[poissons_ratio.isnan()] = 0
        injection_rate[injection_rate.isnan()] = 0
        
        features = {
            "porosity": porosity,
            "permeability": permeability,
            "youngs_modulus": youngs_modulus,
            "poissons_ratio": poissons_ratio,
            "injection_rate": injection_rate,
            "x_y_z": x_y_z,
            "time": time,
        }

        return features

    def _get_targets(self, data: xr.core.dataset.Dataset) -> dict:

        for name, variable_name in self.variable_config['targets'].items():
            globals()[name] = get_variable(data, variable_name)
        
        # Replace all NaNs with ones
        pressure[torch.isnan(pressure)] = 1
        co2_molar_fraction[torch.isnan(co2_molar_fraction)] = 1
        displacement_x[torch.isnan(displacement_x)] = 1
        displacement_y[torch.isnan(displacement_y)] = 1
        displacement_z[torch.isnan(displacement_z)] = 1

        targets = {
            "pressure": pressure,
            "co2_molar_fraction": co2_molar_fraction,
            "displacement_x": displacement_x,
            "displacement_y": displacement_y,
            "displacement_z": displacement_z,
        }

        return targets

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        data_path =  f"{self.data_folder}/sample_{idx}.nc"
        data = get_xarray_data(data_path)

        features = self._get_features(data)

        targets = self._get_targets(data)

        return features, targets


if __name__ == "__main__":

    data_folder = 'data/raw'
    dataset = XarrayDataset(data_folder)

    features, targets = dataset.__getitem__(0)

    print("FEATURES:")
    for key, value in features.items():
        print(f'{key}, size: {value.shape}, dtype: {value.dtype}')

    print()
    print("TARGETS:")
    for key, value in targets.items():
        print(f'{key}, size: {value.shape}, dtype: {value.dtype}')