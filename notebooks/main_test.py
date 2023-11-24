import hybrid_da
import xarray as xr
import sys
import pathlib
import os
import torch

from hybrid_da.dataset import XarrayDataset    

data_folder = 'data/raw/'
dataset = XarrayDataset(data_folder)

features, targets = dataset.__getitem__(0)

print(features.keys())
print(targets.keys())
print(features["porosity"].shape)
print(features["permeability"].shape)
print(features["youngs_modulus"].shape)
print(features["poissons_ratio"].shape)
print(features["injection_rate"].shape)
print(features["time"].shape)
print(features["x_y_z"][0].shape)
print(features["x_y_z"][1].shape)
print(features["x_y_z"][2].shape)
print(targets["pressure"].shape)
print(targets["co2_molar_fraction"].shape)
print(targets["displacement_x"].shape)
print(targets["displacement_y"].shape)
print(targets["displacement_z"].shape)