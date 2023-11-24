import torch
import torch.nn as nn
from hybrid_da.preprocessor import Preprocessor
from hybrid_da.dataset import XarrayDataset
import pdb

data_folder = 'data/raw'
dataset = XarrayDataset(data_folder, num_samples=4)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=False, 
    drop_last=False
)

feature_preprocessor = Preprocessor(no_transform_vars=["x_y_z", "time"])
target_preprocessor = Preprocessor()


original_features = {
    "porosity": {'min': 1e12, 'max': -1e12},
    "permeability": {'min': 1e12, 'max': -1e12},
    "youngs_modulus": {'min': 1e12, 'max': -1e12},
    "poissons_ratio": {'min': 1e12, 'max': -1e12},
    "injection_rate": {'min': 1e12, 'max': -1e12},
    "x_y_z": {'min': 1e12, 'max': -1e12},
    "time": {'min': 1e12, 'max': -1e12},
}

original_targets = {
    "pressure": {'min': 1e12, 'max': -1e12},
    "co2_molar_fraction": {'min': 1e12, 'max': -1e12},
    "displacement_x": {'min': 1e12, 'max': -1e12},
    "displacement_y": {'min': 1e12, 'max': -1e12},
    "displacement_z": {'min': 1e12, 'max': -1e12},
}
for i, (features, targets) in enumerate(dataloader):

    feature_preprocessor.fit(features)
    target_preprocessor.fit(targets)

    for key, value in features.items():
        if key not in feature_preprocessor.no_transform_vars:
            if value.min() < original_features[key]['min']:
                original_features[key]['min'] = value.min()
                
            if value.max() > original_features[key]['max']:
                original_features[key]['max'] = value.max()

    for key, value in targets.items():
        if key not in target_preprocessor.no_transform_vars:
            if value.min() < original_targets[key]['min']:
                original_targets[key]['min'] = value.min()
                
            if value.max() > original_targets[key]['max']:
                original_targets[key]['max'] = value.max()

   
def test_preprocessor_fitting():
    for key, value in original_features.items():
        if key not in feature_preprocessor.no_transform_vars:
            assert value['min'] == feature_preprocessor.scalers[key].min_val

    for key, value in original_features.items():
        if key not in feature_preprocessor.no_transform_vars:
            assert value['max'] == feature_preprocessor.scalers[key].max_val

    for key, value in original_targets.items():
        if key not in target_preprocessor.no_transform_vars:
            assert value['min'] == target_preprocessor.scalers[key].min_val

    for key, value in original_targets.items():
        if key not in target_preprocessor.no_transform_vars:
            assert value['max'] == target_preprocessor.scalers[key].max_val

transformed_features = {
    "porosity": {'min': 1e12, 'max': -1e12},
    "permeability": {'min': 1e12, 'max': -1e12},
    "youngs_modulus": {'min': 1e12, 'max': -1e12},
    "poissons_ratio": {'min': 1e12, 'max': -1e12},
    "injection_rate": {'min': 1e12, 'max': -1e12},
    "x_y_z": {'min': 1e12, 'max': -1e12},
    "time": {'min': 1e12, 'max': -1e12},
}
transformed_targets = {
    "pressure": {'min': 1e12, 'max': -1e12},
    "co2_molar_fraction": {'min': 1e12, 'max': -1e12},
    "displacement_x": {'min': 1e12, 'max': -1e12},
    "displacement_y": {'min': 1e12, 'max': -1e12},
    "displacement_z": {'min': 1e12, 'max': -1e12},
}
def test_transform_preprocessor():

    for i, (features, targets) in enumerate(dataloader):
        features = feature_preprocessor.transform(features)
        targets = target_preprocessor.transform(targets)
                
        
        for key, value in features.items():
            if key not in feature_preprocessor.no_transform_vars:
                if value.min() < transformed_features[key]['min']:
                    transformed_features[key]['min'] = value.min()
                    
                if value.max() > transformed_features[key]['max']:
                    transformed_features[key]['max'] = value.max()

        for key, value in targets.items():
            if key not in target_preprocessor.no_transform_vars:
                if value.min() < transformed_targets[key]['min']:
                    transformed_targets[key]['min'] = value.min()
                    
                if value.max() > transformed_targets[key]['max']:
                    transformed_targets[key]['max'] = value.max()

    for key in transformed_features.keys():
        if key not in feature_preprocessor.no_transform_vars:
            assert transformed_features[key]['min'] >= 0.0
            assert transformed_features[key]['min'] <= 1.0

            assert transformed_features[key]['max'] >= 0.0
            assert transformed_features[key]['max'] <= 1.0

            assert transformed_features[key]['min'] <= transformed_features[key]['max']

    
def test_inverse_transform_preprocessor():
    
    for i, (features, targets) in enumerate(dataloader):
        features_transform = feature_preprocessor.transform(features)
        targets_transform = target_preprocessor.transform(targets)

        features_transform = feature_preprocessor.inverse_transform(features_transform)
        targets_transform = target_preprocessor.inverse_transform(targets_transform)

        for key, value in features_transform.items():
            assert torch.allclose(value, features[key])
        for key, value in targets_transform.items():
            assert torch.allclose(value, targets[key])


'''
for i, (features, targets) in enumerate(dataloader):
    feature_preprocessor.fit(features)
    target_preprocessor.fit(targets)

for i, (features, targets) in enumerate(dataloader):
    features = feature_preprocessor.transform(features)
    targets = target_preprocessor.transform(targets)

print()
print('#'*80)
print()

print("FEATURES:")
for key, value in features.items():
    print(f"{key}: {value.shape}, min: {value.min()}, max: {value.max()}")
print("TARGETS:")
for key, value in targets.items():
    print(f"{key}: {value.shape}, min: {value.min()}, max: {value.max()}")


print()
print('#'*80)
print()
features = feature_preprocessor.inverse_transform(features)
targets = target_preprocessor.inverse_transform(targets)
print("FEATURES:")
for key, value in features.items():
    print(f"{key}: {value.shape}, min: {value.min()}, max: {value.max()}")
print("TARGETS:")
for key, value in targets.items():
    print(f"{key}: {value.shape}, min: {value.min()}, max: {value.max()}")
'''