import os
from tqdm import tqdm
import dataset_utils

# Define the local root path
local_root_path = os.getcwd()

# Place where all data is stored
dataset_dir = 'OFFICE31'

dataset_exp_names = ['usfda_office_31_DtoA']
datasets_sources = [['dslr']]
datasets_targets = ['amazon']

C = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
Cs_dash = ['bike_helmet', 'bookcase', 'bottle', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'letter_tray', 'mobile_phone', 'paper_notebook']
Ct_dash = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

# Shared, source-private, and target-private class details
for dataset_exp_name, datasets_source, datasets_target in tqdm(list(zip(dataset_exp_names, datasets_sources, datasets_targets))):
    print(f'Running experiment: {dataset_exp_name}')

    resolution = 224
    source_train_val_split = 0.8
    target_train_val_split = 0.8

    # exp_dir = os.path.join(local_root_path, dataset_dir, dataset_exp_name)
    # os.makedirs(exp_dir, exist_ok=True)

    print(f"Shared classes: {C}")
    print(f"Source-private classes: {Cs_dash}")
    print(f"Target-private classes: {Ct_dash}")

    # Create Source Data
    dataset_utils.save_data(local_root_path, dataset_dir, dataset_exp_name, 'source_images', datasets_source, C, Cs_dash, source_train_val_split, resolution)

    # Create Target Data
    dataset_utils.save_data(local_root_path, dataset_dir, dataset_exp_name, 'target_images', [datasets_target], C, Ct_dash, target_train_val_split, resolution)
