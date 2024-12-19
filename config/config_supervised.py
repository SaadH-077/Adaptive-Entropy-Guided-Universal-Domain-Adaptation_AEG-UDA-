import os
from root_path import server_root_path
from glob import glob
from natsort import natsorted

import matplotlib
# matplotlib.use('Agg')

import torch

# Settings are passed through this dictionary
settings = {}

# Paths of weights and summaries to be saved
settings['weights_path'] = os.path.join(server_root_path, 'weights')
settings['summaries_path'] = os.path.join(server_root_path, 'summaries')

# For supervised training. Set to false, if Train_supervised.py should not be executed.
settings['running_supervised'] = True

# Maximum number of iterations
settings['start_iter'] = 1
settings['max_iter'] = 500
settings['val_after'] = 50

# Label set relationships
settings['C'] = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
settings['Cs_dash'] = ['bike_helmet', 'bookcase', 'bottle', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'letter_tray', 'mobile_phone', 'paper_notebook']
settings['Ct_dash'] = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

settings['num_C'] = len(settings['C'])
settings['num_Cs_dash'] = len(settings['Cs_dash'])
settings['num_Ct_dash'] = len(settings['Ct_dash'])
settings['num_Cs'] = settings['num_C'] + settings['num_Cs_dash']
settings['num_Ct'] = settings['num_C'] + settings['num_Ct_dash']

# Batch Size and number of samples per iteration
settings['batch_size'] = 64
settings['num_positive_samples'] = 32
settings['num_negative_samples'] = 32
settings['num_positive_images'] = settings['batch_size']
settings['num_negative_images'] = settings['batch_size']

# Model parameters
settings['cnn_to_use'] = 'resnet50'
settings['Fs_dims'] = 256
settings['softmax_temperature'] = 1
settings['online_augmentation_90_degrees'] = True # Used for online rotations in the data loader
settings['val_aug_imgs_mean_before_softmax'] = False
settings['val_aug_imgs_mean_after_softmax'] = True

# Loading weights and experiment name. Change the experiment name here, to save the weights with the 
# corresponding exp_name. The weights of this can then be loaded into another experiment, by setting
# load_exp_name.
settings['load_weights'] = False
settings['load_exp_name'] = 'None'
settings['exp_name'] = 'usfda_office_31_DtoA'

# Define optimizers for the various losses.
settings['optimizer'] = {
	'classification': ['M', 'Fs', 'Cs', 'Cn'], # Classification loss
	'pos_img_recon': ['Fs', 'G'], # Image cyclic reconstruction loss
	'pos_sample_recon': ['Fs', 'G'], # Feature cyclic reconstruction loss
	'logsoftmax': ['Fs'], # Loss over PDF values of features (to make the clusters compact and separated)
}

settings['use_loss'] = {
	'classification': True,
	'pos_img_recon': True,
	'pos_sample_recon': True,
	'logsoftmax': True,
}

settings['losses_after_enough_iters'] = ['logprob', 'logsoftmax', 'pos_sample_recon']

settings['classification_weight'] = [1, 0.2] # Hyperparameter alpha -> second element of the list.

settings['to_train'] = {
	'M': False, # Frozen Resnet-50
	'Fs': True,
	'Ft': False,
	'G': True,
	'Cs': True,
	'Cn': True,
}
settings['lr'] = 1e-4 # 1e-3 default

# settings['gpu'] = 0
# settings['device'] = 'cuda:' + str(settings['gpu'])
# torch.cuda.set_device(settings['gpu'])

if settings['load_weights']:
	best_weights = natsorted(glob(os.path.join(settings['weights_path'], settings['load_exp_name'], '*.pth')))[-1]
	settings['load_weights_path'] = best_weights

settings['dataset_exp_name'] = 'office_31_dataset/usfda_office_31_DtoA'
settings['dataset_path'] = os.path.join(server_root_path, 'data', settings['dataset_exp_name'], 'index_lists') 
settings['negative_data_path'] = os.path.join(server_root_path, 'data', settings['dataset_exp_name'], 'negative_images') 
settings['negative_mask_path'] = os.path.join(server_root_path, 'data', settings['dataset_exp_name'], 'negative_masks')