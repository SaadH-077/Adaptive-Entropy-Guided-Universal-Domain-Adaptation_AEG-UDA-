import matplotlib
import os
import sys
import numpy as np
from skimage import io
from tqdm import tqdm
import dataset_utils
#Server root path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from glob import glob

import torch
# gpu = 1
# torch.cuda.set_device(gpu)

from natsort import natsorted
import torch
import re

# 2-way or 3-way splicing
n_way_splice = 2

#Place where all data is stored
# dataset_dir = 'data/digits'
dataset_dir = 'OFFICE31'

dataset_exp_names = ['usfda_office_31_DtoA']
datasets_sources = [['dslr']]
datasets_targets = ['amazon']

C = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
Cs_dash = ['bike_helmet', 'bookcase', 'bottle', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'letter_tray', 'mobile_phone', 'paper_notebook']
Ct_dash = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

#number of shared classes between source and target
num_shared_classes = len(C)

#number of unknown classes in target domain
num_unknown_target_classes = len(Cs_dash)

#number of unknown classes in source domain
num_unknown_source_classes = len(Ct_dash)

# Negative categories
num_source_classes = len(C) + len(Cs_dash)
temp_negative_category_dict = {}
c = 0
for i in range(num_source_classes):
	for j in range(i+1, num_source_classes):
		temp_negative_category_dict[(i, j)] = c
		c += 1

negative_category_dict = {}

for key in temp_negative_category_dict:
	negative_category_dict[key] = temp_negative_category_dict[key]
	negative_category_dict[(key[1], key[0])] = temp_negative_category_dict[key]

# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is set to:", device)


def generate_spline(num_points, points = None, vertical = False, integer = False):

	'''
		Performs spline interpolation between points, and returns num_points coordinates.
		Args:
			points: list of (y,x) coordinates. If none, then samples from num_points
			vertical: True if points are y coordinates (False if points are x coordinates)
			integer: True if the function should return integer points
	'''

	HMAX, WMAX = num_points, num_points
	window = 30
	center = (HMAX/2, WMAX/2)

	rand_center_point = (np.random.randint(low=int(center[0]-window), high=int(center[0]+window)), np.random.randint(low=int(center[1]-(window/3)), high=int(center[1]+(window/3))))
	
	points = [(0, np.random.randint(HMAX)), rand_center_point, (WMAX-1, np.random.randint(HMAX))]

	px = [p[0] for p in points]
	px.extend([0, 223])

	py = [p[1] for p in points]
	py.extend([0, 223])

	# spl = spline_interpolation([p[0] for p in points], [p[1] for p in points], range(num_points), order=7, kind='smoothest')
	spl = interp1d([p[0] for p in points], [p[1] for p in points], kind='quadratic')(range(num_points))

	x = list(range(num_points))
	y = [p for p in spl]

	if integer:
		x = [int(a) for a in x]
		y = [int(a) for a in y]
	
	if not(vertical):
		return (x, y), [px, py]
	else:
		return (y, x), [py, px]


def get_negative_image(image1, image2, spline = None, vertical_division = False):

	'''
		Returns 2 negative images by merging 2 images horizontally or vertically (vertical_division).
	'''

	if not(vertical_division):
		
		mask1 = torch.ones(image1.shape).to(device) # Image to the left of spline
		for (x, y) in spline:
			mask1[y:, x] = 0
		mask2 = 1 - mask1 # Image to the right spline

		# print(mask1)
		# print(mask2)

		# io.imsave('temp_mask1.jpg', (mask1.cpu().numpy() * 255).astype(np.uint8))
		# io.imsave('temp_mask2.jpg', (mask2.cpu().numpy() * 255).astype(np.uint8))

		mask1 = mask1.float()
		mask2 = mask2.float()

		newim1 = mask1 * image1 + mask2 * image2
		newim2 = mask1 * image2 + mask2 * image1

		return newim1, newim2, mask1, mask2

	else:

		mask1 = torch.ones(image1.shape).to(device) # Image above spline
		for (x, y) in spline:
			mask1[y, x:] = 0
		mask2 = 1 - mask1 # Image below spline

		# print(mask1)
		# print(mask2)

		# io.imsave('temp_mask1.jpg', (mask1.cpu().numpy() * 255).astype(np.uint8))
		# io.imsave('temp_mask2.jpg', (mask2.cpu().numpy() * 255).astype(np.uint8))

		mask1 = mask1.float()
		mask2 = mask2.float()

		newim1 = mask1 * image1 + mask2 * image2
		newim2 = mask1 * image2 + mask2 * image1

		return newim1, newim2, mask1, mask2


def merge_images(image1, image2):

	assert image1.shape == image2.shape

	image1, image2 = torch.from_numpy(image1).to(device), torch.from_numpy(image2).to(device)
	image1, image2 = image1.float(), image2.float()

	vert = False
	(x, y), _ = generate_spline(image1.shape[0], vertical=vert, integer=True)
	spline = list(zip(x, y))
	hor_I1, hor_I2, hor_mask1, hor_mask2 = get_negative_image(image1, image2, spline, vertical_division=vert)

	vert = True
	(x, y), _ = generate_spline(image1.shape[0], vertical=vert, integer=True)
	spline = list(zip(x, y))
	ver_I1, ver_I2, ver_mask1, ver_mask2 = get_negative_image(image1, image2, spline, vertical_division=vert)

	return hor_I1, hor_I2, ver_I1, ver_I2, hor_mask1, hor_mask2, ver_mask1, ver_mask2


def get_negative_image_3(image1, image2, image3, spline_vert, spline_hor):

	'''
		Merges 3 images, in a 3-way splicing.

		spline_vert -> vertical spline (going from top to bottom edge)
		spline_hor -> horizontal spline (going from left to right edge)

		image1 -> class A
		image2 -> class A
		image3 -> class B

		The following 4 scenarios can arise

			A | A   	A |       	  | A     	  B
		1)  -----   2)	--| B   3)	B |--   4)	-----
			  B     	A |       	  | A   	A | A

		The corresponding operations are:

		1) spline_vert_left * image1 + spline_vert_right * image2 + spline_hor_bottom * image3
		2) spline_hor_top * image1 + spline_hor_bottom * image2 + spline_vert_right * image3
		3) spline_hor_top * image1 + spline_hor_bottom * image2 + spline_vert_left * image3
		4) spline_vert_left * image1 + spline_vert_right * image2 + spline_hor_top * image3

		Out of these, some will not have enough representation from one class
	'''

	spline_vert_left = torch.ones(image1.shape).to(device) # Image to the left of spline
	for (x, y) in spline_vert:
		spline_vert_left[y, x:] = 0
	spline_vert_right = 1 - spline_vert_left # Image to the right spline

	spline_hor_top = torch.ones(image1.shape).to(device) # Image to the top of spline
	for (x, y) in spline_hor:
		spline_hor_top[y:, x] = 0
	spline_hor_bottom = 1 - spline_hor_top # Image to the bottom of spline

	I1 = spline_hor_top * (spline_vert_left * image1 + spline_vert_right * image2) + spline_hor_bottom * image3
	I2 = spline_vert_left * (spline_hor_top * image1 + spline_hor_bottom * image2) + spline_vert_right * image3
	I3 = spline_vert_right * (spline_hor_top * image1 + spline_hor_bottom * image2) + spline_vert_left * image3
	I4 = spline_hor_bottom * (spline_vert_left * image1 + spline_vert_right * image2) + spline_hor_top * image3

	M1 = spline_hor_top * (spline_vert_left * torch.ones(image1.shape).to(device) * 32 + spline_vert_right * torch.ones(image1.shape).to(device) * 64) + spline_hor_bottom * torch.ones(image1.shape).to(device) * 192
	M2 = spline_vert_left * (spline_hor_top * torch.ones(image1.shape).to(device) * 32 + spline_hor_bottom * torch.ones(image1.shape).to(device) * 64) + spline_vert_right * torch.ones(image1.shape).to(device) * 192
	M3 = spline_vert_right * (spline_hor_top * torch.ones(image1.shape).to(device) * 32 + spline_hor_bottom * torch.ones(image1.shape).to(device) * 64) + spline_vert_left * torch.ones(image1.shape).to(device) * 192
	M4 = spline_hor_bottom * (spline_vert_left * torch.ones(image1.shape).to(device) * 32 + spline_vert_right * torch.ones(image1.shape).to(device) * 64) + spline_hor_top * torch.ones(image1.shape).to(device) * 192

	return I1, I2, I3, I4, M1, M2, M3, M4


def merge_images_3(image1, image2, image3):

	(x, y), _ = generate_spline(image1.shape[0], vertical=True, integer=True)
	spline_vert = list(zip(x, y))

	(x, y), _ = generate_spline(image1.shape[0], vertical=False, integer=True)
	spline_hor = list(zip(x, y))

	image1, image2, image3 = torch.from_numpy(image1).to(device), torch.from_numpy(image2).to(device), torch.from_numpy(image3).to(device)
	image1, image2, image3 = image1.float(), image2.float(), image3.float()

	return get_negative_image_3(image1, image2, image3, spline_vert, spline_hor)


def get_category_number(c1, c2):

	'''
		Returns the category number given two classes.
	'''

	return negative_category_dict[(c1, c2)]


for dataset_exp_name, datasets_source, datasets_target in tqdm(list(zip(dataset_exp_names, datasets_sources, datasets_targets))):

	print('running', dataset_exp_name)

	print('shared_classes: {}'.format(C))
	print('source_private_classes: {}'.format(Cs_dash))
	print('target_private_classes: {}'.format(Ct_dash))

	# filenames = glob(os.path.join(server_root_path, dataset_dir, dataset_exp_name, 'source_images', 'train', '*.png'))
	filenames = glob(os.path.join(dataset_exp_name, 'source_images', 'train', '*.png'))
	savepath = os.path.join(os.getcwd(), 'negative_images')
	savepath_mask = os.path.join(os.getcwd(), 'negative_masks')

	if os.path.exists(savepath):
		os.system('rm -rf ' + savepath)
	os.mkdir(savepath)

	if os.path.exists(savepath_mask):
		os.system('rm -rf ' + savepath_mask)
	os.mkdir(savepath_mask)

	# Remove augmented files
	# filenames = [a for a in filenames if a.split('/')[-1].split('_')[0]=='category']
	L = len(filenames)
	print('Number of images:', L)

	counter = 0

	class_wise_files = {}

	if n_way_splice == 3:
		for cnum in range(num_source_classes):
			class_wise_files[str(cnum)] = []
			pattern = re.compile('_category_number_' + str(cnum) + '_dataset_' + str(datasets_source[0]) + '_')
			for fname in filenames:
				if pattern.search(fname) is not None:
					class_wise_files[str(cnum)].append(fname)

	for i in tqdm(range(5000)):

		if n_way_splice == 2:

			im1 = np.random.randint(L)
			im2 = np.random.randint(L)

			f1, f2 = filenames[im1], filenames[im2]
			c1, c2 = int(f1.split('_')[-4]), int(f2.split('_')[-4])

			while c1 == c2:
				im2 = np.random.randint(L)
				f2 = filenames[im2]
				c2 = int(f2.split('_')[-4])

			image1, image2 = io.imread(f1), io.imread(f2)

			a, b, c, d, e, f, g, h = merge_images(image1, image2)
			merged_images = [a, b, c, d]
			merged_masks = [e, f, g, h]

			cnum = get_category_number(c1, c2)

			for image, mask in zip(merged_images, merged_masks):
				counter += 1
				save_filename = 'category_' + str(cnum) + '_category_number_' + str(cnum) + '_dataset_' + str(datasets_source[0]) + '_' + str(counter) + '.png'
				save_filename_mask = 'mask_category_' + str(cnum) + '_category_number_' + str(cnum) + '_dataset_' + str(datasets_source[0]) + '_' + str(counter) + '.png'
				io.imsave(os.path.join(savepath, save_filename), image.cpu().numpy().astype(np.uint8))
				io.imsave(os.path.join(savepath_mask, save_filename_mask), (mask * 255).cpu().numpy().astype(np.uint8))

		elif n_way_splice == 3:
			
			# first image
			im1 = np.random.randint(L)
			f1 = filenames[im1]
			c1 = int(f1.split('_')[-4])

			# second image
			im2 = np.random.randint(len(class_wise_files[str(c1)]))
			f2 = class_wise_files[str(c1)][im2]
			c2 = c1

			while f1 == f2:
				im2 = np.random.randint(len(class_wise_files[str(c1)]))
				f2 = class_wise_files[str(c1)][im2]

			# third image
			x = np.random.randint(len(list(class_wise_files.keys())))
			c3 = int((class_wise_files.keys()[x]))

			while c3 == c2:
				x = np.random.randint(len(list(class_wise_files.keys())))
				c3 = int((class_wise_files.keys()[x]))

			im3 = np.random.randint(len(class_wise_files[str(c3)]))
			f3 = class_wise_files[str(c3)][im3]

			image1, image2, image3 = io.imread(f1), io.imread(f2), io.imread(f3)
			i1, i2, i3, i4, m1, m2, m3, m4 = merge_images_3(image1, image2, image3)
			merged_images = [i1, i2, i3, i4]
			merged_masks = [m1, m2, m3, m4]

			cnum = get_category_number(c1, c3)

			for image, mask in zip(merged_images, merged_masks):
				counter += 1
				save_filename = 'category_' + str(cnum) + '_category_number_' + str(cnum) + '_dataset_' + str(datasets_source[0]) + '_' + str(counter) + '.png'
				save_filename_mask = 'mask_category_' + str(cnum) + '_category_number_' + str(cnum) + '_dataset_' + str(datasets_source[0]) + '_' + str(counter) + '.png'
				io.imsave(os.path.join(savepath, save_filename), image.cpu().numpy().astype(np.uint8))
				io.imsave(os.path.join(savepath_mask, save_filename_mask), (mask).cpu().numpy().astype(np.uint8))

		# break