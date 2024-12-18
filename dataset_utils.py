import cv2
from skimage import io
import numpy as np
import os
import glob
from tqdm import tqdm

def save_image(img_path_read, data_dir, category, cat_it, dataset_name, iteration_no, phase, resolution):
    image = io.imread(img_path_read)
    crop = crop_and_pad(image, size=resolution)
    img_name = os.path.join(data_dir, phase, '_'.join(['category', category, 'category_number', str(cat_it), 'dataset', dataset_name, str(iteration_no)])) + '.png'
    image = image.astype('uint8')
    io.imsave(img_name, crop)

def crop_and_pad(img, size=224.0):
    h, w = img.shape[:2]
    if h > w:
        scale = size / h
        new_h, new_w = int(size), int(w * scale)
    else:
        scale = size / w
        new_h, new_w = int(h * scale), int(size)
    resized = cv2.resize(img, (new_w, new_h))
    pad_h = (size - new_h) // 2
    pad_w = (size - new_w) // 2
    return cv2.copyMakeBorder(resized, int(pad_h), int(size - new_h - pad_h),
                              int(pad_w), int(size - new_w - pad_w), cv2.BORDER_CONSTANT, value=[0, 0, 0])

def save_data(root_path, dataset_dir, dataset_exp_name, images_folder_name, datasets, C, C_dash, train_val_split, resolution):
    data_dir = os.path.join(root_path, dataset_dir, dataset_exp_name, images_folder_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)

    categories = list(C) + list(C_dash)
    train_iteration_no, val_iteration_no = 0, 0

    for cat_it, category in tqdm(enumerate(categories), total=len(categories)):
        for dataset_name in datasets:
            # Corrected glob path
            imgs_path = np.array(glob.glob(os.path.join(root_path, dataset_dir, dataset_name, category, '*.*')))
            print(f"Searching in: {os.path.join(root_path, dataset_dir, dataset_name, category)}")
            print(f"Found {len(imgs_path)} images for category '{category}' in dataset '{dataset_name}'")

            if len(imgs_path) == 0:
                print(f"No images found for {category}. Skipping...")
                continue

            np.random.shuffle(imgs_path)
            split_pos = int(train_val_split * len(imgs_path))

            imgs_path_train = imgs_path[:split_pos]
            imgs_path_val = imgs_path[split_pos:]

            for img_path in imgs_path_train:
                print(f"Saving train image: {img_path}")
                save_image(img_path, data_dir, category, cat_it, dataset_name, train_iteration_no, 'train', resolution)
                train_iteration_no += 1

            for img_path in imgs_path_val:
                print(f"Saving val image: {img_path}")
                save_image(img_path, data_dir, category, cat_it, dataset_name, val_iteration_no, 'val', resolution)
                val_iteration_no += 1

    print(f"Dataset saved in {data_dir}")

