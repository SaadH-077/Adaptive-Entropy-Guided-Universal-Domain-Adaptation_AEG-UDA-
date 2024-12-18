from augmentation_utils import rotateImage, random_horizontal_flip, rgb_flip, color_jitter, random_crop
import os
import glob
from tqdm import tqdm
from skimage import io
import numpy as np

def augment_images(images_path):
    print('\nAugmenting Images...')
    for img_path in tqdm(images_path):
        image = io.imread(img_path)

        # Flip
        flipped_img, success = random_horizontal_flip(image)
        if success:
            save_augmented_image(img_path, 'flip', flipped_img)

        # Rotate
        for angle in [-15, -10, -5, 5, 10, 15]:
            rotated_img = rotateImage(image, angle)
            save_augmented_image(img_path, f'rotate{angle}', rotated_img)

        # RGB Flip
        for order in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]:
            rgb_flipped_img = rgb_flip(image, order)
            save_augmented_image(img_path, f'rgbflip{"".join(map(str, order))}', rgb_flipped_img)

        # Color Jitter
        for idx in range(3):
            jittered_img = color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            save_augmented_image(img_path, f'jitter{idx}', jittered_img)

def save_augmented_image(original_path, prefix, augmented_img):
    folder, filename = os.path.split(original_path)
    new_filename = f"{prefix}_{filename}"
    save_path = os.path.join(folder, new_filename)
    io.imsave(save_path, augmented_img)

# Path to source and target images
source_images_path = glob.glob('usfda_office_31_DtoA/source_images/train/*.png')
target_images_path = glob.glob('usfda_office_31_DtoA/target_images/train/*.png')

# Run augmentation
augment_images(source_images_path)
augment_images(target_images_path)