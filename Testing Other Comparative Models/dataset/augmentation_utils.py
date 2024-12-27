import cv2
import numpy as np
import os
from skimage import io
import torchvision.transforms as T

chop_distances = {}

def get_chop_distance(rotated_mat, angle):
    global chop_distances
    if angle in chop_distances:
        return chop_distances[angle]

    x = 0
    y = 0
    if angle > 0:
        while rotated_mat[y, x, 0] == 0:
            y += 1
    else:
        x = rotated_mat.shape[1] - 1
        while rotated_mat[y, x, 0] == 0:
            y += 1

    chop_distances[angle] = y
    return y

def rotateImage(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    d = get_chop_distance(rotated_mat, angle)
    return cv2.resize(rotated_mat[d:-d, d:-d], (224, 224))

def random_horizontal_flip(image, always_flip=True):
    if always_flip or np.random.rand() > 0.5:
        return np.fliplr(image), True
    return image, False

def rgb_flip(img, reorder):
    flipped = img.copy()
    flipped[:, :, 0] = img[:, :, reorder[0]]
    flipped[:, :, 1] = img[:, :, reorder[1]]
    flipped[:, :, 2] = img[:, :, reorder[2]]
    return flipped

def color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
    transform = T.ColorJitter(brightness, contrast, saturation, hue)
    pil_img = T.ToPILImage()(img)
    return np.array(transform(pil_img))

def random_crop(image, cropshape=224, padsize=20):
    H, W, C = image.shape
    pH = pW = padsize
    padded_image = np.zeros((H + 2 * pH, W + 2 * pW, C), dtype=image.dtype)
    padded_image[pH:pH + H, pW:pW + W] = image

    startx = np.random.randint(0, pW * 2)
    starty = np.random.randint(0, pH * 2)
    return padded_image[starty:starty + cropshape, startx:startx + cropshape]
