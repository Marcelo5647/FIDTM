import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

'''change your path'''
root = '/srv/storage/datasets/marcelo/CityPark/'

train_data_path = os.path.join(root, 'test_data', 'images')

if not os.path.exists(train_data_path.replace('images', 'gt_fidt_maps')):
    os.makedirs(train_data_path.replace('images', 'gt_fidt_maps'))

if not os.path.exists(train_data_path.replace('images', 'gt_show')):
    os.makedirs(train_data_path.replace('images', 'gt_show'))

img_paths = []
for img_path in glob.glob(os.path.join(train_data_path, '*.png')):
    img_paths.append(img_path)

img_paths.sort()


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map


for img_path in img_paths:
    print(img_path)
    fidt_map_path = img_path.replace('.png', '.npy').replace('images', 'gt_fidt_maps')
    if not os.path.isfile(fidt_map_path):
        Img_data = cv2.imread(img_path)

        dot_map = np.load(img_path.replace('.png', '.npy').replace('images', 'gt_dots'))
        points_tuple = np.where(dot_map==1)
        Gt_data = [[y_coord, x_coord] for x_coord, y_coord in zip(points_tuple[0], points_tuple[1])]

        fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

        np.save(fidt_map_path, fidt_map1)
