import os
import glob
import math
import scipy.io as io
import numpy as np
import sys

'''please set your dataset path'''
root = '/srv/storage/datasets/marcelo/CityPark/'

dataset_test = os.path.join(root, 'test_data', 'images')
path_sets = [dataset_test]

if not os.path.exists(dataset_test):
    sys.exit("The path is wrong, please check the dataset path.")


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.png')):
        img_paths.append(img_path)

img_paths.sort()

f = open('./city_park_gt.txt', 'w+')
k = 1
for img_path in img_paths:

    print(img_path)
    gt_dot_map_path = img_path.replace('.png', '.npy').replace('images', 'gt_dots')
    dot_map = np.load(gt_dot_map_path)

    indices = np.floor(np.where(dot_map == 1)).astype(int)
    
    f.write('{} {} '.format(k, len(indices[0])))

    for y_index, x_index in zip(indices[0], indices[1]):

        sigma_s = 4
        sigma_l = 8
        f.write('{} {} {} {} {} '.format(math.floor(y_index), math.floor(x_index), sigma_s, sigma_l, 1))
    f.write('\n')

    k = k + 1
f.close()
