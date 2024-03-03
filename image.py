import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import cv2


def load_data_fidt(img_path, args, train=True):
    gt_fidt_path = img_path.replace('.png', '.npy').replace('images', 'gt_fidt_maps')
    gt_loc_path = img_path.replace('.png', '.npy').replace('images', 'gt_dots')
    
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            k = np.load(gt_loc_path)
            fidt_map = np.load(gt_fidt_path)
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
