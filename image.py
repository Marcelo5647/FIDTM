import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2


def load_data_fidt(img_path, args, train=True):
    gt_fidt_path = img_path.replace('.png', '.h5').replace('images', 'gt_fidt_map')
    gt_loc_path = img_path.replace('.png', '.npy').replace('images', 'ground_truth_localization')
    
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            k = np.load(gt_loc_path)
            gt_fidt_file = h5py.File(gt_fidt_path)
            fidt_map = np.asarray(gt_fidt_file['fidt_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
