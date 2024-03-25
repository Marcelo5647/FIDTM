import torch
import torch.nn as nn
import numpy as np
from Networks.HR_Net.seg_hrnet import get_seg_model
from PIL import Image
from torchvision import transforms
import os
import glob
import math
import argparse

parser = argparse.ArgumentParser(description='FIDTM')
parser.add_argument('--model_name', type=str, help='choose model name')
args = parser.parse_args()

def LMDS_counting(input):
    input_max = torch.max(input).item()

    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    
    return kpoint

def main():

    dataset_root = '/srv/storage/datasets/marcelo/CityPark'
    dataset_test = os.path.join(dataset_root, 'test_data', 'images')
    model_name = args.model_name
    path_sets = [dataset_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.png')):
            img_paths.append(img_path)
    img_paths.sort()

    fidtm_model = get_seg_model(train=False)
    fidtm_model = nn.DataParallel(fidtm_model, device_ids=[0])
    fidtm_model = fidtm_model.cuda()
    checkpoint = torch.load(f'/homeLocal/marcelo/FIDTM/save_file/{model_name}/model_best.pth')
    fidtm_model.load_state_dict(checkpoint['state_dict'], strict=False)
    fidtm_model.eval()

    transform = transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]) 
    
    f = open(f'./point_files/{model_name}_pred_fidt.txt', 'w+')
    for index, img_path in enumerate(img_paths):
        if index % 50 == 0:
            print(img_path)

        image = Image.open(img_path).convert('RGB')
        
        img_tensor = transform(image)
            
        # Send to device
        img_tensor = img_tensor[np.newaxis, :, :, :].to('cuda')

        # Eval     
        with torch.no_grad():
            fidt_map = fidtm_model(img_tensor)
            pred_kpoint = LMDS_counting(fidt_map)
            indices = np.floor(np.where(pred_kpoint == 1)).astype(int)

            f.write('{} {} '.format(index+1, len(indices[0])))

            for y_index, x_index in zip(indices[0], indices[1]):

                f.write('{} {} '.format(math.floor(y_index), math.floor(x_index)))
            f.write('\n')


if __name__ == '__main__':
    main()