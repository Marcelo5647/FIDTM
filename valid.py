import torch
import torch.nn as nn
import numpy as np
import json
import sys
import time
from Networks.HR_Net.seg_hrnet import get_seg_model
from dataset_generation.generate_safety_grids import generate_safety_grid
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
import argparse
import csv

parser = argparse.ArgumentParser(description='FIDTM')
parser.add_argument('--model_name', type=str, help='choose model name')
parser.add_argument('--corruption_name', type=str, help='choose corruption name') 
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

dataset_root = '/srv/storage/datasets/marcelo/CityPark'
img_info_root = os.path.join(dataset_root, 'test_data', 'images_info')
valid_list_txt_path = os.path.join(dataset_root, 'test_data', 'img_list_test.txt')
img_root = os.path.join(dataset_root, 'test_data')
def main():

    model_name = args.model_name
    corruption_name = args.corruption_name
    grid_shape = (3,3)
    min_safe_dist = 250

    fidtm_model = get_seg_model(train=False)
    fidtm_model = nn.DataParallel(fidtm_model, device_ids=[0])
    fidtm_model = fidtm_model.cuda()
    checkpoint = torch.load(f'/homeLocal/marcelo/FIDTM/save_file/{model_name}/model_best.pth')
    fidtm_model.load_state_dict(checkpoint['state_dict'], strict=False)
    fidtm_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")

    mse_criterion = nn.MSELoss()
    mse_losses = []
    safest_cells = []
    safest_cells_gt = []
    most_dangerous_cells = []
    most_dangerous_cells_gt = []
    center_cells_safety = []
    center_cells_safety_gt = []
    cells_safety_gt = []
    cells_safety = []
    exec_times = []

    transform = transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]) 
    
    for severity in range(1, 6):
        print(f'Corruption Severity: {severity}')

        img_paths = []
        for filename in np.loadtxt(valid_list_txt_path, dtype=str):
            if filename.split('.')[1] == 'png':
                if severity == 0:
                    img_paths.append(os.path.join(img_root, 'images', filename))
                else:
                    img_paths.append(os.path.join(img_root, f'corrupted_images/{corruption_name}_{severity}', filename))
        img_paths.sort()

        for index, img_path in enumerate(img_paths):
            image = Image.open(img_path).convert('RGB')
            
            img_tensor = transform(image)
                
            # Send to device
            img_tensor = img_tensor[np.newaxis, :, :, :].to('cuda')

            # Eval
            img_name_without_extension = Path(img_path).stem
            gt_safety_grid = np.load(f'/srv/storage/datasets/marcelo/CityPark/test_data/gt_safety_grids/3x3/{img_name_without_extension}.npy', allow_pickle=True)      
            with torch.no_grad():
                img_info_path = os.path.join(img_info_root, img_name_without_extension + '.json')
                if (os.path.isfile(img_info_path)):
                    with open(str(img_info_path), 'r') as json_file:
                        img_info = json.load(json_file)
                    camera_height = (img_info['camera_location']['z'] - img_info['ground_z_coordinate'])
                    depth = camera_height - 200
                    depth = depth/1300 
                else:
                    raise FileNotFoundError('Input image info not found.')

                start = time.time()

                fidt_map = fidtm_model(img_tensor)
                pred_kpoint = LMDS_counting(fidt_map)
                safety_grid = generate_safety_grid(pred_kpoint, grid_shape, img_info, min_safe_dist)
                end = time.time()
               
                mse_loss = mse_criterion(torch.tensor(safety_grid).to('cpu'), torch.tensor(gt_safety_grid))

                mse_losses.append(mse_loss)

                safest_cells_gt.append(np.argmax(gt_safety_grid))
                if gt_safety_grid.flatten()[np.argmax(safety_grid)] == 1:
                    safest_cells.append(np.argmax(gt_safety_grid))
                else:
                    safest_cells.append(np.argmax(safety_grid))

                most_dangerous_cells_gt.append(np.argmin(gt_safety_grid))
                most_dangerous_cells.append(np.argmin(safety_grid))

                center_cells_safety_gt.append(gt_safety_grid[1,1] > 0.5)
                center_cells_safety.append(safety_grid[1,1] > 0.5)

                cells_safety_gt.extend((gt_safety_grid > 0.5).flatten())
                cells_safety.extend((safety_grid > 0.5).flatten())

                exec_times.append((end-start)*10**3) # time in miliseconds

            if index % 100 == 0:
                print(f'Index: {index} Image: {img_name_without_extension} Average MSE Loss: {sum(mse_losses)/len(mse_losses)}')

        # Calculate metrics
        safest_cell_accuracy = 100*np.sum(np.array(safest_cells) == np.array(safest_cells_gt))/len(safest_cells)
        most_dangerous_cell_accuracy = 100*np.sum(np.array(most_dangerous_cells) == np.array(most_dangerous_cells_gt))/len(most_dangerous_cells)
        center_cells_safety_accuracy = 100*np.sum(np.array(center_cells_safety) == np.array(center_cells_safety_gt))/len(center_cells_safety)
        cells_safety_accuracy = 100*np.sum(np.array(cells_safety) == np.array(cells_safety_gt))/len(cells_safety)
        average_mse_loss = sum(mse_losses)/len(mse_losses)
        mean_exec_time = sum(exec_times)/len(exec_times)

        # Print metrics
        print(f'Corruption Severity = {severity}')
        print(f'Safest Cell Accuracy = {safest_cell_accuracy}')
        print(f'Most Dangerous Cell Accuracy = {most_dangerous_cell_accuracy}')
        print(f'Center Cell Safety Accuracy = {center_cells_safety_accuracy}')
        print(f'Cell Safety Accuracy = {cells_safety_accuracy}')
        print(f'Average MSE Loss = {average_mse_loss}')
        print(f'Mean Exec Time (ms) = {mean_exec_time}')

        # Save metrics to CSV file
        csv_file = f'/homeLocal/marcelo/FIDTM/{model_name}_{corruption_name}_safety_metrics.csv'
        header = ['Corruption Severity', 'Safest Cell Accuracy', 'Most Dangerous Cell Accuracy', 'Center Cell Safety Accuracy', 'Cell Safety Accuracy', 'Average MSE Loss', 'Mean Exec Time (ms)']
        data = [severity, safest_cell_accuracy, most_dangerous_cell_accuracy, center_cells_safety_accuracy, cells_safety_accuracy, average_mse_loss, mean_exec_time]

        with open(csv_file, mode='a') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(header)
            writer.writerow(data)

if __name__ == '__main__':
    main()