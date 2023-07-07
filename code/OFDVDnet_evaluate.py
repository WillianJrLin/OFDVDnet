# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:24:38 2021

@author: wlin4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import multiprocessing
import matplotlib.pyplot as plt
import h5py
import time
import csv
import cv2
import os
from PIL import Image
from torchvision import transforms
import os
import argparse

from utils.model_ablation.model_fastDVD_onlume import unetTREE


parser = argparse.ArgumentParser(description='OFDVDnet evaluation')
parser.add_argument('--noiseLevel', type=int, default=3, help='enter the noiselevel, default noiselevel = 3')
#parser.add_argument('--epochs', type=int, default=100, help='enter the number of training epochs, default number of epochs = 100')
parser.add_argument('--dir', help='enter the parent directory of the directories /fl_gt, /fl_noise, /noise_map_est')
args = parser.parse_args()


cuda = torch.cuda.is_available()
# cuda = 0
device = torch.device("cuda" if cuda else "cpu")
CSV_FIELDNAMES = ["Epoch", "Loss"]


""" Initializing Dataset class """
class FLIM_Videos(Dataset):
    def __init__(self, frame_list, num_frames_stack, warp, parent_path, noiseLevel):
        self.frameList = frame_list
        self.frameStack_size = num_frames_stack
        self.warp = warp 
        self.parent_path = parent_path
        self.noiseLevel = noiseLevel
    
    def __len__(self):
        return len(self.frameList)
   
    def _opt_flow_warp(self, img1, img2, towarp,towarp2,towarp3):
        #Calculates optical flow between img1 and img2. 
        #Uses the flow to warp towarp and towarp2 
        #from img2->img1
        img1_gray = img1
        img2_gray = img2
        flow = cv2.calcOpticalFlowFarneback(
            img1_gray, img2_gray, None, 0.5, 23, 23, 3, 5, 1.2, 0)
    
        flow[np.where(np.abs(flow) > 10)] = 0
    
        height, width = flow.shape[:2]
        R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
        pixel_map = (R2 + flow).astype(np.float32)
        
        warped1 = cv2.remap(towarp, pixel_map[..., 0], pixel_map[..., 1],
                             cv2.INTER_NEAREST)
        warped2 = cv2.remap(towarp2, pixel_map[..., 0], pixel_map[..., 1],
                             cv2.INTER_NEAREST,)
        warped3 = cv2.remap(towarp3, pixel_map[..., 0], pixel_map[..., 1],
                             cv2.INTER_NEAREST,)
        return warped1,warped2,warped3,flow
        
    
    def __getitem__(self, index):
        frame_index = self.frameList[index] 
        vid_id = frame_index[0] 
        frame_id = frame_index[1]
        #print(vid_id, frame_id)
        
        
        video_index = f"{vid_id + 1}"
        frame_path_gt = f"{self.parent_path}/fl_gt/{video_index}"
        frame_path_noisy = f"{self.parent_path}/fw_bw/{video_index}"
        frame_counts_map = f"{self.parent_path}/fw_bw/{video_index}"
        frame_path_whiteLight = f"{self.parent_path}/wl/{video_index}"
        
        frame_index_gt = f"{frame_id}.png"
        frame_path_final_gt = f"{frame_path_gt}/{frame_index_gt}"        
        frame_index_whiteLight_target = f"{frame_id}.png"
        frame_path_final_whiteLight_target = f"{frame_path_whiteLight}/{frame_index_whiteLight_target}"
        
        frame_gt = Image.open(frame_path_final_gt)
        frame_whiteLight_target = Image.open(frame_path_final_whiteLight_target)
        
        width, hight = frame_gt.size
        #print(hight, width)
      
        if hight % 4 != 0 or width % 4 != 0:
            resizing = transforms.Compose([transforms.CenterCrop((4*(hight//4), 4*(width//4)))])
            frame_whiteLight_target = resizing(frame_whiteLight_target)
            frame_gt = resizing(frame_gt)
        
        frame_gt = np.array(frame_gt, dtype=np.float32)
        frame_whiteLight_target = np.array(frame_whiteLight_target, dtype=np.float32)
        
        for m in range (int(frame_id-(self.frameStack_size-1)/2), int(1+frame_id+(self.frameStack_size-1)/2)):
            frame_index_noisy = f"{m}.npy"
            frame_index_whiteLight = f"{m}.png"
            frame_index_countsMap = f"{m}.npy"
            
            frame_path_final_noisy = f"{frame_path_noisy}/{frame_index_noisy}"
            frame_path_final_whiteLight = f"{frame_path_whiteLight}/{frame_index_whiteLight}"
            frame_path_final_countsMap = f"{frame_counts_map}/{frame_index_countsMap}"
            
            frame_noisy = np.load(frame_path_final_noisy)[self.noiseLevel, :, :]
            frame_countsMap = np.load(frame_path_final_countsMap)[-1, :, :]
            frame_whiteLight = Image.open(frame_path_final_whiteLight)
            
            if hight % 4 != 0 or width % 4 != 0:
                resizing = transforms.Compose([transforms.CenterCrop((4*(hight//4), 4*(width//4)))])
                frame_whiteLight = resizing(frame_whiteLight)
                frame_noisy = resizing(frame_noisy)
                frame_countsMap = resizing(frame_countsMap)
            
            frame_noisy = np.array(frame_noisy, dtype=np.float32)
            frame_whiteLight =  np.array(frame_whiteLight, dtype=np.float32)
            frame_countsMap = np.array(frame_countsMap, dtype=np.float32)
            
            #print(type(np.array(frame_noisy)), np.array(frame_noisy).shape, self.warp)
            
            if self.warp:
                if m != frame_id:
                    frame_whiteLight,frame_noisy,frame_countsMap,flow = self._opt_flow_warp(frame_whiteLight_target, \
                                                                       frame_whiteLight, \
                                                                       frame_whiteLight, \
                                                                       frame_noisy, \
                                                                       frame_countsMap)
            
            #frame_whiteLight = frame_whiteLight[::4, ::4]
            #frame_noisy = frame_noisy[::4, ::4]
            #frame_countsMap = frame_countsMap[::4, ::4]
            
            frame_noisy = np.expand_dims(frame_noisy, axis=2)
            frame_whiteLight = np.expand_dims(frame_whiteLight, axis=2)
            frame_countsMap = np.expand_dims(frame_countsMap, axis=2)
                    
            if m == int(frame_id-(self.frameStack_size-1)/2):
                frame_stack_noisy = frame_noisy
                frame_stack_whiteLight = frame_whiteLight
                frame_stack_countsMap = frame_countsMap
            else:
                frame_stack_noisy = np.concatenate((frame_stack_noisy, frame_noisy), axis=2)
                frame_stack_whiteLight = np.concatenate((frame_stack_whiteLight, frame_whiteLight), axis=2)
                frame_stack_countsMap = np.concatenate((frame_stack_countsMap, frame_countsMap), axis=2)
                
        #frame_gt = frame_gt[::4, ::4]
        #print(frame_stack_noisy.shape, frame_stack_whiteLight.shape, frame_stack_countsMap.shape, frame_gt.shape) 
        
        return frame_stack_noisy, frame_stack_whiteLight, frame_stack_countsMap, frame_gt, vid_id, frame_id
      

def evaluate(model, device, loader, Loss_fun, noiseLevel):
    #output_visualize_array = []
    
    with torch.no_grad():
        model.eval()  
        running_loss = 0.0
        
        for i, (frame_stack_noisy, frame_stack_whiteLight, frame_stack_countsMap, frame_gt, vid_id, frame_id) in enumerate(loader):   
            frame_stack_noisy = frame_stack_noisy.permute(0, 3, 1, 2)
            frame_stack_whiteLight = frame_stack_whiteLight.permute(0, 3, 1, 2)
            frame_stack_countsMap = frame_stack_countsMap.permute(0, 3, 1, 2)
            
            frame_stack_noisy = frame_stack_noisy.to(device = device, dtype=torch.float)
            frame_stack_whiteLight = frame_stack_whiteLight.to(device = device, dtype=torch.float)
            frame_stack_countsMap = frame_stack_countsMap.to(device = device, dtype=torch.float)
            frame_gt = frame_gt.to(device = device, dtype=torch.float)
            
            fx = model(frame_stack_noisy, frame_stack_whiteLight, frame_stack_countsMap)
            #print("fx shape: ", fx.shape)
            fx = fx.permute(1, 0, 2, 3)
              
            loss = Loss_fun(fx[0], frame_gt)
            running_loss += loss.item() 
            print("index: ", i, "loss: ", loss, flush=True)
            #print('idx = ', i, 'loss = ', loss)
            
            denoised_image_path = f"./output_eval_fastDVD/eval_onlume_noiseLevel{noiseLevel}/denoised/vid_{vid_id.cpu().detach().numpy()[0]}/"
            if not os.path.exists(os.path.dirname(denoised_image_path)):
                os.makedirs(os.path.dirname(denoised_image_path)) 
            
            gt_image_path = f"./output_eval_fastDVD/eval_onlume_noiseLevel{noiseLevel}/gt/vid_{vid_id.cpu().detach().numpy()[0]}/"
            if not os.path.exists(os.path.dirname(gt_image_path)):
                os.makedirs(os.path.dirname(gt_image_path)) 
            
            noisy_image_path = f"./output_eval_fastDVD/eval_onlume_noiseLevel{noiseLevel}/noisy/vid_{vid_id.cpu().detach().numpy()[0]}/"
            if not os.path.exists(os.path.dirname(noisy_image_path)):
                os.makedirs(os.path.dirname(noisy_image_path))     
                
            for i in range(frame_stack_noisy.shape[0]):    
                save_frame_index = f"{frame_id.cpu().detach().numpy()[i]}".zfill(3)    
                denoised_image_path_full = f"{denoised_image_path}frame_{save_frame_index}.png"
                gt_image_path_full = f"{gt_image_path}frame_{save_frame_index}.png"
                noisy_image_path_full = f"{noisy_image_path}frame_{save_frame_index}.png"
                
                output_visualize = fx[0][i, :, :]
                noisy_visualize = frame_stack_noisy[i, int((frame_stack_noisy.shape[1]+1)/2), :, :]
                gt_visualize = frame_gt[i, :, :]
                #print("output_visualize shape: ", output_visualize.shape)
                output_visualize = output_visualize.cpu().detach().numpy()
                output_visualize = clip_pixelVal(output_visualize)
                output_visualize = output_visualize.astype('uint8')
                
                gt_visualize = gt_visualize.cpu().detach().numpy()
                gt_visualize = clip_pixelVal(gt_visualize)
                gt_visualize = gt_visualize.astype('uint8')
                
                noisy_visualize = noisy_visualize.cpu().detach().numpy()
                noisy_visualize = clip_pixelVal(noisy_visualize)
                noisy_visualize = noisy_visualize.astype('uint8')
                
                cv2.imwrite(denoised_image_path_full, output_visualize)
                cv2.imwrite(gt_image_path_full, gt_visualize)
                cv2.imwrite(noisy_image_path_full, noisy_visualize)
          
        running_loss /= len(loader)   
        
    return running_loss

def clip_pixelVal(image):
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8)
    return image


if __name__ == '__main__':
    #num_videos = 20
    num_frames_video = 100
    num_frames_stack = 5
    input_frames_modUNET = 3
    channels_per_frame = 3
    frame_list = []
    
    noiseLevel = args.noiseLevel
    parent_path = args.dir
    
    for vid in range(302, 502, 2):
        for frame in range(int((num_frames_stack-1)/2), int(num_frames_video-(num_frames_stack-1)/2)):
            index = (vid, frame)
            frame_list.append(index)
    
    test_set = FLIM_Videos(frame_list, num_frames_stack=num_frames_stack, warp=True, parent_path=parent_path, noiseLevel=noiseLevel)
    testLoader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=0)
    
    model = unetTREE(input_frames_tree = num_frames_stack, input_frames_modUNET = input_frames_modUNET, channels_per_frame = channels_per_frame).float()
    model.load_state_dict(torch.load(f"./saved_model/fastDVD/saved_model_fastDVD_noiseLevel{noiseLevel}.pt"))
    model = model.to(device)
    
    Loss_fun  = nn.MSELoss()
    running_loss = evaluate(model, device, testLoader, Loss_fun, noiseLevel)
    print('rn_loss = ', running_loss)