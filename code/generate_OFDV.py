
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import os.path
from tqdm import tqdm
import sys
from joblib import Parallel, delayed
np.float = np.float64
np.int = np.int_
import skvideo.io  

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from optical_flow_denoising import TwoChannelVideo
def scale(img,minVal,maxVal):
    img = img/np.max(img)
    return img*(maxVal-minVal)+minVal



base_file = "../"
folders =["data"]
videoPaths = []
for fold in folders:
    videoPaths = np.append(videoPaths,np.array(glob.glob(os.path.join(base_file+fold, '*.mp4'))))
print("num videos: ", len(videoPaths))

def load_video(file):
    vidNum =0 # selects video from names array.
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        raise IOError("Cannot open video")
    #%%loads frames from video
    vid = []

    ret = True
    while ret: # a total of 1827 frames in OnLume-MouseTumorResection.mp4
        ret, newframe = cap.read()  

        if len(vid)==0:
            vid =[newframe]
        else:
            vid=np.append(vid,[newframe],axis=0)

    cap.release()
    return vid

def parse_video(num,wl,fl):  
    path1 = base_file+to_save_wl+str(num)+"/"
    path2 = base_file+to_save_fl+str(num)+"/"
    path3 = base_file+to_save_fl_noise+str(num)+"/"
    path4 = base_file+to_save_fw_bw+str(num)+"/"
    path5 = base_file+to_save_fw+str(num)+"/"
    
    tcv = TwoChannelVideo.TwoChannel_forward_backward(noise_obj,call_back_objects=call_backs)
    tcv.process_forward_backward(wl,fl)
    
    fw_flow = np.zeros_like(tcv.fw_bw_fluorescence)
    fw_bw_flow = np.zeros_like(tcv.fw_bw_fluorescence)

    for i in range(np.shape(fw_flow)[1]-1):
        fw_flow[:,i,:,:]=tcv.forward_fl[:,i]/tcv.forward_fl[:,-1]
    fw_flow[:,-1,:,:]=tcv.forward_fl[:,-1]
    
    fw_flow = np.clip(fw_flow,0,2**16)
    fw_flow = np.array(fw_flow,dtype=np.uint16)
    
    for i in range(np.shape(fw_bw_flow)[1]-1):
        fw_bw_flow[:,i,:,:]=tcv.fw_bw_fluorescence[:,i]/tcv.fw_bw_fluorescence[:,-1]
    fw_bw_flow[:,-1,:,:]=tcv.fw_bw_fluorescence[:,-1]
    
    fw_bw_flow = np.clip(fw_bw_flow,0,2**16)
    fw_bw_flow = np.array(fw_bw_flow,dtype=np.uint16)

    for frame in range(len(wl)):
        cv2.imwrite(path1+str(frame)+".png", cv2.cvtColor(wl[frame],cv2.COLOR_BGR2GRAY))
        cv2.imwrite(path2+str(frame)+".png", fl[frame])
        
        np.save(path3+str(frame)+".npy",tcv.fluorescent_frames_noise[frame])
        
        np.save(path4+str(frame)+".npy",fw_bw_flow[frame])
        
temp = np.array([8.1],dtype=np.uint8)
print(temp)
call_backs=[]

max_phots = [1,5,10,20,100]
min_phots = [10,10,10,10,10]

#WEI: change folders
#modify noise_obj to use use dark count noise
noise_obj=TwoChannelVideo.PoissonNoise_multi(max_phots,min_phots)
to_save_fl = "data/fl_gt/"
to_save_wl = "data/wl/"
to_save_fl_noise = "data/fl_noise/"
to_save_fw_bw = "data/fw_bw/"

n_cores = 30

saveNum = 1

numFrames = 100

vids_at_once = 3
#for num in tqdm(range(0,len(videoPaths),vids_at_once)):
for num in tqdm(range(0,len(videoPaths),vids_at_once)):
    wl_t = []
    fl_t=[]
    for j in range(vids_at_once):
        print(num+j)
        
        if num+j >= len(videoPaths):
            break

        
        vid = skvideo.io.vread(videoPaths[num+j]) 
        
        wl_t2 = vid[:,768:,:1024]
        wl_t2 = wl_t2[:,::4,::4]
        
        fl_temp = vid[:,:768,1024:]
        fl_temp = fl_temp[:,::4,::4]
        
        fl_t2 = np.array([cv2.cvtColor(fl_temp[i],cv2.COLOR_BGR2GRAY) for i in range(len(fl_temp))])
        
        totFrames = len(fl_t2)
        numCuts = int(totFrames/numFrames)
        if len(wl_t)==0:
            wl_t = wl_t2[:numCuts*numFrames]
            fl_t = fl_t2[:numCuts*numFrames]
        else:
            wl_t = np.append(wl_t,wl_t2[:numCuts*numFrames],axis=0)
            fl_t = np.append(fl_t,fl_t2[:numCuts*numFrames],axis=0)
            
    
    totFrames = len(fl_t)
    numCuts = int(totFrames/numFrames)
    
    
    for num in range(saveNum, saveNum+numCuts):
        path1 = base_file+to_save_wl+str(num)+"/"
        path2 = base_file+to_save_fl+str(num)+"/"
        path3 = base_file+to_save_fl_noise+str(num)+"/"
        path4 = base_file+to_save_fw_bw+str(num)+"/"
        path5 = base_file+to_save_fw+str(num)+"/"

        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
        if not os.path.exists(path3):
            os.makedirs(path3)
        if not os.path.exists(path4):
            os.makedirs(path4)
        if not os.path.exists(path5):
            os.makedirs(path5)
            
    print(numCuts)
    
    Parallel(n_jobs=n_cores)(delayed(parse_video)(saveNum+i,wl_t[i*numFrames:(i+1)*numFrames],fl_t[i*numFrames:(i+1)*numFrames]) for i in tqdm(range(numCuts)))
        
    saveNum += numCuts
print("next saveNum",saveNum)


noise_map_path = base_file+"data/noise_map_est/"
noisy_path = base_file+to_save_fl_noise

max_phots = np.array(max_phots)
min_phots =np.array(min_phots)

def calc_noise_map(vid):
    for frame in range(numFrames):
        vid_frame = str(vid)+"/"+str(frame)+".npy"
        n = np.load(noisy_path+vid_frame)
        n_map =[]
        for nlvl in range(0,len(max_phots)):
            x = n[nlvl]*max_phots[nlvl]/255+min_phots[nlvl]
            m=np.sqrt(np.average(x))
            m=m*255/max_phots[nlvl]
            n_map.append(m)
        if not os.path.exists(noise_map_path+str(vid)+"/"):
            os.makedirs(noise_map_path+str(vid)+"/")

        np.save(noise_map_path+vid_frame,n_map)
        
Parallel(n_jobs=10)(delayed(calc_noise_map)(i) for i in tqdm(range(1,591)))