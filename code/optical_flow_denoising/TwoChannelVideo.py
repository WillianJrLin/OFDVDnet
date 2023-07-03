# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:22:57 2021

@author: 19522
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def round_img(image_input,dtype=np.uint8):
    ## clip pixel values and casts to type given by dtype
    info = np.iinfo(dtype)
    image = np.clip(image_input,info.min,info.max)
    image = image.astype(dtype)
    return image
  
    

def opt_flow_farneback(img1,img2):
    #Returns optical flow between img1 and img2
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 23, 23, 3, 5, 1.2, 0)
    return flow    


def warp_from_flow(flow, to_warp_arr,inter = cv2.INTER_NEAREST):
    #warps each image along axis=0 of to_warp_arr according to flow 
    if to_warp_arr.ndim ==2:
        to_warp_arr = np.array([to_warp_arr])
    height, width = flow.shape[:2]
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    pixel_map = (R2 + flow).astype(np.float32)
        
    to_return = np.zeros_like(to_warp_arr)
    for i in range(len(to_warp_arr)):
        to_warp = to_warp_arr[i]
        to_return[i,:,:] = cv2.remap(to_warp, pixel_map[..., 0], pixel_map[..., 1],inter)
    return to_return
        
def apply_mask(mask, to_mask_replace_ones,to_mask_replace_zeros=0):
    if to_mask_replace_ones.ndim ==2:
        to_mask_replace_ones = np.array([to_mask_replace_ones])
        
    if len(to_mask_replace_zeros) == 0:
        to_mask_replace_zeros = np.zeros_like(to_mask_replace_ones)
            
    if to_mask_replace_ones.shape != to_mask_replace_zeros.shape:
        print("Error in apply_mask: to_mask_replace_ones and to_mask_replace_zeros need to be same shape")
        return
            
    mask=np.round(np.clip(mask,0,1))
    if mask.ndim ==2:
        mask=np.tile(mask,(len(to_mask_replace_ones),1,1))
    return mask*to_mask_replace_ones+(1-mask)*to_mask_replace_zeros



class GausNoise():
    def __init__(self, sd):
        self.sd = sd
        
    def add_noise(self,fluorescent):
        noisy=fluorescent+np.random.normal(0,self.sd,np.shape(fluorescent))
        return np.clip(noisy,0,np.inf)
    
    def prob_of_classf(self,val1,val2,N):
        mid = (val1+val2)/2
        sigma = self.sd/np.sqrt(N)
        return norm.cdf(mid,min(val1,val2),sigma)
    
class PoissonNoise():
    def __init__(self, maxPhots,minPhots):
        self.maxPhots = maxPhots
        self.minPhots = minPhots
        
    def add_noise(self,fluorescent):
        fl = (fluorescent-np.min(fluorescent))/(np.max(fluorescent)-np.min(fluorescent))
        return np.random.poisson(fl*maxPhots+minPhots)
    
#WEI: Modify this to add dark counts and true noise map from experimental work
class PoissonNoise_multi():
    def __init__(self, maxPhots,minPhots):
        self.maxPhots = maxPhots
        self.minPhots = minPhots
        
    def add_noise(self,fluorescent):
        fl = fluorescent/255.0
        fl = np.expand_dims(fl,1)
        toRe = []
        for i in range(len(self.maxPhots)):
            if len(toRe)==0:
                toRe = (np.random.poisson(fl*self.maxPhots[i]+self.minPhots[i])-self.minPhots[i])/self.maxPhots[i]*255
            else:     
                toRe = np.append(toRe,(np.random.poisson(fl*self.maxPhots[i]+self.minPhots[i])-self.minPhots[i])/self.maxPhots[i]*255,axis=1)
                
        return toRe
    
        

class Video_Callback:
    def __init__(self, show_video=True):
        self.reset()
        self.show_video = show_video
    
    def get_callback_name():
        return "Video_Callback"
    
    def reset(self):
        self.fluorescent_video=[]
        
    def update(self,next_white_light,next_fluorescent_no_noise,next_fluorescent_noise,two_channel_obj):
        estimated_frame =two_channel_obj.fluorescent_estimate()
        if len(self.fluorescent_video)==0:
            self.fluorescent_video = [estimated_frame]
            return
        self.fluorescent_video = np.append(self.fluorescent_video,[estimated_frame])
        
        if self.show_video:
            cv2.imshow('GT fl', round_img(next_fluorescent_no_noise))
        
            gray = cv2.cvtColor(round_img(next_white_light), cv2.COLOR_BGR2GRAY)
            cv2.imshow('Input', gray) 
               
            cv2.imshow('estimated fl - motion averaged', round_img(estimated_frame))
            cv2.imshow('Noisy fl frame', round_img(next_fluorescent_noise))
            
            c = cv2.waitKey(1)
            if c==27:
                raise Exception('Keyboard interrupt')

        
class PSNR_Callback:
    def __init__(self,peak=255):
        self.peak = peak
        self.reset()
    
    def get_callback_name():
        return "PSNR_Callback"
    
    def reset(self):
        self.PSNR=[]
        
    def calc_psnr(self,frame, gt_frame):
        #frame_norm = 255*frame/frame.max()
        #gt_frame_norm = 255*gt_frame/gt_frame.max()
        diff_sqr = np.square(frame - gt_frame)
        diff_sqr_sum = np.sum(diff_sqr, axis=1)
        diff_sqr_sum = np.sum(diff_sqr_sum)
        mse = diff_sqr_sum/(frame.shape[0]*frame.shape[1])
        #psnr = diff_sqr_sum
        psnr = 10*np.log10((self.peak**2)/mse)
        return psnr
    
    def update(self,next_white_light,next_fluorescent_no_noise,next_fluorescent_noise,two_channel_obj):
        estimated_frame =two_channel_obj.fluorescent_estimate()
        self.PSNR = np.append(self.PSNR,self.calc_psnr(estimated_frame,next_fluorescent_no_noise))


class TwoChannelVideo:
    
    def __init__(self,noise_object, flow_func=opt_flow_farneback,flow_mask_threshold=.07,call_back_objects=[Video_Callback(),PSNR_Callback()]):
        
        self.white_light = []
        self.fluorescent = [] #fluorescent channel(s) can track multiple channels\
        self.noise_object= noise_object
        
        #self.flow_func(img1,img2) is a function that calculates flow from img2->img1
        self.flow_func = flow_func
        self.flow_mask_threshold = flow_mask_threshold
        self.callbacks = call_back_objects
    
    def get_mask_forward(self, next_white_light,curr_white_light,flow):
        next_gray = cv2.cvtColor(next_white_light, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_white_light, cv2.COLOR_BGR2GRAY)
        warped_curr = warp_from_flow(flow,curr_gray,cv2.INTER_CUBIC)
        oldMask=np.zeros_like(warped_curr)
        oldMask[np.abs(1-warped_curr/next_gray)<self.flow_mask_threshold]=1
        return oldMask
    
    def deartifact_warped(self,next_white_light,next_fluorescent_noise,flow):
        mask = self.get_mask_forward(next_white_light,self.white_light,flow)
        
        self.fluorescent = apply_mask(mask, self.fluorescent,next_fluorescent_noise)
    
    def process_next_frame(self, next_white_light,next_fluorescent_no_noise,add_noise=True,next_fluorescent_noise=[]):
        #make sure flim signal is shape 3, this allows processing of say different noise models all at once
        if next_fluorescent_no_noise.ndim ==2:
            next_fluorescent_no_noise = np.array([next_fluorescent_no_noise])
            
        if next_fluorescent_noise.ndim ==2:
            next_fluorescent_noise = np.array([next_fluorescent_noise])
            
        #add noise
        if add_noise:
            next_fluorescent_noise = self.noise_object.add_noise(next_fluorescent_no_noise)

        #add counts channel to end
        next_fluorescent_noise = np.append(next_fluorescent_noise, [np.ones_like(next_fluorescent_noise[0])],axis=0)
        
        
        if len(self.white_light) == 0:
            self.white_light = next_white_light
            self.fluorescent = next_fluorescent_noise
            return
        
        #apply flow
        flow = self.flow_func(next_white_light,self.white_light)
        self.fluorescent = warp_from_flow(flow,self.fluorescent)+next_fluorescent_noise
        
        #deartifact - default is masking based on flow errors
        self.deartifact_warped(next_white_light,next_fluorescent_noise,flow)
        
        #update current frame
        self.white_light = next_white_light

        #callbacks save some run history info. we pass in the new frames and the current object so the callback can have any data it needs
        for callback in self.callbacks:
            callback.update(next_white_light,next_fluorescent_no_noise[0],next_fluorescent_noise[0],self)
    
    def fluorescent_estimate(self):
        #return emperical average of current frame
        return self.fluorescent[0]/self.fluorescent[-1]
        

    def get_call_backs(self):
        return self.callbacks
    
        
    
    def reset_call_backs(self):
        for callback in self.callbacks:
            callback.reset()
        
    def display_current_frame(self):
        if len(self.white_light) == 0:
            print("Error: Video has not been initialized")
            return
        plt.figure(0)
        plt.imshow(self.white_light)
        plt.figure(1)
        plt.imshow(self.fluorescent_estimate())
        
        
        
class TwoChannel_forward_backward(TwoChannelVideo):
    def __init__(self,noise_object, flow_func=opt_flow_farneback,flow_mask_threshold=.07,call_back_objects=[Video_Callback(),PSNR_Callback()]):
        TwoChannelVideo.__init__(self,noise_object, flow_func,flow_mask_threshold,call_back_objects)
            
        self.forward_fl = []
        self.backward_fl = []
        self.fluorescent_frames_noise = []
        
    def process_forward_backward(self, white_light_frames,fluorescent_frames_no_noise):
        self.fluorescent_frames_noise = self.noise_object.add_noise(fluorescent_frames_no_noise)
        #print(np.shape(fluorescent_frames_noise))
        self.reset_call_backs()
        for i in range(len(white_light_frames)):
            self.process_next_frame(white_light_frames[i],fluorescent_frames_no_noise[i],add_noise=False,next_fluorescent_noise=self.fluorescent_frames_noise[i])

            if len(self.forward_fl) ==0:
                self.forward_fl=[self.fluorescent]
            else:
                self.forward_fl= np.append(self.forward_fl,[self.fluorescent],axis=0)
            
        #print(np.shape(self.forward_fl))

        self.white_light = []
        self.fluorescent = []
        self.reset_call_backs()
        for i in range(len(white_light_frames)):
            self.process_next_frame(white_light_frames[len(white_light_frames)-1-i],fluorescent_frames_no_noise[len(white_light_frames)-1-i],add_noise=False,next_fluorescent_noise=self.fluorescent_frames_noise[len(white_light_frames)-1-i])
            
            if len(self.backward_fl) ==0:
                self.backward_fl=[self.fluorescent]
            else:
                self.backward_fl= np.append(self.backward_fl,[self.fluorescent],axis=0)
        
        self.backward_fl=np.flip(self.backward_fl,axis=0)
        
        self.white_light = []
        self.fluorescent = []
        self.reset_call_backs()
    
        self.fw_bw_fluorescence = np.zeros_like(self.forward_fl)
        self.fw_bw_fluorescence[:,:-1] = self.backward_fl[:,:-1]+self.forward_fl[:,:-1]-self.fluorescent_frames_noise
        self.fw_bw_fluorescence[:,-1] = self.backward_fl[:,-1]+self.forward_fl[:,-1]-np.ones_like(self.forward_fl[:,-1])
        
        #fw_bw_masks = self.forward_fl[:,-1]>self.backward_fl[:,-1]
        #self.fw_bw_fluorescence[:,0] = apply_mask(fw_bw_masks, self.forward_fl[:,0],np.flip(self.backward_fl[:,0],axis=0))   
        #self.fw_bw_fluorescence[:,-1] = apply_mask(fw_bw_masks, self.forward_fl[:,-1],np.flip(self.backward_fl[:,-1],axis=0))   
        
    def get_fw_bw_estimate_at(self,time):
        return self.fw_bw_fluorescence[time,0]/self.fw_bw_fluorescence[time,1] 
    
    def get_fw_estimate_at(self,time):
        return self.forward_fl[time,0]/self.forward_fl[time,1] 
    
    def get_bw_estimate_at(self,time):
        return self.backward_fl[time,0]/self.backward_fl[time,1] 
