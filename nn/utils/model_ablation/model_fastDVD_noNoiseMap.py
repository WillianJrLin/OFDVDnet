# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:27:52 2021

@author: wlin4
"""

import torch
import torch.nn as nn

class doubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(doubleConv, self).__init__()
        self.doubConv = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True)
		)
        
    def forward(self, x):
        return self.doubConv(x)
    
    
class downSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(downSample, self).__init__()
        self.downSamp = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
            doubleConv(out_channel, out_channel)
		)
        
    def forward(self, x):
        return self.downSamp(x)     
    
    
class upSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upSample, self).__init__()
        self.upSamp = nn.Sequential(
            doubleConv(in_channel, in_channel),
			nn.Conv2d(in_channel, out_channel*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
            #doubleConv(out_channel, out_channel)
		)
        
    def forward(self, x):
        return self.upSamp(x)   
    

class inputConv(nn.Module):
    def __init__(self, num_in_frames, channels_per_frame,  out_channel):
        super(inputConv, self).__init__()
        self.interm_ch = 30
        self.inConv = nn.Sequential(
            nn.Conv2d(num_in_frames*channels_per_frame, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),      
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True)    
		)
        
    def forward(self, x):
        return self.inConv(x) 
    
    
class outputConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(outputConv, self).__init__()
        self.outConv = nn.Sequential(
			nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
		)
        
    def forward(self, x):
        return self.outConv(x)     
        

class modUNET(nn.Module):
    def __init__(self, input_frames_modUNET, channels_per_frame, out_channel = 1, features=[32, 64, 128]):
         super(modUNET, self).__init__()
         self.inputConv = inputConv(input_frames_modUNET, channels_per_frame, features[0])
         self.dwSamp_0 = downSample(features[0], features[1])
         self.dwSamp_1 = downSample(features[1], features[2])
         self.upSamp_2 = upSample(features[2], features[1])
         self.upSamp_1 = upSample(features[1], features[0])
         self.outputConv = outputConv(features[0], out_channel)
         
         self.reset_params()
         
    @staticmethod
    def weight_init(m):
	    if isinstance(m, nn.Conv2d):
		   	nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
	    for _, m in enumerate(self.modules()):
		    self.weight_init(m)
            

    def forward(self, fl_0, wl_0, fl_1, wl_1, fl_2, wl_2):
        
        ## input        
        x_0 =  torch.cat((fl_0, wl_0, fl_1, wl_1, fl_2, wl_2), dim=1)
        print("x_0 shape = ", x_0.shape)
        
        ## down sample
        x_0 = self.inputConv(x_0)
        #print("x_0 shape = ", x_0.shape)
        x_1 = self.dwSamp_0(x_0)
        #print("x_1 shape = ", x_1.shape)
        x_2 = self.dwSamp_1(x_1)
        #print("x_2 shape = ", x_2.shape)
        
        ## up sample
        x_2 = self.upSamp_2(x_2)
        #print("x_2 shape = ", x_2.shape)
        #x_4 = torch.cat((x_4, x_2), dim=1)
        #x_4 = x_4 + x_2
        #print("cat x_4 x_2 shape = ", x_4.shape)
        
        x_1 = self.upSamp_1(x_1 + x_2)
        #print("x_1 shape = ", x_1.shape)
        #x_5 = torch.cat((x_5, x_1), dim=1)
        #x_out = x_5 + x_1
        #print("cat x_5 x_1 shape = ", x_5.shape)
        
        ## output convolution
        x_out = self.outputConv(x_0 + x_1)
        #print("x_out shape = ", x_out.shape)
        
        # residual 
        #mid_index = int(((fluorescent_frame_stack.shape[1]+1)/2)-1)
        #print("mid_index = ", mid_index)
        #fluorescent_frame_mid = fluorescent_frame_stack[:, mid_index, :, :]
        x_out = fl_1 - x_out
        
        return x_out
   
   
class unetTREE(nn.Module):    
    def __init__(self, input_frames_tree, input_frames_modUNET, channels_per_frame):
        super(unetTREE, self).__init__()
        self.num_inputs_tree = input_frames_tree
        self.num_inputs_mod = input_frames_modUNET
        self.treeLevel_0 = modUNET(input_frames_modUNET, channels_per_frame, out_channel = 1, features=[32, 64, 128])
        self.treeLevel_1 = modUNET(input_frames_modUNET, channels_per_frame, out_channel = 1, features=[32, 64, 128])
        
        self.reset_params()
        
    @staticmethod
    def weight_init(m):
	    if isinstance(m, nn.Conv2d):
		    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
	    for _, m in enumerate(self.modules()):
		    self.weight_init(m)
            
    def forward(self, fluorescent_frames, whiteLight_frames):
        
        # tree level 0
        print("dimenson of unetTREE level_0 input: ", fluorescent_frames.shape, whiteLight_frames.shape)   
        
        (frame_0, frame_1, frame_2, frame_3, frame_4) = tuple(fluorescent_frames[:, m:m+1, :, :] for m in range(self.num_inputs_tree))
        (wl_frame_0, wl_frame_1, wl_frame_2, wl_frame_3, wl_frame_4) = tuple(whiteLight_frames[:, m:m+1, :, :] for m in range(self.num_inputs_tree))
        
        frame_20 = self.treeLevel_0(frame_0, wl_frame_0, frame_1, wl_frame_1, frame_2, wl_frame_2)
        frame_21 = self.treeLevel_0(frame_1, wl_frame_1, frame_2, wl_frame_2, frame_3, wl_frame_3)
        frame_22 = self.treeLevel_0(frame_2, wl_frame_2, frame_3, wl_frame_3, frame_4, wl_frame_4)
        
        print("dimenson of unetTREE level_0 output: ", frame_20.shape, frame_21.shape, frame_22.shape)   
        
        # tree level 1
        frame_out = self.treeLevel_1(frame_20, wl_frame_1, frame_21, wl_frame_2, frame_22, wl_frame_3)
            
        print("dimenson of unetTREE level_1 output: ", frame_out.shape)    
        
        return frame_out   
    
'''
def test():
    
    fluorescent_frame_0 = torch.randn((1, 1, 320, 320))    
    fluorescent_frame_1 = torch.randn((1, 1, 320, 320))  
    fluorescent_frame_2 = torch.randn((1, 1, 320, 320))
    fluorescent_frame_3 = torch.randn((1, 1, 320, 320))
    fluorescent_frame_4 = torch.randn((1, 1, 320, 320))
    
    fluorescent_frames_0 = torch.cat((fluorescent_frame_0, fluorescent_frame_1, \
                                    fluorescent_frame_2, fluorescent_frame_3, \
                                    fluorescent_frame_4), dim=1) 
    
    whiteLight_frames_0 = torch.cat((fluorescent_frame_0, fluorescent_frame_1, \
                                    fluorescent_frame_2, fluorescent_frame_3, \
                                    fluorescent_frame_4), dim=1)    
    
    cmap_frames_0 = torch.cat((fluorescent_frame_0, fluorescent_frame_1, \
                                    fluorescent_frame_2, fluorescent_frame_3, \
                                    fluorescent_frame_4), dim=1)        
    
    model = unetTREE(input_frames_tree = 5, input_frames_modUNET = 3, channels_per_frame = 3)    
    fx = model(fluorescent_frames_0, whiteLight_frames_0, cmap_frames_0)    
    print("output shape", fx.shape)
    
    return fx

if __name__ == "__main__":
    out = test()        
'''        