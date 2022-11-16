#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:12:16 2020

@author: shafin0608
"""

#%%
import os
import cv2
import os
import time

#%%
imagePath = '/content/gdrive/My Drive/night_blurred_44_100_gaussian5/train/images'
labelPath = '/content/gdrive/My Drive/night_blurred_44_100_gaussian5/train/labels'
subcount = [2350, 2275, 4811]

#%% 
def draw_box(image, labelname):
    import cv2
    import numpy as np
    #image = cv2.imread(filename)
    height, width, layers = image.shape
    size = (width,height)
    
    color = (255, 0 , 0)
    thickness = 1 
     
    try:
        label = np.ceil(np.loadtxt(labelname)[:, 1:]*width).astype(int)
        for i in range(label.shape[0]):
            start_point = (label[i, 0]-label[i, 2]//2, label[i, 1]-label[i, 3]//3)
            end_point = (label[i, 0]+label[i, 2]//2, label[i, 1]+label[i, 3]//2)
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
    except IndexError:
        label = np.ceil(np.loadtxt(labelname)[1:]*width).astype(int)
        if label.size == 0:
            return image, size
        start_point = (label[0]-label[2]//2, label[1]-label[3]//3)
        end_point = (label[0]+label[2]//2, label[1]+label[3]//2)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
            
    
        
    return image, size

#%%
pathIn= imagePath
fps = 12
frame_array = []

index = 0
k = 0
flag = 1

for image in sorted(os.listdir(imagePath)):
    flag = 0
    filename = pathIn + image
    print(filename)
    labelname = labelPath + image[:-4] + '.txt'
    
    #reading each files, calling draw box which will return the image with bbox drawn over objects
    img = cv2.imread(filename)
    img, size = draw_box(img, labelname)
    frame_array.append(img)
    index = index + 1
    
    if index==subcount[k]:
        pathOut = '/content/gdrive/My Drive/video'+ str(k) +'.avi'
        index = 0
        k = k+1
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        # writing to a image array
        for i in range(len(frame_array)):
            out.write(frame_array[i])
            
        out.release()
        frame_array = []
        print('Done with video ', k-1)
        if k==2:
            break
    

        
    

    
    
    


















