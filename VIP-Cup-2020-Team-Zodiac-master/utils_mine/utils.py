#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:12:16 2020

@author: shafin0608
"""

#%%
import os
import cv2
#import numpy as np
import os
#from os.path import isfile, join
from transformations import horizontal_flip, vertical_flip, brightness, translation
#import glob


#%%
filePath = input()
if filePath == 'day':
    imagePath = os.path.join('./' , filePath, 'images/train/')
    labelPath = os.path.join('./' , filePath, 'labels/train/')
else:
    imagePath = os.path.join('./' , filePath, 'images/')
    labelPath = os.path.join('./' , filePath, 'labels/')
print(imagePath)


substr = '000000'
subcount = []
videos = []
subcount_index = 0
for imageName in sorted(os.listdir(imagePath)):
    if substr in imageName:
        if (imageName.startswith('01_fisheye_day_') | 
            imageName.startswith('CLIP_20200628-210253_000000')):
            subcount_index = subcount_index + 1
            video_name = imageName[:-4]
            videos.append(video_name)
            continue
        else:
            subcount.append(subcount_index)
            subcount_index = 0
        print(imageName)
        video_name = imageName[:-4]
        videos.append(video_name)
    subcount_index = subcount_index + 1
      
subcount.append(subcount_index)

print(len(sorted(os.listdir(imagePath))))
print(sum(subcount))
print(subcount)

#%%
pathIn= imagePath
fps = 12
frame_array = []

index = 0
k = 0
flag = 1

for image in sorted(os.listdir(imagePath)):
    # if image.startswith('CLIP_20200610-213821') == False & flag:
    #     #print('Ekhono na')
    #     continue
    #print(image)
    #print('Ekhon')
    #print(image)
    #break
    flag = 0
    filename = pathIn + image
    print(filename)
    labelname = labelPath + image[:-4] + '.txt'
    
    #reading each files, calling draw box which will return the image with bbox drawn over objects
    
    img = cv2.imread(filename)
    img, size = draw_box(img, labelname)
    
    # img = translation(img, 10, 15)
    # height, width, layers = img.shape
    # size = (width,height)
    
    
    frame_array.append(img)
    index = index + 1
    
    if index==subcount[k]:
        #exit()
        pathOut = videos[k] + '.avi'
        index = 0
        k = k+1
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        # writing to a image array
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
            
        out.release()
        frame_array = []
        print('Done with ' + videos[k-1])
    
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
        
    

    
    
    


















