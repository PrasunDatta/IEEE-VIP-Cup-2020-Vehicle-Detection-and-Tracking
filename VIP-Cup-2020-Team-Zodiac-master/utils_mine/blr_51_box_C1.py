# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:56:01 2020

@author: HED
"""

#%% Imports
import cv2
import os
import numpy as np
import pandas as pd
import copy
import pdb

#%% 

filepath = '/content/gdrive/My Drive/night_normal/train/images/'
blurred_labels = '/content/gdrive/My Drive/blurred_labels_44_constraint_2/train/'
blurred_image_path = '/content/gdrive/My Drive/night_blurred_51_box_constraint_2/train/images/'
blur_level = 51
std = 20

allFiles = sorted(os.listdir(filepath))
allLabels = sorted(os.listdir(blurred_labels))

            
def blur(blurred_image):
    import cv2
    if blurred_image.shape[0]==1280:
        patch_size=320
    elif blurred_image.shape[0]==1024:
        patch_size=256

    for i in range(4):
        for j in range(4):       
            patch = blurred_image[i*patch_size+1: (i+1)*patch_size, j*patch_size+1: (j+1)*patch_size ]
            to_blur = current_label.iloc[i,j]
            
            #print(patch.shape)
            if to_blur:
                patch_blurred = cv2.blur(patch, (blur_level,blur_level))
                #patch_blurred = cv2.GaussianBlur(patch,(blur_level,blur_level),std)  
            else:
                patch_blurred = patch

            blurred_image[i*patch_size+1: (i+1)*patch_size, j*patch_size+1: (j+1)*patch_size] = patch_blurred
    
    return blurred_image

#iterate over the whole folder
#take an imagepath
#load that image
#read the blurred label
#break the image down to patches
#blur each patch according to the label
#after image is blurred, write it
count = 0
for basefilename in allFiles:
    current_img_path = os.path.join(filepath, basefilename)
    count = count + 1
    print('New Image ', count)
    print(current_img_path)
    current_img_blurred_path = os.path.join(blurred_image_path, basefilename)
    current_img = cv2.imread(current_img_path)
    
    baselabelname = basefilename[:-3] + 'txt'
    current_label_path = os.path.join(blurred_labels, baselabelname)
    current_label = pd.read_csv(current_label_path, header = None)
    
    blurred_image = copy.deepcopy(current_img)

    blurred_image[:,:,0] = blur(blurred_image[:,:,0])
    blurred_image[:,:,1] = blur(blurred_image[:,:,1])
    blurred_image[:,:,2] = blur(blurred_image[:,:,2])

    cv2.imwrite(current_img_blurred_path, blurred_image)

