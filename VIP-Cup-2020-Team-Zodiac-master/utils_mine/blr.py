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
import matplotlib.pyplot as plt
#%% 

filepath = 'F:/VIP Cup 2020 Resources/Big Data/night/images/'
blurred_labels = 'F:/VIP Cup 2020 Resources/Big Data/blurred_labels_44_constraint_3/train/'
#blurred_image_path = 'F:/VIP Cup 2020 Resources/Big Data/night_blurred/train/'
blurred_image_path = 'C:/Users/HED/Desktop/'

allFiles = sorted(os.listdir(filepath))
allLabels = sorted(os.listdir(blurred_labels))

patch_config = 4

#iterate over the whole folder
#take an imagepath
#load that image
#read the blurred label
#break the image down to patches
#blur each patch according to the label
#after image is blurred, write it

for basefilename in allFiles:
    current_img_path = os.path.join(filepath, basefilename)
    print(current_img_path)
    current_img_blurred_path = os.path.join(blurred_image_path, basefilename)
    current_img = cv2.imread(current_img_path)
    
    baselabelname = basefilename[:-3] + 'txt'
    current_label_path = os.path.join(blurred_labels, baselabelname)
    current_label = pd.read_csv(current_label_path, header = None)
    
    blurred_image = copy.deepcopy(current_img)
    #blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    
    
    blurred_image[:,:,0] = blur(blurred_image[:,:,0], patch_config)
    blurred_image[:,:,1] = blur(blurred_image[:,:,1], patch_config)
    blurred_image[:,:,2] = blur(blurred_image[:,:,2], patch_config)
    
    
    #blurred_image = blur(current_img)
    
    cv2.imwrite(current_img_blurred_path, blurred_image)
    plt.imshow(blurred_image)
    plt.imshow(current_img)
    #plt.imshow(cv2.GaussianBlur(current_img, (15,15),0))
    #plt.imshow(cv2.bilateralFilter(current_img,9,75,75)) 
    plt.imshow(cv2.blur(current_img, (100,100)))
            

#%% 
def blur(blurred_image, patch_config):
    #import cv2
    #patch_size = int(blurred_image.shape[0]/4)
    
    if blurred_image.shape[0]==1280:
        if patch_config==4:
            patch_size=320
        elif patch_config==8:
            patch_size = 160
    elif blurred_image.shape[0]==1024:
        if patch_config==4:
            patch_size=256
        elif patch_config==8:
            patch_size = 128
        
    for i in range(4):
        for j in range(4):       
            patch = blurred_image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size ]
            to_blur = current_label.iloc[i,j]
            
            #print(patch.shape)
            if to_blur:
                #patch_blurred = np.zeros((256,256))
                patch_blurred = cv2.blur(patch, (201,201))
                #patch_blurred = cv2.GaussianBlur(patch,(101,101),30)
                #patch_blurred = cv2.medianBlur(patch,5)
                
            else:
                patch_blurred = patch
                
                
            blurred_image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = patch_blurred
    
    return blurred_image

#%% seeing the gaussian kernel

import cv2
import numpy as np
patch = np.zeros((7,7))
patch[3,3] = 1
#patch_blurred = cv2.blur(patch, (51,51))
patch_blurred = cv2.GaussianBlur(patch,(5,5),2)

for i in range(len(patch_blurred)):
    for j in range(len(patch_blurred[i])):
        patch_blurred[i,j] = float("{0:.3f}".format(patch_blurred[i,j]))

print(patch_blurred)






#%%
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
import matplotlib.pyplot as plt
#%% 

filepath = 'F:/VIP Cup 2020 Resources/Big Data/night/images/'
blurred_labels = 'F:/VIP Cup 2020 Resources/Big Data/blurred_labels_44_prev/train/'
blurred_image_path = 'F:/VIP Cup 2020 Resources/Big Data/night_blurred/train/'

blurred_labels_1 = 'F:/VIP Cup 2020 Resources/Big Data/blurred_labels_44_constraint_1/train/'
blurred_labels_2 = 'F:/VIP Cup 2020 Resources/Big Data/blurred_labels_44_constraint_2/train/'
blurred_labels_3 = 'F:/VIP Cup 2020 Resources/Big Data/blurred_labels_44_constraint_3/train/'


allFiles = sorted(os.listdir(filepath))
allLabels = sorted(os.listdir(blurred_labels))

patch_config = 4

#iterate over the whole folder
#take an imagepath
#load that image
#read the blurred label
#break the image down to patches
#blur each patch according to the label
#after image is blurred, write it

for basefilename in allFiles:
    current_img_path = os.path.join(filepath, basefilename)
    print(current_img_path)
    current_img_blurred_path = os.path.join(blurred_image_path, basefilename)
    current_img = cv2.imread(current_img_path)
    
    baselabelname = basefilename[:-3] + 'txt'
    current_label_path = os.path.join(blurred_labels_3, baselabelname)
    current_label = pd.read_csv(current_label_path, header = None)
    
    blurred_image = copy.deepcopy(current_img)
    #blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    
    
    blurred_image[:,:,0] = blur(blurred_image[:,:,0], patch_config)
    blurred_image[:,:,1] = blur(blurred_image[:,:,1], patch_config)
    blurred_image[:,:,2] = blur(blurred_image[:,:,2], patch_config)
    
    
    #blurred_image = blur(current_img)
    
    cv2.imwrite(current_img_blurred_path, blurred_image)
    plt.imshow(blurred_image)
    plt.imshow(current_img)
    #plt.imshow(cv2.GaussianBlur(current_img, (15,15),0))
    #plt.imshow(cv2.bilateralFilter(current_img,9,75,75)) 
    plt.imshow(cv2.blur(current_img, (100,100)))
            

#%% 
def blur(blurred_image, patch_config):
    #import cv2
    #patch_size = int(blurred_image.shape[0]/4)
    
    if blurred_image.shape[0]==1280:
        if patch_config==4:
            patch_size=320
        elif patch_config==8:
            patch_size = 160
    elif blurred_image.shape[0]==1024:
        if patch_config==4:
            patch_size=256
        elif patch_config==8:
            patch_size = 128
        
    for i in range(4):
        for j in range(4):       
            patch = blurred_image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size ]
            to_blur = current_label.iloc[i,j]
            
            #print(patch.shape)
            if to_blur:
                patch_blurred = cv2.blur(patch, (201,201))
                #patch_blurred = cv2.GaussianBlur(patch,(201,201),45)
                #patch_blurred = cv2.medianBlur(patch,5)
                
            else:
                patch_blurred = patch
                
                
            blurred_image[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = patch_blurred
    
    return blurred_image

#%% seeing the gaussian kernel

import cv2
import numpy as np
patch = np.zeros((7,7))
patch[1,1] = 1
patch_blurred = cv2.blur(patch, (3,3))
#patch_blurred = cv2.GaussianBlur(patch,(3,3),1)

for i in range(len(patch_blurred)):
    for j in range(len(patch_blurred[i])):
        patch_blurred[i,j] = float("{0:.5f}".format(patch_blurred[i,j]))
print(patch)
print(patch_blurred)
    

    
    
    
    
    
    
    
    