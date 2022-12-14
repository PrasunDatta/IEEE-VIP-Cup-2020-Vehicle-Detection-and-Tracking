#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:15:01 2020

@author: shafin0608
"""

#%%
import cv2
import numpy as np
import random

def draw_box(image, labelname):
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

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img



def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img


def brightness(img, low, high):
    value = random.uniform(low, high)
    #print('Value of brightness is ', value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img


def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
    
    
def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def translation(img, low, high):
    
    height, width = img.shape[:2] 
    
    scale = int(random.uniform(low, high))
  
    translation_height, translation_width = height / scale, width / scale
  
    T = np.float32([[1, 0, translation_width], [0, 1, translation_height]]) 
  
    # We use warpAffine to transform 
    # the image using the matrix, T 
    img = cv2.warpAffine(img, T, (width, height)) 
    
    return img
    
    

























































