# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 20:22:56 2020

@author: HED
"""

#%% 
import json
dayfileID = open('../../test_detections/day_detection_results.bbox.json' , 'r')
day_results = json.loads(dayfileID.read())
nightfileID = open('../../test_detections/night_normal_detection_results.bbox.json' , 'r')
night_results = json.loads(nightfileID.read())

#%% 
#load the two json files
#from the first json read the current image id
#if the current image id is equal to the running image id, then keep appending
#if the current image id exceeds the running image id, then move to the next json
#if the current image id is equal to the running image id, then keep appending
#increment the running image id

cocofile = []

for image_id in range(2826):
    
    print('Scanning day results for id ', image_id)
    
    for day_entry in day_results:
        
        if day_entry["image_id"] == image_id:
            cocofile.append(day_entry)
            
        elif day_entry["image_id"] > image_id:
            break
    
    print('Scanning night results for id ', image_id)
            
    for night_entry in night_results:

        if night_entry["image_id"] == image_id:
            cocofile.append(night_entry)
            
        elif night_entry["image_id"] > image_id:
            break
        
    print('\n')
    
#%% 
jsonpath = '../../test_detections/day_night_normal_detection_results.json'        

with open(jsonpath, 'w') as outjson:
    json.dump(cocofile, outjson, indent=4)   

    
        
        
        
