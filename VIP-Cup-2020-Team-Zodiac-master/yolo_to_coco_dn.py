import os
import argparse
import json
import cv2
import pandas as pd

#%% 
def create_coco(imagefolder, labelfolder, jsonpath):

    cocofile = {
        "info":{
            "year": 2020,
            "version": 1,
            "description": "VIP CUP 2020"
        },
        "categories":[
            {"id": 0, "name": "day_vehicle"},
            {"id": 1, "name": "night_vehicle"},
        ],
        "images":[],
        "annotations": []
    }
    
    imagenames = sorted(os.listdir(imagefolder))

    ann_id = 0
    for image_id, imgnm in enumerate(imagenames):
        print("New image %d . \n", image_id)
        imagepath = os.path.join(imagefolder, imgnm)
        image = cv2.imread(imagepath)
        img_height, img_width = image.shape[:-1]
        image_info = {
            "id": image_id,
            "file_name": imgnm,
            "height": img_height,
            "width": img_width
        }
        cocofile["images"].append(image_info)

        basename = os.path.splitext(imgnm)[0]
        labelname = f'{basename}.txt'
        labelpath = os.path.join(labelfolder, labelname)

        labels = pd.read_csv(
            labelpath,
            delim_whitespace=True,
            names=["category_id", "xcentroid", "ycentroid", "boxwidth", "boxheight"]
        )

        labels = labels * [1, img_width, img_height, img_width, img_height]
        labels["xtopleft"] = labels["xcentroid"] - labels["boxwidth"]/2
        labels["ytopleft"] = labels["ycentroid"] - labels["boxheight"]/2
        
        #########Checking Day vehicle or Night Vehicle
        ctg_id = checkDN(imgnm)

        for i, row in labels.iterrows():
            ann_info = {
                "id": ann_id,
                "image_id": image_id,
                #"category_id": int(row["category_id"]),
                "category_id": ctg_id,
                "bbox": [row["xtopleft"], row["ytopleft"], row["boxwidth"], row["boxheight"]],
                "area": row["boxwidth"] * row["boxheight"],
                "segmentation": [],
                "iscrowd": 0
            }
            cocofile["annotations"].append(ann_info)
            ann_id += 1
    
    with open(jsonpath, 'w') as outjson:
        json.dump(cocofile, outjson, indent=4)

    return None
   
def checkDN(image_name):
    if image_name.startswith('CLIP'):
        return 1
    else:
        return 0
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create coco formatted json labels")

    parser.add_argument("imagefolder", help="path to the existing imagefolder")
    parser.add_argument("labelfolder", help="path to the existing labelfolder")
    parser.add_argument("jsonpath", help="path to the json file (to be created)")
    args = parser.parse_args()

    create_coco(args.imagefolder, args.labelfolder, args.jsonpath)
    

    
    
    
    
    