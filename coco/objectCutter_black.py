import numpy as np
import cv2
import pycocotools.mask as mask
from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
import random

dataset_type = 'train' #use "train" or "val" depending on which dataset you want to process
annotation_file = f'coco/annotations/instances_{dataset_type}2017.json'
coco = COCO(annotation_file)

def imgPadder(img, cent_x, cent_y):
    h, w = img.shape[:2]
    target_cent = 112
    shift_x = target_cent - cent_x
    shift_y = target_cent - cent_y
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    padded_img = cv2.warpAffine(img, M, (224, 224))

    return padded_img

def imgVariator(cropped_mask, cropped_img, img_id, i):
    cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
    cropped_mask = cv2.resize(cropped_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    coord_y, coord_x = np.where(cropped_mask > 0)
    out_dir = ''

    # Only process if there are valid coordinates in the mask
    if len(coord_x) > 0:
        #masked_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
        out_dir = os.path.join(f'{dataset_type}Coco_black',img_id,img_id+str(i))
        os.makedirs(out_dir, exist_ok=True)

        #calculate the center of the object in the cropped image
        min_x, max_x = np.min(coord_x), np.max(coord_x)
        min_y, max_y = np.min(coord_y), np.max(coord_y)
        cent_x, cent_y = (min_x + max_x) // 2, (min_y + max_y) // 2

        #save the original cropped image
        global count
        #masked_img = cropped_img
        masked_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
        img_name1 = img_id + '_' + str(cent_x) + '_' + str(cent_y) + '.png'
        cv2.imwrite(os.path.join(out_dir,img_name1), masked_img)
        

        masked_img_cent = imgPadder(cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask), cent_x, cent_y)
        img_name2 = img_id + '_112_112.png'
        cv2.imwrite(os.path.join(out_dir,img_name2), masked_img_cent)
        

        img_counter = len(os.listdir(out_dir))
        while img_counter < 4:

            # mask = cv2.cvtColor(masked_img_cent, cv2.COLOR_BGR2GRAY)
            # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            if np.sum(cropped_mask) == 0:
                print(f"Warning: Mask for {img_id} is empty. Skipping augmentation.")
                break
            cor_y, cor_x = np.where(cropped_mask > 0)
            min_x, max_x = np.min(cor_x), np.max(cor_x)
            min_y, max_y = np.min(cor_y), np.max(cor_y)
            
            # Randomly shift the image 
            new_x, new_y = random.randint(0, 224) , random.randint(0, 224)
            shift_x = new_x - cent_x
            shift_y = new_y - cent_y
            

            # Create a binary mask for the shifted image
            # shifted_mask = np.zeros_like(cropped_mask)
            # shifted_mask[shift_y:, shift_x:] = cropped_mask[:-shift_y, :-shift_x]
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted_img = cv2.warpAffine(cropped_img, M, (224, 224))
            shifted_mask = cv2.warpAffine(cropped_mask, M, (224, 224), flags=cv2.INTER_NEAREST)
            #mask_bg = cv2.bitwise_not(shifted_mask)

            # Crop the shifted image
            shifted_img = cv2.bitwise_and(shifted_img, shifted_img, mask=shifted_mask)
            global img_list
            #rando_bg_path = os.path.join(f'{dataset_type}2017/', random.choice(img_list))
            #bg = cv2.resize(cv2.imread(rando_bg_path), (224, 224), interpolation=cv2.INTER_AREA)
            #bg = cv2.bitwise_and(bg, bg, mask=mask_bg)

            # Use the mask to combine the shifted image with a random background
            #shifted_img = cv2.add(bg, shifted_img)


            img_name_shift = img_id + '_' + str(new_x) + '_' + str(new_y) + '.png'

            if not os.path.exists(os.path.join(out_dir, img_name_shift)):
                cv2.imwrite(os.path.join(out_dir,img_name_shift), shifted_img)
                img_counter += 1


        if len(os.listdir(out_dir)) != 4:
            print(f"Warning: {out_dir} has {len(os.listdir(out_dir))} images instead of 4.")
            print(f"Proceed to delete the dir")
            for file in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, file))
            os.rmdir(out_dir)
        
        else:
            count += len(os.listdir(out_dir))
    
    if os.path.exists(out_dir):
        if len(os.listdir(out_dir)) == 0:
            os.rmdir(out_dir)

def imgProcessor(img_name,img_path):
    #get annotation list
    img_id = img_name.split('.')[0]
    ann_list = coco.getAnnIds(imgIds=[int(img_id)])

    #get image
    og_img = cv2.imread(img_path)
    h, w = og_img.shape[:2]
    cropped_size = min(w, h)
    cent_x, cent_y = w//2, h//2
    x1 = max(0, cent_x - cropped_size//2)
    y1 = max(0, cent_y - cropped_size//2)
    x2 = min(w, x1 + cropped_size)
    y2 = min(h, y1 + cropped_size)
    cropped_img = og_img[y1:y2, x1:x2]

    
    #tackle each annotation for the image
    
    for i, ann_id in enumerate(ann_list):
        ann = coco.loadAnns(ann_id)[0]
        binary_mask = coco.annToMask(ann)
        binary_mask = (binary_mask * 255).astype(np.uint8)
        cropped_mask = binary_mask[y1:y2, x1:x2]

        imgVariator(cropped_mask, cropped_img, img_id, i)
        
global img_list
img_list = os.listdir(f'coco/{dataset_type}2017/')
global count 
count = 0
for img_name in tqdm(img_list):
    img_path = os.path.join(f'coco/{dataset_type}2017/', img_name)
    imgProcessor(img_name, img_path)
    
print(f"Produced {count} images.")
