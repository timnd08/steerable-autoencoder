# Generate 224x224 crops from ImageNet images for training the steerable autoencoder. 
# The crops are overlapping with stride 56px
# Eye movement vector is saved as part of crop name

import os
import itertools
from PIL import Image
from tqdm import tqdm


def create_crops(image_path, output_dir, crop_size=224):
    image = Image.open(image_path)
    width, height = image.size
    
    os.makedirs(output_dir, exist_ok=True)
    # Resize the image while maintaining the aspect ratio
    short_side = min(width, height)
    scale = 500 / short_side
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height))


    # Crop center
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    right = left + 448
    bottom = top + 448
    image = image.crop((left, top, right, bottom))

    # Crop into 224x224 patches with 50% overlap
    w,h = image.size
    for i, j in itertools.product(range(0, w-crop_size+1, crop_size//4), range(0, h-crop_size+1, crop_size//4)):
        box = (i, j, i + crop_size, j + crop_size)
        cropped_image = image.crop(box)
        cropped_image.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.png"))

if __name__ == "__main__":
    dataset = "train" #use "train" or "val" depending on which dataset you want to process
    input_dir = f"../imageNet/imagenet-mini/{dataset}/" #replace with ImageNet training directory
    output_dir = f"../imageNet/crops/{dataset}/" #replace with output directory for training crops
    count = 0
    for folder in tqdm(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        output_dir_cat = os.path.join(output_dir, folder)
        os.makedirs(output_dir_cat, exist_ok=True)
        for filename in os.listdir(folder_path):
            output_dir_img = os.path.join(output_dir_cat, os.path.splitext(filename)[0])
            create_crops(os.path.join(folder_path, filename), output_dir_img)
            count += 1

    print(f"Processed {count} train images.")