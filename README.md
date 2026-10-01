# Install Anaconda + Set up conda environment 
This project uses **Conda** to manage its runtime environment, think of it as a sandbox for your software.  
Conda installation instruction can be found here (if you haven't installed):
```
https://www.anaconda.com/docs/getting-started/installation
```

To generate a conda environment (aka create a sandbox)
In the terminal:
```
conda create --name steerable-autoencoder python=3.11
```

To activate the newly created conda environment:
```
conda activate steerable-autoencoder
```

# Install required Python/Python packages 
Python 3.11

To install required libraries:
```
pip install -r requirements.txt
```
*Note: Not all libraries are required, some are for attempted experiments and visualizations, feel free to remove the ones not needed. If not sure, just install them all.*

# Genertate ImageNet Mini Pre-train Dataset
In this project, we generate a dataset that represents definite eye movements of 56px by cropping overlapping images from ImageNet. We use ImageNet Mini since it's sufficient and no need for full ImageNet dataset.

Download ImageNet Mini from here:
```
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
*ImageNet is also acceptable, just a bit overkill.*
```
Unzip and save in folder **\imageNet**
You should now have folder that looks something like this
```
$\imageNet\imagenet-mini
                    |──train
                            |──class1
                                |──xxxx.jpg
                                |──...
                            |──classN
                                |──xxxx.jpg
                                |──...
                    |──val
                            |──class1
                                |──xxxx.jpg
                                |──...
                            |──classN
                                |──xxxx.jpg
                                |──...
```

Generate Pre-train dataset from ImageMini
```
python imageNet/imageGenerator.py
```
*Note: To change whether generating train set or val dataset, change the variable **dataset***

# Generate black background dataset from COCO dataset
For main training dataset, we generate a dataset with objects at random locations on black background from COCO dataset.  
In this project, we use COCO 2017 Train and Val images, along with 2017 Train/Val annotations.  
COCO dataset can be found here.
```
https://cocodataset.org/#download
```
Download, save and unzip in folder **coco/**.  
Folder **coco/** should now have following folders **train2017, val2017, annotations_trainval2017**.

To generate black black background dataset:
```
python coco/objectCutter_black.py
```
*Note: To change whether generating train set or val dataset, change the variable **dataset_type** in file **python imageGenerator.py***