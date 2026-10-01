# Install requirements Python/Python packages
Python 3.11

To install required libraries:
```
pip install -r requirements.txt
```
*Note: Not all libraries are required, some are for attempted experiments and visualizations, feel free to remove the ones not needed. If not sure, just install them all*

# Pre-train ImageNet Mini Dataset
In this project, we generate definite eye movements of 56px by cropping overlapping images from ImageNet. We use ImageNet Mini since it's sufficient and no need for full ImageNet dataset.

Download ImageNet Mini from here: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
Save in folder
```
\imageNet-mini
```
*ImageNet is also acceptable, just a bit overkill.*