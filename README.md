## Data Preparing

Your dataset should looks like:

```
$your_dataset_path
    |──class1
        |──xxxx.jpg
        |──...
    |──class2
        |──xxxx.jpg
        |──...
    |──...
    |──classN
        |──xxxx.jpg
        |──...
```

The you can use ```tools/generate_list.py``` to generate list of training samples. Here we do not use ```torchvision.datasets.ImageFolder``` because it is very slow when dataset is pretty large. You can run

```shell
python tools/generate_list.py --name {name your dataset such as caltech256} --path {path to your dataset}
```

Then two files will be generated under ```list``` folder, one  ```*_list.txt``` save every image path and its class(here no use); one ```*_name.txt``` save index of every class and its class name.


## Train and Evaluate

For training

```shell
bash run/train.sh {model architecture such as vgg16} {you dataset name}
# For example
bash run/train.sh vgg16 caltech256
```

For evaluating single checkpoint:

```shell
bash run/eval.sh {model architecture} {checkpoint path} {dataset name}
# For example
bash run/eval.sh vgg16 results/caltech256-vgg16/099.pth caltech101
```

For evaluating all checkpoints under specific folder:

```shell
bash run/evalall.sh {model architecture} {checkpoints path} {dataset name}
# For example
bash run/evalall.sh vgg16 results/caltech256-vgg16/ caltech101
```
When all checkpoints are evaluated, a scatter diagram ```figs/evalall.jpg``` will be generated to show the evaluate loss trend.

For model architecture, now we support ```vgg11,vgg13,vgg16,vgg19``` and ```resnet18, resnet34, resnet50, resnet101, resnet152```.


# Tools

We provide several tools to better visualize the auto-encoder results.

```reconstruct.py```

Reconstruct images from original one. This code will sample 64 of them and save the comparison results to ```figs/reconstruction.jpg```.

```shell
python tools/reconstruct.py --arch {model architecture} --resume {checkpoint path} --val_list {*_list.txt of your dataset}
# For example
python tools/reconstruct_recenter.py --arch vgg16 --resume results/trainCoco-vgg16/056.pth --val_list list/valCoco_list.txt
```
# Experiment
```shell
python experiment/experimentConductor.py --arch vgg16 --resume results/trainCoco-vgg16/299.pth --val_list list/unseen_list.txt
```
