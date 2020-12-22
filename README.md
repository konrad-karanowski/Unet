# Unet

Unet is powerful image segmentation algorithm. In the beginning it was mainly use for segmentation of histopathological images.
Checkout this [paper](https://arxiv.org/pdf/1505.04597.pdf). 
Mine version is slightly different, as the input and output has the same size (256x256).

![example](https://user-images.githubusercontent.com/55788735/102884000-280e5600-4451-11eb-9afa-6a4afd91c036.png)



### Installation

Our version of UNet is build with [PyTorch](https://pytorch.org/), preferable way of installing Torch is using [conda](https://www.anaconda.com/). Other dependencies can be easily installed follows: 

```sh
$ pip install -r requirements.txt
```

### Data structure

Data set should be placed in two directories: *train* and *validation*. Each directory should contain two subdirectories: *inputs* that contain every inputs images for model (the preferred format is PNG) and *masks* that contain more subdirectories representing classes. Every class directory contains masks corresponding to it. 
Masks are binary images with only two values: 0 and 1. 1 means, that certain pixel belongs to class and 0 means, that do not. Masks MUST to be named the same as image they refer to. 
Example data set structure may look as follows: 

```sh
    ├───callbacks
    ├───data
    ├───train
    │   ├───inputs
    │   │   ├─── 0.png
    │   │   └─── 1.png
    │   └───masks
    │       ├───class1
    │       │   ├─── 0.png
    │       │   └─── 1.png
    │       └───class2
    │           ├─── 0.png
    │           └─── 1.png
    └───validate
        ├───inputs
        │   └─── test0.png
        └───masks
            ├───class1
            │   └─── test0.png
            └───class2
                └─── test0.png
```

### Create background
Sometimes, you want to have background as separate class. Luckily, you don't have to create this by hand. To create background class for all your images in *data* folder, just use this script:

```sh
$ python create_background.py
```

If there is no *background* folder in *train/masks* and/or *validate/masks*, it will be created for you. Background is everything that masks do not cover.

### How to train model

Usage: 

```sh
$ train.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-lr ETA] [-c N_CALLBACKS] [-m MOMENTUM] [-l MODEL_PATH] [-g AS_GRAY]
```

You can specify training parameters such as:

```sh
optional arguments
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 5)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size (default: 1)
  -lr ETA, --learning_rate ETA
                        Learning rate (default: 0.0001)
  -c N_CALLBACKS, --callbacks N_CALLBACKS
                        How many epochs to save weights (default: 1)
  -m MOMENTUM, --momentum MOMENTUM
                        Nesterov momentum value (default: 0.9)
  -l MODEL_PATH, --load MODEL_PATH
                        Path to weights if you want to use pretrained model. (default: None)
  -g AS_GRAY, --as_gray AS_GRAY
                        Whether the pictures should be converted to gray (True) or coloured (False) (default: False)

```

For example, I want to train completely new model in 100 epochs, save weights every 5 epochs, with learning rate 0.001 and convert all my images to gray: 
```sh
$ python train.py -e 100 -c 5 -lr 0.001 -g True
```

At the end of training, program will always save final model. All callbacks will be stored as [model state dicts](https://pytorch.org/tutorials/beginner/saving_loading_models.html) in folder *callbacks* (watch "Data structure").


### Metrics
Metrics of model performance are:
- fuzzy version of IoU - ratio of the intersection to the union
- dice coefficient - F1 score for image segmentation
- loss function value - Dice loss (1 - dice coefficient)

Great video explaining this metrics: [click](https://www.youtube.com/watch?v=AZr64OxshLo&t=797s).

### How to predict

After training model you can use try it on images. To run prediction use this script:
```sh
$ predict.py [-h] -c NUM_CLASSES -m MODEL_PATH -i INPUT_PATH [-g AS_GRAY] [-t THRESHOLD] [-o OUTPUT]
```

You can specify parameters such as:

```
  -h, --help            show this help message and exit
required arguments:
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        On how many classes model was trained (default: None)
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to pretrained weights (default: None)
  -i INPUT_PATH, --input INPUT_PATH
                        Path to image or directory with images (default: None)
optional arguments:
  -g AS_GRAY, --gray AS_GRAY
                        Whether input images are gray or coloured (default: False)
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for which we want to classify pixel as belonging to the class (default: 0.5)
  -o OUTPUT, --output OUTPUT
                        Where do you want to save output (if None, predictions won't be saved) (default: None)

```

For example I want to use my final model from last training (stored as *model.pth* in *callbacks*), which was using gray images and 3 output classes on single image *image.png* and save it in the same directory:
```
$ python predict.py -c 3 -g True -m callbacks/model.pth -i image.png -o .
```

### Todos
 - Change SummaryWriter class