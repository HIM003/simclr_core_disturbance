#!/usr/bin/env python
# coding: utf-8
#
# Script used for augmenting images within a directory 
#

import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os
from os import walk

path = r"/jmain02/home/J2AD015/axf03/hxm18-axf03/images/for_training_exp385_further_aug"
os.chdir(path)

f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

for dirname in dirnames:
    temp_path = os.path.join(path,dirname)
    f = []
    for (dirpath2, dirnames2, filenames2) in walk(temp_path):
        f.extend(filenames2)
        break
    for ff in f:
        print(os.path.join(path,dirname,ff))  
        image = plt.imread(os.path.join(path,dirname,ff))
        
        transform = A.HorizontalFlip(p=1)
        hor_image = transform(image=image)['image']
        cv2.imwrite(os.path.join(path,dirname,ff.split(".jpg")[0]+"_hor.jpg"), hor_image)
        
        transform = A.VerticalFlip(p=1)
        vert_image = transform(image=image)['image']
        cv2.imwrite(os.path.join(path,dirname,ff.split(".jpg")[0]+"_vert.jpg"), vert_image)
        
        transform = A.VerticalFlip(p=1)
        hor_vert_image = transform(image=hor_image)['image']
        cv2.imwrite(os.path.join(path,dirname,ff.split(".jpg")[0]+"_hor_vert.jpg"), hor_vert_image)

        transform = A.Compose([A.Blur(blur_limit=3),
            A.CLAHE(),
            A.ChannelShuffle(),
            A.Emboss(),
            A.GaussianBlur(),
            A.RandomContrast(),
            A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
        ])
        random.seed(42) 
        augmented_image = transform(image=image)['image']
        cv2.imwrite(os.path.join(path,dirname,ff.split(".jpg")[0]+"_aug.jpg"), augmented_image)
