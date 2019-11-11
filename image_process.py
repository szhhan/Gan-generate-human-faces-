#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:02:01 2019

@author: ricky
"""

import math
import os
from urllib.request import urlretrieve
import zipfile
import shutil
import numpy as np
from PIL import Image


def download_and_unzip(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
    extract_path = os.path.join(data_path, 'img_align_celeba')
    save_path = os.path.join(data_path, 'celeba.zip')

    if os.path.exists(extract_path):
        print('Data already exists')
        return

    if not os.path.exists(save_path):
        print("Downloading..")
        urlretrieve(url,save_path)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    try:
        with zipfile.ZipFile(save_path) as zf:
            print("Extracting...")
            zf.extractall(data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  
        raise err

def image_process(image_path, width, height):
    image = Image.open(image_path)

    if image.size != (width, height): 
       face_width = face_height = 108
       j = (image.size[0] - face_width) // 2
       i = (image.size[1] - face_height) // 2
       image = image.crop([j, i, j + face_width, i + face_height])
       image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert("RGB"))


def process_image_batch(image_files, width, height):
    dat = np.array(
        [image_process(sample_file, width, height) for sample_file in image_files]).astype(np.float32)
    return dat


def plot_example_image(images):
    dim = math.floor(np.sqrt(images.shape[0]))
    images = images.astype(np.uint8)
    images_new = np.reshape(images[:dim**2],(dim, dim, images.shape[1], images.shape[2], images.shape[3]))
    new_im = Image.new("RGB", (images.shape[1] * dim, images.shape[2] * dim))
    for col_i, col_images in enumerate(images_new):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, "RGB")
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im