import os
import sys
import subprocess
import random
import math
import re
import time
import numpy as np
import cv2
import json
import matplotlib
import matplotlib.pyplot as plt
import skimage.transform

ROOT_DIR = os.path.abspath("../")

# Import Mask
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from satellite import SatelliteDataset, SatelliteConfig

# directory to save logs and trained model


checkpoint_dir = 'logs/model_search'
MODEL_DIR = os.path.join(ROOT_DIR, checkpoint_dir)


config = SatelliteConfig()
config.display()



def get_ax(rows=1, cols=1, size=8):
    """visualization."""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax


       


dataset_val = SatelliteDataset()
dataset_val.config_dataset(dataset_name='satellite', 
                                 json_file='/home/ubuntu/data/satellite/sate/annotations/instances_val2018.json',
                                 skip_classes=['Road','Body_Of_water','Tree'],
                                 root_dir='./001')

dataset_val.prepare()


class InferenceConfig(SatelliteConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

# model_path = model.find_last()
model_path = "/home/ubuntu/mask_rcnn/logs/model_search/satellite20190103T0208/mask_rcnn_satellite_0005.h5"
print("Load weghts from ", model_path)
print("Specify model path")
model.load_weights(model_path, by_name=True)



image_root_dir = "/home/ubuntu/data/satellite/sliced_img"
image_list = os.listdir(image_root_dir) 
pred_dir = '/home/ubuntu/data/satellite/sliced_pred_search'

"""
print("Clean up prediction directory")
bash_cmd = "rm -rf " + pred_dir
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
"""

if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)

original_image_shape = (1024, 1024)
for filename in image_list:
    file_path = os.path.join(pred_dir, filename)
    if os.path.isfile(file_path):
        # skip the file
        continue
    # Get path for image with image_id
    img_path = os.path.join(image_root_dir, filename)
    original_image = \
        modellib.load_image_only(img_path, inference_config)

    print(original_image.shape)

    # original_image = cv2.resize(original_image, (256, 256))
    results = model.detect([original_image], verbose=1)
    r = results[0]

    pred_image = visualize.save_instances(original_image, r['rois'], r['masks'],
                                          r['class_ids'],dataset_val.class_names, 
                                          r['scores'], 
                                          original_image_shape=original_image_shape) 

    print(os.path.join(pred_dir, filename))
    cv2.imwrite(os.path.join(pred_dir, filename), pred_image)

