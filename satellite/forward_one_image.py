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
from satellite import SatelliteDataset, SatelliteConfig, InferenceConfig



# read dataset information from json file, including class name, class id ...
dataset_val = SatelliteDataset()
dataset_val.config_dataset(dataset_name='satellite',
                                 json_file='/home/ubuntu/data/satellite/sate/annotations/instances_val2018.json',
                                 skip_classes=['Road','Body_Of_water','Tree'],
                                 root_dir='./001')

dataset_val.prepare()


inference_config = InferenceConfig()

# create model
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir='./tmp')

# path for model parameters
model_path = "/home/ubuntu/mask_rcnn/logs/model_search/satellite20190103T0208/mask_rcnn_satellite_0005.h5"

# load parameters into model 
model.load_weights(model_path, by_name=True)

# image path to be predicted
image_path = "/home/ubuntu/data/satellite/sliced_img/5120_6144_55296_56320.png"

# load image
original_image = modellib.load_image_only(image_path, inference_config)

image_shape = original_image.shape
print(original_image.shape)
# forward detection
results = model.detect([original_image], verbose=1)
r = results[0]

# process results
pred_image = visualize.save_instances(original_image, r['rois'], r['masks'],
                                          r['class_ids'],dataset_val.class_names, 
                                          r['scores'], 
                                          original_image_shape=(1024, 1024))

print(pred_image.shape)
cv2.imwrite("out.jpg", pred_image)

