import os
import sys
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

ROOT_DIR = os.path.abspath("../../")

# Import Mask
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download coco trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# Configuration
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overwrides values specific to the toy shapes dataset.
    """
    # Give the configuration a recognizable name.
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We put multiple images on each GPU because the images are small.
    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes
    """
    color_dict = {'background': (255, 255, 255), 'Tree': (128, 255, 0),\
              'Parking_Lot': (0, 0, 204), 'Road': (204, 0, 204), \
              'Vehicles': (255, 0, 0), 'Well': (255, 255, 0), \
              'Buildings': (0, 255, 255)
             }
    """
    # background
    # Tree
    # Vehicles
    # Building 
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for fast traning. Set the limits of the small side and the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    # Use smaller anchors because our images and objects are small.
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Reduce ROIs per image because the images are small and have few objects. Aim to allow ROI sampling to pick 33%  positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple.
    STEPS_PER_EPOCH = 100
    # Use a small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()



def get_ax(rows=1, cols=1, size=8):
    """visualization."""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax

# Extend synthetic datasets.

"""Overwrite
load_image()
load_mask()
image_reference()
"""


class SatelliteDataset(utils.Dataset):

    def __init__(self):
        """ Prepare dataset. """
        # This is just a template version.
        super(SatelliteDataset, self).__init__()

        # add class
        self.add_class("satellite", 1, "Tree")
        self.add_class("satellite", 2, "Vehicles")
        self.add_class("satellite", 3, "Buildings")
        
        # add image
        for _ in range(500):
            self.add_image(source="satellite", image_id=0,
                       path="./mask_annotation/source/a2.jpg",
                       mask_path="new_mask.png", 
                       json_path="./mask_annotation/area/a2_area.json")
                        

    def load_image(self, image_id):
        image = super(SatelliteDataset, self).load_image(image_id) 
        # resize the image for preprocessing
        # print(image.shape)
        image = skimage.transform.resize(image, (256, 256))
        
        return image

    def load_mask(self, image_id):
        """
        ============================================================
           For this preliminary task, we only work on three classes
           
                        Tree,
                        Vehicles,
                        Buildings,
        ===========================================================
        """
        class_dict = {'Tree': 1, 'Vehicles': 2, 'Buildings': 3}
        json_file = self.image_info[image_id]['json_path']
      
        with open(json_file) as f:
            json_info = json.load(f)
        mask_array = []
        class_id_array = []
        for obj in json_info['objects']:
            if obj['label'] not in class_dict:
                continue
            black_ground = np.zeros((json_info['imgHeight'],
                                   json_info['imgWidth']))
            cv2.fillPoly(black_ground, np.array([obj['polygon']]).astype(int), (1.,))
            # for test, I just resize the mask into a smaller scaler
            black_ground = cv2.resize(black_ground, (256, 256))
            # simply for debugging
            mask_array.append(black_ground)
            class_id_array.append(class_dict[obj['label']])
        
        
        mask_array = np.array(mask_array).transpose(1, 2, 0).astype(bool)
        class_id_array = np.array(class_id_array)
        # print("mask_array.shape = {}".format(mask_array.shape))
        # print("class_id_array.shape = {}".format(class_id_array.shape))
        return mask_array, class_id_array
        
satellite_dataset = SatelliteDataset()
# print(satellite_dataset.class_info)
# print(satellite_dataset.image_info)
img = satellite_dataset.load_image(0)
mask, cls = satellite_dataset.load_mask(0)


# print('prepare dataset')
dataset_train = SatelliteDataset()
dataset_train.prepare()

dataset_val = SatelliteDataset()
dataset_val.prepare()
"""

print('visualize dataset')

image_ids = [0]
print(image_ids)
for image_id in image_ids:
    print(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
"""

# create model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
init_with = 'coco'

if init_with == 'imagenet':
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == 'coco':
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', "mrcnn_mask"])
elif init_with == 'last':
    model.load_weights(model.find_last(), by_name=True)

# Skip training, Only do validation.
"""
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE /10,
            epochs=5,
            layers='all')
"""
print("Finish training")
# sys.exit()



class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()
print("Load weghts from ", model_path)
model.load_weights(model_path, by_name=True)


image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
print(image_meta, gt_class_id, gt_bbox)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print("END of Log")
# resize the original image
# original_image = cv2.resize(original_image, (256, 256))
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8,8))
results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'])


# Evaluation
image_ids = np.random.choice(dataset_val.image_ids, 10)

APs = []
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    resuts = model.detect([image], verbose=0)
    r = resuts[0]
    AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                    r['rois'], r["class_ids"], r["scores"], r["masks"])
    APs.append(AP)

print("mAP: ", np.mean(APs))
