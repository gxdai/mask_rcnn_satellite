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
class SatelliteConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overwrides values specific to the toy shapes dataset.
    """
    # Give the configuration a recognizable name.
    NAME = "satellite"

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
    NUM_CLASSES = 1 + 68  # background + 3 shapes

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


config = SatelliteConfig()
config.display()



def get_ax(rows=1, cols=1, size=8):
    """visualization."""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax


class SatelliteDataset(utils.Dataset):

    def __init__(self):
        """ Prepare dataset. """
        # This is just a template version.
        super(SatelliteDataset, self).__init__()


    def config_dataset(self, dataset_name, json_file, root_dir):
        """
        Args:
            dataset_name: string
            json_file: string
        """

        def get_gt_info(idx, annotations):
            # Get the groundtruth information for each image, including label and polygon
            object_list = []
            polygon_list = []
            for anno in annotations:
                if idx == anno['image_id']:
                    object_list.append(anno['category_id'])
                    poly = anno['segmentation'][0]
                    polygon_list.append([poly[i:i+2] for i in range(0, len(poly), 2)])

            return object_list, polygon_list

        # prepare dataset from the json file.
        with open(json_file) as f:
            data_info = json.load(f)

        images_list = data_info['images']
        for i, img_dict in enumerate(images_list):
            image_path = os.path.join(root_dir, img_dict['coco_url'])
            width = img_dict['width']
            height = img_dict['height']
            object_list, polygon_list = get_gt_info(img_dict['id'], data_info['annotations'])
            
            self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )
 
        # add all the class info
        for category in data_info['categories']:
            self.add_class(dataset_name, category['id'], category['name'])


    def load_image(self, image_id):
        image = super(SatelliteDataset, self).load_image(image_id) 
        # resize the image for preprocessing
        # print(image.shape)
        # image = skimage.transform.resize(image, (256, 256))
        # print(image.shape) 
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
        
        object_list = self.image_info[image_id]['object_list']
        polygon_list = self.image_info[image_id]['polygon_list']
        width = self.image_info[image_id]['width']
        height = self.image_info[image_id]['height']
        # print(self.image_info[image_id]['path'])

        # print("object_list = {}".format(object_list))
        # print("polygon_list = {}".format(polygon_list))
        assert len(object_list) == len(polygon_list), "object number and ploy number doesn't match"
      
        mask_array = []
        class_id_array = []
        
        for label, polygon in zip(object_list, polygon_list):
            black_ground = np.zeros((height, width))
            cv2.fillPoly(black_ground, np.array([polygon]).astype(int), (1.,))
            # black_ground = cv2.resize(black_ground, (256, 256))
            # simply for debugging
            mask_array.append(black_ground)
        
        
        mask_array = np.array(mask_array).transpose(1, 2, 0).astype(bool)
        class_id_array = np.array(object_list)
        
        # print(mask_array.shape)
        return mask_array, class_id_array
        


dataset_val = SatelliteDataset()
dataset_val.config_dataset(dataset_name='satellite', 
                                 json_file='./001/val.json',
                                 root_dir='./001')

dataset_val.prepare()


# create model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

class InferenceConfig(SatelliteConfig):
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

GT = 'GT_dir'
Prediction = './result/Prediction'

if not os.path.isdir(GT):
    os.makedirs(GT)

if not os.path.isdir(Prediction):
    os.makedirs(Prediction)

for image_id in dataset_val.image_ids:
    # Get path for image with image_id
    path = dataset_val.image_info[image_id]['path']
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    # resize the original image
    # original_image = cv2.resize(original_image, (256, 256))
    filename = dataset_val.image_info[image_id]['path'].split('/')[-1]
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_val.class_names, figsize=(8,8),
                            filename=os.path.join(GT, filename))

    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], figsize=(8,8), 
                            filename=os.path.join(Prediction, filename))



"""

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(mask_GT)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(mask_Prediction)
fig.savefig('compare.png')


# Evaluation
# image_ids = np.random.choice(dataset_val.image_ids, 10)
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
"""
