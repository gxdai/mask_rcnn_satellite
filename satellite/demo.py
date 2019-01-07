import os
import sys
import random
import math
import skimage.io
import matplotlib.pyplot as plt
# root directory of the project
ROOT_DIR = os.path.abspath('../')

# import mask rcnn
sys.path.append(ROOT_DIR)

# import local libraries
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# import coco
sys.path.append(os.path.join(ROOT_DIR, "samples/coco"))
import coco

# Directory for log and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# local path to trained weights file.
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# IMAGE DIRECTORY
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Configuire

class InferenceConfig(coco.CocoConfig):
    # set batch size = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# create model and load pretrained weight
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# print class name

# load coco dataset
"""
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, 'train')
dataset.prepare()

print(dataset.class_names)
"""
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        class_names, r['scores'])
