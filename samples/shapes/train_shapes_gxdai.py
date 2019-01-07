import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

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
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for fast traning. Set the limits of the small side and the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # Use smaller anchors because our images and objects are small.
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Reduce ROIs per image because the images are small and have few objects. Aim to allow ROI sampling to pick 33%  positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

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
class ShapesDataset(utils.Dataset):
    """Synthetic dataset."""
    def load_shapes(self, count, height, width):
        # add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # add images
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image id.
        In practice, load an natural image.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape((1, 1, 3))
        # image = np.ones((info['height'], info['width'], 3), dtype=np.uint8)
        image = np.ones((info['height'], info['width'], 3), dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)

        return image

    def draw_shape_gxdai(self, image, shape, dims, color):
        x, y, s = dims
        print("type(image) = {}".format(type(image)))
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == 'triangle':
            points = np.array([[(x, y-s),
                               (x-s/math.sin(math.radians(60)), y+s),
                               (x+s/math.sin(math.radians(60)), y+s)
                               ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)

        return image
    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                 (x-s/math.sin(math.radians(60)), y+s),
                 (x+s/math.sin(math.radians(60)), y+s),
                 ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)

        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(), shape, dims, 1)

        occlusion = np.logical_not(mask[:,:,-1]).astype(np.uint8)

        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:,:,i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:,:,i]))

        """Map class names to class id."""
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])

        return mask.astype(np.bool), class_ids.astype(np.int32)

    def load_mask_original(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """

        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(), shape, dims, 1)
            # Handle occlusions occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8) for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])

        return mask.astype(np.bool), class_ids.astype(np.int32)



    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)



    def random_shape(self, height, width):
        """Generate specifications of a random shape, that lies within
        the given hieght and boundary.
        Return a tuple of three values:
        * The shape name (square, circle, ...)
        * shape color:
        * Shape dims:
        """
        shape = random.choice(["square", "circle", "triangle"])
        color = tuple([random.randint(0, 255) for _ in range(3)])
        buffer = 20
        """center"""
        y = random.randint(buffer, height-buffer-1)
        x = random.randint(buffer, width-buffer-1)
        """size"""
        s = random.randint(buffer, height//4)
        # s = random.randint(0, buffer)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Generate random images."""
        # randomly pick bg color (3 channels)
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])

        # genereate shapes and record their bouding boxes
        shapes = []
        boxes = []
        N = random.randint(1,4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        """ threshold = 0.3 for non-max suppersion."""
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]

        return bg_color, shapes
print('prepare dataset')
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()


dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
"""
print('visualize dataset')

image_ids = np.random.choice(dataset_train.image_ids, 4)
print(image_ids)
for image_id in image_ids:
    print(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()



# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
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
