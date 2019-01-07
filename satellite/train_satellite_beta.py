import os
import sys
import argparse
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






parser = argparse.ArgumentParser(description='For satellite image dataset')
parser.add_argument('--checkpoint_dir', type=str, default='logs/model_65_001_007',
                    help='directory for saving all the checkpoint file.')

parser.add_argument('--dataset_name', type=str, default='satellite')
parser.add_argument('--train_json', type=str, default='train.json')
parser.add_argument('--test_json', type=str, default='test.json')
parser.add_argument('--image_root_dir', type=str, default='./')
def main():
    ROOT_DIR = os.path.abspath("../")

    # Import Mask
    sys.path.append(ROOT_DIR)
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn.model import log
    from satellite import SatelliteDataset, SatelliteConfig

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir 
    dataset_name = args.dataset_name
    train_json = args.train_json
    test_json = args.test_json
    image_root_dir = args.image_root_dir

    MODEL_DIR = os.path.join(ROOT_DIR, checkpoint_dir)

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    print("COCO_MODEL_PATH = {}".format(COCO_MODEL_PATH))
    # Download coco trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # config dataset
    config = SatelliteConfig()
    config.display()


    dataset_train = SatelliteDataset()
    dataset_train.config_dataset(dataset_name=dataset_name, 
                             json_file=train_json,
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir=image_root_dir)



    # prepare dataset
    dataset_train.prepare()


    dataset_val = SatelliteDataset()
    dataset_val.config_dataset(dataset_name=dataset_name, 
                             json_file=test_json,
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir=image_root_dir)

    dataset_val.prepare()



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
    # Skip training, Only do validation.
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')
    # fine-tune
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE /10,
            epochs=5,
            layers='all')
    """ 
    print("Finish training")

    print("Validation")
    class InferenceConfig(SatelliteConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()
    # inference_config.GPU_COUNT = 1
    # inference_config.IMAGES_PER_GPU = 1
    # inference_config.BATCH_SIZE = 1
    model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

    model_path = model.find_last()
    print("Load weghts from ", model_path)
    model.load_weights(model_path, by_name=True)


    # prepare dataset

# Evaluation

    APs = []
    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # print(model.config.BATCH_SIZE)
        resuts = model.detect([image], verbose=0)
        r = resuts[0]
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                    r['rois'], r["class_ids"], r["scores"], r["masks"])
        APs.append(AP)

    print("mAP: ", np.mean(APs))

if __name__ == '__main__':
    main()
