import os
import sys
import json
import numpy as np
import cv2

ROOT_DIR = os.path.abspath("../")
# Import Mask
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn.config import Config


class SatelliteDataset(utils.Dataset):

    def __init__(self):
        """ Prepare dataset. """
        # This is just a template version.
        super(SatelliteDataset, self).__init__()

    def get_gt_info(self, idx, annotations, category_dict, updated_category_dict):
        # Get the groundtruth information for each image, including label and polygon
        object_list = []
        polygon_list = []
            
        # Go through all the annotations for certain image_id (idx)
        for anno in annotations:
            if idx == anno['image_id']:
                category_id = anno['category_id']
                category_name = category_dict[category_id] 
                    
                if category_name not in updated_category_dict:
                    # skip all the irrelevant classes.
                    continue
                object_list.append(updated_category_dict[category_name])
                poly = anno['segmentation'][0]
                polygon_list.append([poly[i:i+2] for i in range(0, len(poly), 2)])

        return object_list, polygon_list


    def read_json(self, json_file):
        with open(json_file) as f:
            data_info = json.load(f)
    
        return data_info


    def find_skip_classes(self, data_info, threshold):
        """
        skip those categories with number of instances less than theshold.
        """
        # create empyt dict 
        static = dict()
        id_name_map = dict()

        for cls in data_info['categories']:
            if cls['id'] not in id_name_map:
                id_name_map[cls['id']] = cls['name']


        for ins in data_info['annotations']:
            cls_name = id_name_map[ins['category_id']]
            if cls_name not in static:
                static[cls_name] = 0
            else:
                static[cls_name] += 1

        skip_category =[]
        for idx in id_name_map:
            category_name = id_name_map[idx]
            if category_name in static:
                if static[category_name] < threshold:
                        skip_category.append(category_name)
            else:
                skip_category.append(category_name)

        return skip_category

    def config_dataset(self, dataset_name, json_file, skip_classes, root_dir):
        """
        Args:
            dataset_name: string
            json_file: string
            skip_classes: List[string]
        """





        
        # skip the category with 0 instances
        # skip_classes = find_skip_classes(data_info, threshold=1)            
        skip_classes = ['Running_Track', 'Flag', 'Train', 'Cricket_Pitch',\
                        'Horse_Track', 'Artillery', 'Racing_Track', \
                        'Power_Station', 'Refinery', 'Mosques', 'Prison',\
                        'Market/Bazaar', 'Quarry', 'School', 'Graveyard',\
                        'Rifle_Range', 'Train_Station', 'Telephone_Line',\
                        'Vehicle_Control_Point', 'Hospital']

        def get_category_dict(categories, 
                              skip_classes=['Road','Body_Of_water','Tree']):
           
            category_dict = dict() 
            updated_category_dict = dict() 
            id_counter = 1
            for category in categories:
                if category['id'] not in category_dict:
                     category_dict[category['id']] = category['name']
                
                if category['name'] not in updated_category_dict \
                                 and category['name'] not in skip_classes:
                     updated_category_dict[category['name']] = id_counter
                     id_counter += 1 

            return category_dict, updated_category_dict


        category_dict, updated_category_dict = get_category_dict(
                                                  data_info['categories'],                                                                  skip_classes=skip_classes)
        

        images_list = data_info['images']
        for i, img_dict in enumerate(images_list):
            image_path = os.path.join(root_dir, img_dict['coco_url'])
            width = img_dict['width']
            height = img_dict['height']
            object_list, polygon_list = get_gt_info(img_dict['id'], 
                                                    data_info['annotations'],
                                                    category_dict,
                                                    updated_category_dict)
            # skip all those empty frames
            if not object_list:
                continue
            
            self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )
 
        # add all the class info
        for category in updated_category_dict:
            # self.add_class(dataset_name, category['id'], category['name'])
            self.add_class(dataset_name, 
                           updated_category_dict[category], 
                           category)

        print(len(updated_category_dict))
        print(updated_category_dict)
        print(skip_classes)


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

            ## TO Be contnued for ignoring empty image

            # ======================================== # 
            cv2.fillPoly(black_ground, np.array([polygon]).astype(int), (1.,))
            # black_ground = cv2.resize(black_ground, (256, 256))
            # simply for debugging
            mask_array.append(black_ground)
        
        
        mask_array = np.array(mask_array).transpose(1, 2, 0).astype(bool)
        class_id_array = np.array(object_list)
        
        # print(mask_array.shape)
        return mask_array, class_id_array
        



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

 
    NUM_CLASSES = 1 + 45  # background + # classes

    # Use small images for fast traning. Set the limits of the small side and the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Use smaller anchors because our images and objects are small.
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Reduce ROIs per image because the images are small and have few objects. Aim to allow ROI sampling to pick 33%  positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple.
    STEPS_PER_EPOCH = 300
    # Use a small validation steps since the epoch is small
    VALIDATION_STEPS = 5







if __name__ == '__main__':

    dataset_train = SatelliteDataset()
    dataset_train.config_dataset(dataset_name='satellite', 
                             json_file='/home/ubuntu/data/satellite/sate/annotations/instances_train2018.json',
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir='/home/ubuntu/data/satellite/sate')


    config = SatelliteConfig()
    config.display()
