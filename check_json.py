import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
import sys

json_file = "/home/gxdai/MMVC_LARGE/Guoxian_Dai/sourcecode/Mask_RCNN/001/annotations/train.json"
root_dir = '/home/gxdai/MMVC_LARGE/Guoxian_Dai/sourcecode/Mask_RCNN/001'
with open(json_file) as f:
    annotation = json.load(f)

print(type(annotation))
print(annotation['annotations'][0]['image_id'])


# check the iscrowd

for anno in annotation['annotations']:
    if len(anno['segmentation']) > 1:
        print("="*20)
        print("anno['iscrowd'] = {}".format(anno['iscrowd']))
        print("="*20)
        print(len(anno['segmentation']))
        print("="*20)
        print("anno['category_id']= {}".format(anno['category_id']))
print("finish")




id_info = annotation['annotations'][0]['image_id']
cate_id_info = annotation['annotations'][0]['category_id']
print("id_info = {:5d}".format(id_info))


print("Get image path")

for image_info in annotation['images']:
    if image_info['id'] == id_info:
        image_info_  = image_info
        print(image_info_)
        continue
print("get image label")

for cate_info in annotation['categories']:
    if cate_info['id'] == cate_id_info:
        cate_info_ = cate_info
        print(cate_info_)
        continue


print(cate_info_)
print(image_info_)


x_min, y_min, x_max, y_max = int(annotation['annotations'][0]['bbox'][0]), \
                             int(annotation['annotations'][0]['bbox'][1]), \
                             int(annotation['annotations'][0]['bbox'][0]+annotation['annotations'][0]['bbox'][2]), \
                             int(annotation['annotations'][0]['bbox'][1]+annotation['annotations'][0]['bbox'][3])

polygon = annotation['annotations'][0]['segmentation']
print(polygon)
polygon = [[poly[i:i+2] for i in range(0, len(poly), 2)] for poly in polygon]
print(polygon)
print(polygon)
print(os.path.isfile(os.path.join(root_dir, image_info_['coco_url'])))

img = cv2.imread(os.path.join(root_dir, image_info_['coco_url']))
cv2.rectangle(img, (x_min,y_min),(x_max, y_max), (0, 255, 0), 3)
result = cate_info_['name']
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,result,(x_max,y_max), font, 1, (200,0,0), 3, cv2.LINE_AA)
# cv2.fillPoly(img, np.array(polygon).astype(int), (0, 0, 255))
cv2.imwrite('output.png', img)
