import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

mask = plt.imread('./mask_annotation/area/a2_area_rejion.png')
mask_label = np.unique(mask)
print(mask_label.shape)
print(mask_label)

with open('./mask_annotation/area/a2_area.json') as f:
    data = json.load(f)

print(data['imgHeight'])
print(data['imgWidth'])

print(type(data['objects']))
print(len(data['objects']))

# Get label set
label_set = set(['background', 'Tree',\
                  'Parking_Lot', 'Road', \
                  'Vehicles', 'Well', \
                  'Buildings'
                 ]
                )


# Here is the label set
# {'Tree', 'Road', 'Parking_Lot', 'background', 'Vehicles', 'Well', 'Buildings'}

# The default channel order is B, G, R
color_dict = {'background': (255, 255, 255), 'Tree': (128, 255, 0),\
              'Parking_Lot': (0, 0, 204), 'Road': (204, 0, 204), \
              'Vehicles': (255, 0, 0), 'Well': (255, 255, 0), \
              'Buildings': (0, 255, 255)
             }


# black_image = np.zeros((data['imgHeight'], data['imgWidth'], 3), np.uint8)
counter1 = 0 # for park and vichiel
counter2 = 0
for label in label_set:
    black_image = np.zeros((data['imgHeight'], data['imgWidth'], 3), np.uint8)
    for obj in data['objects']:
        if obj['label'] != label:
           continue
        if label == 'Parking_Lot':
            counter1 += 1
        elif label == 'Vehicles':
            counter2 += 1
        cv2.fillPoly(black_image, np.array([obj['polygon']]).astype(int), color_dict[obj['label']])
    # label_set.add(obj['label'])
    cv2.imwrite(label+'.png', black_image)
black_image = np.zeros((data['imgHeight'], data['imgWidth'], 3), np.uint8)
print(counter1, counter2)
counter1, counter2 = 0, 0
for obj in data['objects']:
    if obj['label'] == 'Parking_Lot':
        counter1 += 1
    elif obj['label'] == 'Vehicles':
        counter2 += 1
    cv2.fillPoly(black_image, np.array([obj['polygon']]).astype(int), color_dict[obj['label']])
    # label_set.add(obj['label'])
cv2.imwrite('new_mask2.png', black_image)

print(counter1, counter2)
"""
cv2.imshow('mask', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("new_mask.png", black_image)
"""
# print(label_set)
