import json
 
with open('train.json') as f:
     data = json.load(f)
 
 
print(data.keys())
print(len(data['annotations']))
print(len(data['images']))
print(len(data['categories']))
print(data['annotations'][0])
for ann in data['annotations']:
     # if ann['image_id'] == 558840:
     if ann['image_id'] == 156:
         print("ann['image_id'] = {}".format(ann['image_id']))
         print("ann['category_id'] = {}".format(ann['category_id']))
         print("ann['id'] = {}".format(ann['id']))

