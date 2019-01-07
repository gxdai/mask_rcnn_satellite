import json
 
with open('/home/ubuntu/data/complete_set/annotations/train.json') as f:
     data = json.load(f)
 
 
print(data.keys())
print(len(data['annotations']))
print(len(data['images']))
print(len(data['categories']))
print(data['annotations'][0])
image_ids = []
fid = open('airport.txt', 'w') 
for ann in data['annotations']:
     # if ann['image_id'] == 558840:
     if ann['category_id'] == 18:
         print("ann['image_id'] = {}".format(ann['image_id']))
         print("ann['category_id'] = {}".format(ann['category_id']))
         print("ann['id'] = {}".format(ann['id']))
         fid.write("image_id: {}\n".format(ann['image_id']))

fid.close()
         

print(data['categories'][17]['name'])
print(data['categories'][17]['id'])

