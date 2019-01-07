import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2


GT = 'result/GT_dir'
Prediction = 'result/Prediction'
Compare = 'result/Compare'

if not os.path.isdir(Compare):
    os.makedirs(Compare)

file_list = os.listdir(GT)
"""
for fl in file_list:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    img = plt.imread(os.path.join(GT, fl))
    ax1.imshow(img)
    ax1.axis('off')

    ax2 = fig.add_subplot(1,2,2)
    img = plt.imread(os.path.join(Prediction, fl))
    ax2.imshow(img)
    ax2.axis('off')

    # Save the full figure...
    fig.savefig(os.path.join(Compare, fl))
"""

for fl in file_list:
    img_gt = cv2.imread(os.path.join(GT, fl))
    print(img_gt.shape)
    shape = img_gt.shape
    con_img = np.zeros((shape[0], 2*shape[1] + 20, shape[2]))
    img_pred = cv2.imread(os.path.join(Prediction, fl))
    img_pred = cv2.resize(img_pred, (shape[0], shape[1]))
    print(img_pred.shape)
    con_img[:, :shape[1], :] = img_gt
    con_img[:, shape[1]+20:, :] = img_pred
    # Save the full figure...
    cv2.imwrite(os.path.join(Compare, fl), con_img)


"""


plt.figure(figsize = (1,2))
gs1 = gridspec.GridSpec(1, 2)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
ax1 = plt.subplot(gs1[0]) 
ax2 = plt.subplot(gs1[1])
for fl in file_list:
    img = plt.imread(os.path.join(GT, fl))
    ax1.imshow(img)
    ax1.axis('off')

    img = plt.imread(os.path.join(Prediction, fl))
    ax2.imshow(img)
    ax2.axis('off')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    # Save the full figure...
    plt.savefig(os.path.join(Compare, fl))


"""
"""
for i in range(2):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    plt.subp

plt.show()
"""
