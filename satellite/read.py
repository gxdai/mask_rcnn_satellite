from PIL import Image
import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# img = Image.open("/data1/Guoxian_Dai/satellite_image/20jpng.png")
# img = cv2.imread("/data1/Guoxian_Dai/satellite_image/20jpng.png")
img = plt.imread("/data1/Guoxian_Dai/satellite_image/20jpng.png")
# img = np.random.random((3865, 6656, 4))
width_step = 1024
height_step = 1024

height_number = img.shape[0] // height_step
width_number = img.shape[1] // width_step
print(img.shape)
print(height_number, width_number)
sys.exit()
output_dir = "sliced_img"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for i in range(height_number):
    for j in range(width_number):
        img_i_j = img[i*height_step:(i+1)*height_step, j*width_step:(j+1)*width_step, :]
        print(img_i_j.shape)
        outfile = os.path.join(output_dir, str(i*height_step)\
                                    +'_' + str((i+1)*height_step)\
                                    +'_' +str(j*width_step)\
                                    +'_'+str((j+1)*width_step)\
                                    +'.png')
        plt.imsave(outfile, img_i_j)
        
