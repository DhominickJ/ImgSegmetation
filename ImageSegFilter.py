
import cv2
import matplotlib.pyplot as plt
import numpy as np

def _plt(res_img, cmap, title, fsize, p):

    if p == 0:
        plt.title(title, fontsize = fsize)
        plt.imshow(res_img, cmap = cmap)

    if p == 1:
        plt.imshow(res_img)

    else:
        plt.imshow()

image_1 = cv2.imread('pokemon/9.png')
cvrt_img_1 = cv2.cvtcolor( image_1, cv2.COLOR_BGR2RGB)
_plt