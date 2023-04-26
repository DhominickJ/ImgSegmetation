import cv2
import matplotlib.pyplot as plt
import numpy as np

image_1 = cv2.imread('pokemon/9.png', 0)
cvrt_img_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(cvrt_img_1, cv2.COLOR_RGB2HSV)
#define blue color range
light_blue = np.array([110,50,50])
dark_blue = np.array([130,255,255])


def plt( res_img, cmap, title, fsize, p):

    if p == 0:
        plt.title(title, fontsize=fsize)
        plt.imshow(res_img, cmap=cmap)

    if p == 1:
        plt.imshow(res_img)

    else:
        plt.imshow(res_img, cmap=cmap)

    _plt(cvrt_img_1, '', '', '', 1)
    
def filterColorRed(image_1):
    #Threshold the HSV image to get only blue colors 
    mask = cv2.inRange (hsv, light_blue, dark_blue)
    #Bitwise-AND mask and original image
    output = cv2.bitwise_and (cvrt_img_1, cvrt_img_1, mask= mask)
    plt.imshow(np.hstack ((cvrt_img_1, output)))
    filter_color_red =(cvrt_img_1[:,:,0] > 150)
    _plt filter_color_red, 'gray', '', '', 2
    
    mask_img = cvrt_img_1.copy()
    mask_img[:,:, 0] = mask_img[:, :, 0]*filter_color_red 
    mask_img[:, :, 1] = mask_img[:, :, 1]*filter_color_red 
    mask_img[:, :, 2] = mask_img[:, :, 2]*filter_color_red 
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    _plt (mask_img,'', '', '', 1)
    
    filter_color_red = (cvrt_img_1[:,:,0] > 150) & (cvrt_img_1[:,:,1] < 100) & (cvrt_img_1[:,:,2] < 110) mask_img = cvrt_img_1.copy()
    mask_img[:, :, 0]= mask_img[:, :, 0] * filter_color_red mask_img[:, :, 1] = mask_img[:, :, 1] * filter_color_red
    mask_img[:, :, 2]= mask_img[:, :, 2] * filter_color_red
    plt.figure(num=None, figsize=(8, 6), dpi=80) plt.imshow(mask_img);
    
def filterHSV(image_1):
    img_hsv = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
    hsv_list = ['Hue', 'Saturation', 'Value']
    fig, ax = plt.subplots(1, 3, figsize=(15,7), sharey = True)
    ax[0].imshow(img_hsv[:,:,0], cmap = 'hsv')
    ax[0].set_title (hsv_list[0], fontsize = 20) 
    ax[0].axis('off')

    ax[1].imshow(img_hsv[:,:,1], cmap = 'Greys') 
    ax[1].set_title (hsv_list[1], fontsize 20) 
    ax[1].axis('off')

    ax[2].imshow(img_hsv[:,:,2], cmap = 'gray') 
    ax[2].set_title (hsv_list[2], fontsize = 20) 
    ax[2].axis('off')
    plt.show()
    
def showHSV(image_1):
    plt.figure(num=None, figsize=(8, 6), dpi=80) 
    plt.imshow(img_hsv[:,:,0], cmap='hsv')
    plt.colorbar();
    
    lower_mask = upper_mask
    img_hsv[,,0] > 150
    =
    img_hsv[,,0] < 220
    mask = upper_mask*lower_mask
    red =
    cvrt_img_1[:,:,0]*mask
    green = cvrt_img_1[:,:,1]*mask
    blue = cvrt_img_1[:,:,2]*mask
    img_masked = np.dstack((red, green, blue))
    plt.figure(num=None, figsize=(8, 6), dpi=80) 
    plt.imshow(img_masked);
    
def masking(image_1):
    lower_mask = img_hsv[:,:,0] > 150
    upper_mask = img_hsv[:,:, 0] < 220
    saturation = img_hsv[:,:,1] > 100
    
    mask = upper_mask*lower_mask*saturation
    red = cvrt_img_1[:,:,0]*mask
    green = cvrt_img_1[:,:,1]*mask
    blue = cvrt_img_1[:,:,2]*mask
    img_masked = np.dstack ((red, green, blue))
    plt.figure(num=None, figsize=(8, 6), dpi=80) 
    plt.imshow(img_masked);

#lower_mask = img_hsv[:,:,0] > 150 upper_mask = img_hsv[:,:, 0] < 220 saturation = img_hsv[:,:,1] > 100
# mask upper_mask*lower_mask* saturation
# red
# cvrt_img_1[:,:,0]*mask
# green
# cvrt_img_1[:,:,1]*mask
# blue = cvrt_img_1[:,:,2]*mask
# img_masked = np.dstack ((red, green, blue))
# plt.figure(num=None, figsize=(8, 6), dpi=80) plt.imshow(img_masked);

def showMask(image_1):
    lower_mask_1 = img_hsv[:,:,0] > 95 
    upper_mask_1 = img_hsv[:,,0] < 155 
    saturation_1 = img_hsv[:,:,1] > 90
    mask_1= lower_mask_1*upper_mask_1*saturation_1
    sky_filtered = np.dstack (( cvrt_img_1[:,:,0]*mask_1,
                                cvrt_img_1[:,:,1]*mask_1,
                                cvrt_img_1[:,:,2]*mask_1))
                                
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(sky_filtered);
