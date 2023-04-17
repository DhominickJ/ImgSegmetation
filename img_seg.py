
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
def _plt( img, cmap, title, fsize, p): 
    if p == 0: 
        plt.title(title, fontsize=fsize) 
        plt.imshow(img, cmap=cmap) 
    else: 
        plt.imshow(img, cmap=cmap) 
    
image_1 = cv2.imread('pokemon/9.png', 0) 
        
#Method I 
s1 = image_1 > 150 
s2 = image_1 > 125 

fig, ax = plt.subplots(1,3,figsize=(15,5)) 
imgs_list = [ image_1, s1, s2 ]
for i in range( len( imgs_list)): 
    ax[i].imshow(imgs_list[i], cmap = 'gray') 
    aa = ax[0].imshow(imgs_list[0], cmap = 'gray') 
    if ax[i] == ax[0]: 
        fig.colorbar(aa, ax=ax[0]) 

#Method II
thresh, binaryImg = cv2.threshold(image_1, 0, 255, cv2.THRESH_OTSU)
# binaryImg = cv2.bitwise_not(binaryImg)
print("Thresh:", thresh)
_plt(binaryImg, 'gray', '', '', 1)

#Method III
# from skimage.filters import threshold_otsu

# thresh = threshold_otsu(image_1)
# s3 = image_1 > thresh
# _plt(s3, 'gray', '', '', 1)

#Watershed Algorithm
image_2 = cv2.imread("pokemon/10.png")
gray = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY )
ret, thresh = cv2.threshold(gray , 0 ,255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = 1)
imgs_list = [thresh, closing]
fig, ax = plt.subplots(1, 2, figsize=(15,7), sharey = True)
for i in range(len( imgs_list)):
    ax[i].imshow(imgs_list[i], cmap = 'gray')
    ax[i].axis('off')
plt.show() 

#Seperation
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel, iterations = 1)
dist1 = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
ret, dist2 = cv2.threshold(dist1, 0.6*dist1.max(), 255, 0)
imgs_list = [ closing, dist1, dist2 ]
fig, ax = plt.subplots(1, 3, figsize=(15,7), sharey = True)
for i in range(len(imgs_list)):
    ax[i].imshow(imgs_list[i], cmap = 'gray')
    ax[i].axis('off')
plt.show()

#Marker Labelling
markers1 = np.zeros(dist1.shape, dtype=np.int32)
dist_8u = dist2.astype('uint8')

contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cv2.drawContours(markers1, contours, i, (i+1), -1)

markers2 = cv2.circle(markers1, (15,15), 5, len(contours)+1, -1)
    
markers3 = cv2.watershed(image_2, markers2)
image_2[markers3 == -1] = [0,0,255]

imgs_list = [ markers1, markers2, markers3, image_2 ]
fig, ax = plt.subplots(1, 4, figsize=(15,7), sharey = True)
for i in range(len( imgs_list)):
    ax[i].imshow(imgs_list[i])
    ax[i].axis('off')
plt.show()