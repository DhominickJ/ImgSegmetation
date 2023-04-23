
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import glob as gl
import PIL.Image as Image
import streamlit as st
from skimage.filters import threshold_otsu
from skimage.io import imshow, imread 
from skimage.color import rgb2hsv, hsv2rgb

def _plt( img, cmap, title, fsize, p): 
    if p == 0: 
        plt.title(title, fontsize=fsize) 
        plt.imshow(img, cmap=cmap) 
    else: 
        plt.imshow(img, cmap=cmap) 
    
# def Img_Choice():
#     filename = gl.glob('pokemon/*.png')
#     for i in range(len(filename)):
#         print(str(i) + ":" + filename[i])

#     choice = int(input("Enter the choice image of choiceg: "))
#     image_1 = cv2.imread(filename[choice], 0)
#     s1 = image_1 > 150 
#     s2 = image_1 > 125 
#     return image_1
#     # image_1 = cv2.imread('pokemon/9.png', 0)       
# #Method I 

# def ForLoopIteration(image):
#     # fig = plt.figure(figsize=(15,5))
#     s1 = image > 150 
#     s2 = image > 125 
#     fig, ax = plt.subplots(1,3,figsize=(15,5)) 
#     imgs_list = [ image, s1, s2 ]
#     for i in range( len( imgs_list)): 
#         ax[i].imshow(imgs_list[i], cmap = 'gray') 
#         aa = ax[0].imshow(imgs_list[0], cmap = 'gray') 
#         if ax[i] == ax[0]: 
#             fig.colorbar(aa, ax=ax[0]) 
#     # plt.show()
#     # plt.imshow(image, cmap='gray') 
#     st.pyplot()

#Method II
def ThreshtoBinaryImg(image_1):
    thresh, binaryImg = cv2.threshold(image_1, 0, 255, cv2.THRESH_OTSU)
    # binaryImg = cv2.bitwise_not(binaryImg)
    print("Thresh:", thresh)
    _plt(binaryImg, 'gray', '', '', 1)

# Method III
def WithOtsu(image_1):
    thresh = threshold_otsu(image_1)
    s3 = image_1 > thresh
    _plt(s3, 'gray', '', '', 1)

#Watershed Algorithm
def WatershedSegmentation(image):
    # image_2 = cv2.imread("pokemon/10.png")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY )
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
def Seperation(thresh):
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
def MarkerLabelling(image, dist1, dist2):
    markers1 = np.zeros(dist1.shape, dtype=np.int32)
    dist_8u = dist2.astype('uint8')

    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(markers1, contours, i, (i+1), -1)

    markers2 = cv2.circle(markers1, (15,15), 5, len(contours)+1, -1)
        
    markers3 = cv2.watershed(image, markers2)
    image[markers3 == -1] = [0,0,255]

    imgs_list = [ markers1, markers2, markers3, image]
    fig, ax = plt.subplots(1, 4, figsize=(15,7), sharey = True)
    for i in range(len( imgs_list)):
        ax[i].imshow(imgs_list[i])
        ax[i].axis('off')
    plt.show()

def main():
    # image_1 = Img_Choice()
    with st.sidebar:
        st.slider('Select a value', 1, 10, 1)
        uploaded_image = st.file_uploader('Upload a file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
        if uploaded_image is not None:
            # st.write('You selected `%s`' % uploaded_image.name)
            image_raw = Image.open(uploaded_image).convert('RGB')
            image = np.array(image_raw)
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            st.error('Please upload an image')

        function = st.selectbox('Select a Function: ', ('Image Segmentation', 'Threshold to Binary Image', 'With Otsu', 'Watershed Segmentation', 'Seperation', 'Marker Labelling'))
        if function == 'Image Segmentation':
            st.title("Currently Debugging!")
            # image = cv2.imread('pokemon/10.png')
            # Img_Segmentation(image)
        elif function == 'Threshold to Binary Image':
            ThreshtoBinaryImg(image)
        elif function == 'With Otsu':
            WithOtsu(image)
        elif function == 'Watershed Segmentation':
            WatershedSegmentation(image)
        elif function == 'Seperation':
            Seperation(thresh)
        elif function == 'Marker Labelling':
            MarkerLabelling(image, dist1, dist2)
        
if __name__ == '__main__':
    main()
