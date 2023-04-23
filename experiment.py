import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# uploaded_image = st.file_uploader('Upload a file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
# image_raw = Image.open(uploaded_image)
# image = np.array(image_raw.convert('RGB'))
image_1 = plt.imread('pokemon/10.png')
s1 = image_1 > 150 
s2 = image_1 > 125 
fig, ax = plt.subplots(1, 3, figsize=(15,5))

imgs_list = [ image_1, s1, s2 ]

for i in range(3): 
    plt.imshow(imgs_list[i])
    # ax[i].imshow(imgs_list[i], cmap = 'gray') 

    # aa = ax[0].imshow(imgs_list[0], cmap = 'gray') 
    # if ax[i] == ax[0]: 
    #     fig.colorbar(aa, ax=ax[0]) 
        
plt.show()

# plt.imshow(image_1, cmap='gray')
# st.pyplot(plt.gcf())


