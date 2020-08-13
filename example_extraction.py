# -*- coding: utf-8 -*-

from resnet import FeaturesExtractor
import cv2
import numpy as np

# Extractor initialization
extractor = FeaturesExtractor()


# Infer a single Image   ------------------------------------------

# Data Loading
img = cv2.imread('quokka.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# img size has to be (W,H,C), output has size (2048,)
features = extractor.getFeatures(img)
 
# -----------------------------------------------------------------
 



# Infer Multiple Image (All images must have the same size) -------

img1 = cv2.imread('quokka.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1,(400,400))

img2 = cv2.imread('leone.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2,(400,400))

img3 = cv2.imread('gatto.jpg')
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img3,(400,400))

img_list = np.stack([img1,img2,img3])

# img_list size has to be (N,W,H,C), output has size (N,2048)
features = extractor.getFeaturesOfList(img_list)

# -----------------------------------------------------------------