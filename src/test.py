import cv2 as cv
import descriptors
import detection
import numpy as np

img1 = cv.imread('../data/test/image1.jpg')
img2 = cv.imread('../data/test/image2.jpg')
kp1, des1 = descriptors.createDescriptors('test/image1.jpg')
kp2, des2 = descriptors.createDescriptors('test/image2.jpg')
matches = detection.match(img1, des1, kp1, img2, des2, kp2)