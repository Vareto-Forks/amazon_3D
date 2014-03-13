import cv2 as cv
import descriptors
import detection
import numpy as np

img1 = cv.imread('../data/vase/front.jpg')
img2 = cv.imread('../data/vase/side.jpg')
kp1, des1 = descriptors.surf('vase/front.jpg')
kp2, des2 = descriptors.surf('vase/side.jpg')
matches = detection.match(img1, des1, kp1, img2, des2, kp2)