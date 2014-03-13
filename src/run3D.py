import cv2 as cv
import descriptors
import detection
import numpy as np
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotStructure
import makeD

PATH_TO_DATA = "../data/"
IMAGE1 = 'dj/front.jpg'
IMAGE2 = 'dj/side.jpg'

img1 = cv.imread(PATH_TO_DATA + IMAGE1)
img2 = cv.imread(PATH_TO_DATA + IMAGE2)
kp1, des1 = descriptors.surf(IMAGE1)
kp2, des2 = descriptors.surf(IMAGE2)
matches = detection.match(img1, des1, kp1, img2, des2, kp2)
D = makeD.makeD(matches)
plotStructure.plotStructure(D)