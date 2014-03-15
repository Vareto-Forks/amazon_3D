import cv2 as cv
import descriptors
import detection
import numpy as np
#import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< HEAD
# import plotStructure
import makeD

PATH_TO_DATA = "../data/"
IMAGE1 = 'smartwatch/leftside.jpg'
IMAGE2 = 'smartwatch/rightside.jpg'
=======
import plotStructure
import makeD

PATH_TO_DATA = "../data/"
IMAGE1 = 'smartwatch/rightside.jpg'
IMAGE2 = 'smartwatch/leftside.jpg'
>>>>>>> 8f568aec5dea9d9c2faba071cbbf780fed06b4f4

img1 = cv.imread(PATH_TO_DATA + IMAGE1)
img2 = cv.imread(PATH_TO_DATA + IMAGE2)
kp1, des1 = descriptors.sift(IMAGE1)
kp2, des2 = descriptors.sift(IMAGE2)
matches = detection.match(img1, des1, kp1, img2, des2, kp2)
<<<<<<< HEAD
D = makeD.makeD(matches, kp1, kp2)
# plotStructure.plotStructure(D)
=======
D = makeD.makeD(matches,kp1,kp2)
plotStructure.plotStructure(D)
>>>>>>> 8f568aec5dea9d9c2faba071cbbf780fed06b4f4
