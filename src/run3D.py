import cv2 as cv
import descriptors
import detection
import numpy as np
#import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotStructure
import makeD

PATH_TO_DATA = '../data/'

IMAGE1 = 'vball/leftside.JPG'
IMAGE2 = 'vball/rightside.JPG'

img1 = cv.imread(PATH_TO_DATA + IMAGE1)
img2 = cv.imread(PATH_TO_DATA + IMAGE2)
kp1, des1 = descriptors.surf(IMAGE1)
kp2, des2 = descriptors.surf(IMAGE2)
matches = detection.match(img1, des1, kp1, img2, des2, kp2)
# plotStructure.plotStructure(D)
D = makeD.makeD(matches,kp1,kp2)
plotStructure.plotStructure(D)
