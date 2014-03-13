import cv2 as cv
import descriptors
import detection
import numpy as np
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img1 = cv.imread('../data/test/image1.jpg')
img2 = cv.imread('../data/test/image2.jpg')
kp1, des1 = descriptors.surf('test/image1.jpg')
kp2, des2 = descriptors.surf('test/image2.jpg')
matches = detection.match(img1, des1, kp1, img2, des2, kp2)
D = []
x1s = []
y1s = []
x2s = []
y2s = []
for index in range(len(matches)):
	point1 = kp1[matches[index][0]].pt
	point2 = kp2[matches[index][1]].pt
	x1s.append(point1[0])
	y1s.append(point1[1])
	x2s.append(point2[0])
	y2s.append(point2[1])
x1Avg = sum(x1s)/float(len(x1s))
y1Avg = sum(y1s)/float(len(y1s))
x2Avg = sum(x2s)/float(len(x2s))
y2Avg = sum(y2s)/float(len(y2s))
for index in xrange(len(x1s)):
	x1s[index] -= x1Avg
	y1s[index] -= y1Avg
	x2s[index] -= x2Avg
	y2s[index] -= y2Avg
D.append(x1s)
D.append(y1s)
D.append(x2s)
D.append(y2s)
D = np.matrix(D)

U, W, V = np.linalg.svd(D)
V = np.transpose(V)
U3 = U[:,0:3]
W3 = W[0:3]
V3 = V[:,0:3]
W3 = np.diag(W3)
Structure = scipy.linalg.sqrtm(W3)* np.transpose(V3)
plotStructure = np.transpose(Structure)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = []
ys = []
zs = []
for index in xrange(len(x1s)):
	xs.append(plotStructure[index,0])
	ys.append(plotStructure[index,1])
	zs.append(plotStructure[index,2])
ax.scatter(xs,ys,zs, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()