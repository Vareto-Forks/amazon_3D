import cv2 as cv
import re
import numpy as np

PATH_TO_DATA = '../data/'

def createDescriptors(filename):
	img = cv.imread(PATH_TO_DATA+filename)
	sift = cv.SIFT()
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray, None)
	folder = re.findall('(.*)/.*\.jpg', filename)[0]
	imgName = re.findall('.*/(.*)\.jpg', filename)[0]
	toWriteName = imgName + '.descriptors'
	fileToWrite = open(PATH_TO_DATA + folder + "/" + toWriteName, 'w')
	for index, d in enumerate(des):
		kpi = kp[index]
		kpList = [str(kpi.pt[0]), str(kpi.pt[1]), str(kpi.angle), str(kpi.size)]
		toWrite = ", ".join(kpList) + ':'
		for dimension in d:
			toWrite += str(dimension) + ','
		toWrite += '\n'	
		fileToWrite.write(toWrite)
	return kp, des