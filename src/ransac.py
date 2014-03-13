import random
import numpy as np 

NUM_ITER = 1000
NUM_SAMPLES = 4
PIXEL_THRESH = 50

def ransac(matches, kp1, kp2):
	pts1 = []
	pts2 = []
	for match in matches:
		ind1, ind2 = match.trainIdx, match.queryIdx
		pt1 = kp1[ind1].pt
		pt2 = kp2[ind2].pt
		pts1.append((pt1[0], pt1[1], 1.))
		pts2.append((pt2[0], pt2[1], 1.))
	inliers = []
	numInliers = 0	
	model = None
	for i in range(NUM_ITER):
		sampleInd = random.sample(range(len(matches)), NUM_SAMPLES)
		samples1 = np.zeros((3, NUM_SAMPLES))
		samples2 = np.zeros((3, NUM_SAMPLES))
		for j in range(NUM_SAMPLES):
			samples1[:,j] = pts1[sampleInd[j]]
			samples2[:,j] = pts2[sampleInd[j]]
		H = np.matrix(samples1)*np.matrix(samples2).I	
		inlierTemp = []
		for j in range(len(pts1)):
			x = np.matrix(pts1[j]).T
			xPrime = H*x
			x2 = np.matrix(pts2[j]).T
			# print xPrime
			# print x2
			diff = np.linalg.norm(x2-xPrime)
			if diff < PIXEL_THRESH:
				inlierTemp.append(matches[j])
		if len(inlierTemp) > numInliers:
			numInliers = len(inlierTemp)
			inliers = inlierTemp
			model = H
	return inliers, model
