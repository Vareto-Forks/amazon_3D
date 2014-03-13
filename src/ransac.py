import random
import numpy as np 
import scipy.linalg
import cv2 as cv 

NUM_ITER = 1000
NUM_SAMPLES = 4
PIXEL_THRESH = 20

def ransac(matches, kp1, kp2):
	pts1 = []
	pts2 = []
	for match in matches:
		ind1, ind2 = match
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
		if i == 0:
			print np.matrix(samples2)
			print np.matrix(samples2).I
		H = np.matrix(samples2)*np.matrix(samples1).I	
		inlierTemp = []
		for j in range(len(pts1)):
			x = np.matrix(pts1[j]).T
			xPrime = H*x
			x2 = np.matrix(pts2[j]).T
			xPrime[0] = xPrime[0][0]/xPrime[2][0]
			xPrime[1] = xPrime[1][0]/xPrime[2][0]
			xPrime = xPrime[0:2]
			x2[0] = x2[0][0]/x2[2][0]
			x2[1] = x2[1][0]/x2[2][0]
			x2 = x2[0:2]
			diff = np.linalg.norm(x2-xPrime)
			if diff < PIXEL_THRESH:
				inlierTemp.append(matches[j])
		if len(inlierTemp) > numInliers:
			numInliers = len(inlierTemp)
			inliers = inlierTemp
			model = H
	return inliers, model

def ransac2(matches, kp1, kp2):
	pts1 = []
	pts2 = []
	for match in matches:
		ind1, ind2 = match
		pt1 = kp1[ind1].pt
		pt2 = kp2[ind2].pt
		pts1.append((pt1[0], pt1[1], 1.))
		pts2.append((pt2[0], pt2[1], 1.))
	inliers = []
	numInliers = 0	
	model = None
	X = np.array([(a, b, c) for (a,b,c) in pts1])
	Xp = np.array([(a, b, c) for (a,b,c) in pts2])
	for iteration in range(NUM_ITER):
		sampleInd = random.sample(range(len(matches)), NUM_SAMPLES)
		samples1 = np.zeros((3, NUM_SAMPLES))
		samples2 = np.zeros((3, NUM_SAMPLES))
		for j in range(NUM_SAMPLES):
			samples1[:,j] = X[sampleInd[j]]
			samples2[:,j] = Xp[sampleInd[j]]
		A = np.zeros((2*NUM_SAMPLES,9))
		for i in range(NUM_SAMPLES):
			Ai = np.zeros((2,9))
			xi = samples1[:,i]
			xp = samples2[:,i]
			Ai[0,3:6] = -xi.transpose()
			Ai[0,6:9] = xp[1]*xi.transpose()
			Ai[1,0:3] = xi.transpose()
			Ai[1,6:9] = -xp[0]*xi.transpose()

			A[(2*i):(2*i + 2),:]=Ai
		U, S, V = np.linalg.svd(A)
		h = V[:,-1]
		H = np.zeros((3,3))
		H[0,0] = h[0]
		H[0,1] = h[1]
		H[0,2] = h[2]
		H[1,0] = h[3]
		H[1,1] = h[4]
		H[1,2] = h[5]
		H[2,0] = h[6]
		H[2,1] = h[7]
		H[2,2] = h[8]


		test = np.dot(H,X.transpose())

		test[0,:] = test[0,:]/test[2,:]
		test[1,:] = test[1,:]/test[2,:]
		test[2,:] = test[2,:]/test[2,:]

		inlierTemp = []
		for j in range(len(pts1)):
			error = np.linalg.norm(test[:,j]-Xp[j,:].transpose())
			if error < PIXEL_THRESH:
				inlierTemp.append(matches[j])
		if len(inlierTemp) > numInliers:
			numInliers = len(inlierTemp)
			inliers = inlierTemp
			model = H
	return inliers, model

def ransac3(matches, kp1, kp2):
	pts1 = []
	pts2 = []
	for match in matches:
		ind1, ind2 = match
		pt1 = kp1[ind1].pt
		pt2 = kp2[ind2].pt
		pts1.append((pt1[0], pt1[1], 1.))
		pts2.append((pt2[0], pt2[1], 1.))
	X = np.array([(a, b, c) for (a,b,c) in pts1])
	Xp = np.array([(a, b, c) for (a,b,c) in pts2])
	src_pts = np.float32([kp1[m[0]].pt for m in matches]).reshape(-1,1,2)
	dst_pts = np.float32([kp2[m[1]].pt for m in matches]).reshape(-1,1,2)
	H,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)

	test = np.dot(H,X.transpose())

	test[0,:] = test[0,:]/test[2,:]
	test[1,:] = test[1,:]/test[2,:]
	test[2,:] = test[2,:]/test[2,:]

	inliers = []
	for j in range(len(pts1)):
		error = np.linalg.norm(test[:,j]-Xp[j,:].transpose())
		if error < PIXEL_THRESH:
			inliers.append(matches[j])
	return inliers, H

