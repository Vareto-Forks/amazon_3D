import cv2 as cv 
import numpy as np 
import ransac

KP_THRESH = 0.7

def match(img1, des1, kp1, img2, des2, kp2):
	matches = matchKeypoints(des1, des2, KP_THRESH)
	matches, model = ransac.ransac(matches, kp1, kp2)
	height1, width1, depth1 = img1.shape
	height2, width2, depth2 = img2.shape
	height = height1
	width = width1 + width2
	img3 = np.zeros((height,width,3), np.uint8)
	img3[0:height, 0:width1] = img1.copy()
	img3[0:height, width1:width] = img2.copy()
	for match in matches:
		ind1, ind2 = match
		pt1 = kp1[ind1]
		pt2 = kp2[ind2]
		pt1 = pt1.pt
		pt2 = pt2.pt
		pt1 = (int(pt1[0]), int(pt1[1]))
		pt2 = (int(pt2[0]) + width1, int(pt2[1]))
		cv.line(img3, pt1, pt2, 255)
	cv.imshow('img', img3)
	print "... displaying matches ... "
	cv.waitKey(0);
	return matches

def matchKeypoints(des1, des2, thresh):
	matches = []
	for i in range(len(des1)):
		best_val = float('Inf')
		best_ind = 0
		second_val = float('Inf')
		v1 = des1[i]
		for j in range(len(des2)):
			v2 = des2[j]
			d = np.linalg.norm(v1-v2)
			if d < best_val:
				last_best_val = best_val
				best_val = d
				best_ind = j
				if last_best_val < second_val:
					second_val = last_best_val
			elif d < second_val:
				second_val = d
		d_ratio = float(best_val)/second_val
		if d_ratio < thresh:
			matches.append((i, best_ind))
	return matches