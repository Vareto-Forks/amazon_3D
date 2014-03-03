import cv2 as cv 
import numpy as np 

KP_THRESH = 0.7

def match(img1, des1, kp1, img2, des2, kp2):
	matches = matchKeypoints(des1, des2, KP_THRESH)


def matchKeypoints(des1, des2, thresh):
	matches = []
	for i in range(len(des1)):
		best_val = -float('Inf')
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
                	second_val = last_best_val;
            elif d < second_val:
            	second_val = d
        d_ratio = float(best_val)/second_val
        if d_ratio < thresh:
        	matches.append((i, best_ind))
    return matches