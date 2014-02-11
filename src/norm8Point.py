import numpy as np
import math

PATH_TO_DATA = '../data/'

def normalize(points):
	x1_avg = float(sum(x[0] for x in points))/(len(points))
	y1_avg = float(sum(x[1] for x in points))/(len(points))
	x2_avg = float(sum(x[2] for x in points))/(len(points))
	y2_avg = float(sum(x[3] for x in points))/(len(points))
	RMS1 = 0.0
	RMS2 = 0.0
	for x in points:
		RMS1 += math.sqrt(math.pow(x[0]-x1_avg, 2) + math.pow(x[1]-y1_avg, 2))
		RMS2 += math.sqrt(math.pow(x[2]-x2_avg, 2) + math.pow(x[3]-y2_avg, 2))
	RMS1 = float(RMS1)/len(points)
	RMS2 = float(RMS2)/len(points)
	s1 = math.sqrt(2)/RMS1
	s2 = math.sqrt(2)/RMS2
	T1 = np.matrix([[s1, 0, s1*x1_avg],[0, s1, s1*y1_avg],[0, 0, 1]])
	T2 = np.matrix([[s2, 0, s2*x2_avg],[0, s2, s2*y2_avg],[0, 0, 1]])
	points1 = []
	for x in points:
		points1.append([x[0], x[1], 1])
	points2 = []
	for x in points:
		points2.append([x[2], x[3], 1])
	points1 = np.matrix(points1)
	points2 = np.matrix(points2)
	newPoints1 = points1*T1
	newPoints2 = points2*T2
	return newPoints1, newPoints2

def constructW(A, B):
	W = []
	for i in range(len(A)):
		a = A[i]
		b = B[i]
		row = [a.item(0)*b.item(0), a.item(1)*b.item(0), b.item(0), a.item(0)*b.item(1), a.item(1)*b.item(1), b.item(1), a.item(0), a.item(1), 1]
		W.append(row)
	W = np.matrix(W)
	return W 

cFilename = raw_input('Give the name of the correspondece file from /data/ :')
cFile = open(PATH_TO_DATA + cFilename, 'r')
points = []
for line in cFile:
	x1, y1, x2, y2 = line.split()
	points.append([float(x1), float(y1), float(x2), float(y2)])

points1, points2 = normalize(points)
W = constructW(points1, points2)
U, D, V = np.linalg.svd(W)
f = V[:, 8]
f = f.reshape(3,3)
U, D, V = np.linalg.svd(f)
D.itemset(2, 0.0)
F = U*np.diag(D)*V
print F