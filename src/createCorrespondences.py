import cv2

def getxy(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print x, y

img = cv.imread('../data/shoe/shoe-1.jpg')
cv.namedWindow('image')
cv.setCallback('image', getxy)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()