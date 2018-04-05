#Hough Transform
import numpy as np
import matplotlib.pyplot as plt
import cv2


def HoughTransformLines(inArray):
	"""
	Adapted from Alyssa Quek's GitHub page @ https://alyssaq.github.io/2014/understanding-hough-transform/
	Perform Linear Hough Transform for lines on image in inArray
	Parameters:
		inArray: 2D binary numpy array of edges (ie 1 if pixel is classified as part of an edge)
	Output:
		accumulator: 2D numpy array indicating how many pixels could be classified by a given (rho,theta) in linear Hough Space
		rho: list of all 'rho' values used to generate HoughTransform. Rho is min distance of a line from the origin.
		thetas: list of angles (theta) used 	
		
	"""
	thetas = np.deg2rad(np.arange(-90.0, 90.0))
	width, height = inArray.shape
	diag_len = np.ceil(np.sqrt(width * width + height * height))
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
	
	cos_thetas = np.cos(thetas)
	sin_thetas = np.sin(thetas)
	num_thetas = len(thetas)
	
	accumulator = np.zeros((int(2 * diag_len), num_thetas), dtype=np.uint64)
	row_idxs, col_idxs = np.nonzero(inArray)
	
	for i in range(len(col_idxs)):
		col = col_idxs[i]
		row = row_idxs[i]
		
		for t_idx in range(num_thetas):
			rho = round(col * cos_thetas[t_idx] + row * sin_thetas[t_idx])
			accumulator[int(rho + diag_len), int(t_idx)] += 1
			
	return accumulator, thetas, rhos
	

def CreateTestImage():
	IMG_SIZE = 50
	testImage = np.zeros((IMG_SIZE, IMG_SIZE + 10))
	
	lineStart1 = (30,15)
	lineEnd1 = (40,17)
	
	lineStart2 = (12,6)
	lineEnd2 = (28,36)
	
	lineStart3 = (30,35)
	lineEnd3 = (23,6)
	
	lineStart4 = (10,2)
	lineEnd4 = (8,23)
	
	OverlayLineOnImage(testImage, lineStart1, lineEnd1)
	OverlayLineOnImage(testImage, lineStart2, lineEnd2)
	OverlayLineOnImage(testImage, lineStart3, lineEnd3)
	OverlayLineOnImage(testImage, lineStart4, lineEnd4)
		
	return np.uint8(testImage)
	
def OverlayLineOnImage(inImage, lineStart, lineEnd):
	lineSlope = float(lineEnd[0] - lineStart[0])/ float(lineEnd[1] - lineStart[1])
	
	lineUnitVector = [ lineSlope / np.sqrt(lineSlope * lineSlope + 1), 1.0 / np.sqrt(lineSlope * lineSlope + 1)]
	if (np.sign(lineEnd[0] - lineStart[0]) != np.sign(lineUnitVector[0])):
		lineUnitVector[0] *= -1
		lineUnitVector[1] *= -1
	
	lineLength = np.sqrt(pow((lineEnd[0] - lineStart[0]),2) + pow((lineEnd[1] - lineStart[1]),2))
	
	#Generate points along line1
	distOnLine = 0
	while distOnLine < lineLength:
		currRow = np.round(lineStart[0] + distOnLine * lineUnitVector[0])
		currCol = np.round(lineStart[1] + distOnLine * lineUnitVector[1])
		inImage[int(currRow),int(currCol)] = 254
		distOnLine += 1
	return inImage
	
def ReturnMaximaImage(inImage, checkRadius):
	maxImage = np.zeros(inImage.shape)
	height, width = inImage.shape
	for col in range(2,width-1):
		for row in range(2,height-1):
			thisVal = inImage[row,col]
			leftCol = max([0,col-checkRadius])
			rightCol = min([col+checkRadius,width])
			topRow = max([0,row-checkRadius])
			btmRow = min([row+checkRadius,height])
			
			checkRegion = inImage[topRow:(btmRow+1),leftCol:(rightCol+1)]
			if thisVal >= checkRegion.max() and thisVal > 15 and (checkRegion == thisVal).sum() < 4:
				maxImage[row,col] = 1
	return maxImage
			
	


myImage = cv2.imread('cameraman.tif')
#myImage = CreateTestImage()
plt.figure(1)
plt.imshow(myImage)

edgeImage = cv2.Canny(myImage,100,200)
plt.figure(2)
plt.imshow(edgeImage)
plt.title('Canny Edges')

edgeArray = np.array(edgeImage)
outAccum, outThetas, outRhos = HoughTransformLines(edgeImage)
plt.figure(3)
plt.imshow(outAccum)
plt.title('Hough Transform')
plt.xlabel('Angle (deg)')
plt.ylabel('Rho (pixels)')

plt.figure(4)
plt.imshow(ReturnMaximaImage(outAccum, 15))
plt.title('Hough Transform Peaks (candidate edges)')
plt.xlabel('Angle (deg)')
plt.ylabel('Rho (pixels)')
plt.show()


