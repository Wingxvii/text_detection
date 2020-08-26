"""
.. module:: text_detection
   :synopsis: Detects text from input images

.. moduleauthor:: John Wang <john.wang@uoit.net>
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import glob
import os

# confidence value
conf = 0.9

# resize value
width = 640 # must be multiple of 32
height = 352 # must be multiple of 32

# load the input image and grab the image dimensions
outputFolder = "C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/opencv-text-detection/Cropped/{}-{}.png"

# size ratios
rW = 1
rH = 1


def drawBoxes(boxes, frame):
	"""
	Draw boxes over detected text

	:param boxes: array of boxes coordinates in order of startX, startY, endX, endY
	:type boxes: int[[]]
	:param frame: source cv image matrix
	:type frame: mat
	"""

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the image
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

		# show the output image
		cv2.imshow("Text Detection", frame)
		cv2.waitKey(0)


# Function returns output boxes in memory
def cutBoxes(boxes, image):
	"""
	Cut images using detected boxes and returns it

	:param boxes: array of boxes coordinates in order of startX, startY, endX, endY
	:type boxes: int[[]]
	:param image: source cv image matrix
	:type image: mat
	:return: list of cut text cv image matrices
	:rtype: mat[]
	"""

	buffer = 4
	images = []
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		if startY > buffer:
			startY = startY - buffer

		if startX > buffer:
			startX = startX - buffer

		if endX < (width-buffer):
			endX = endX + buffer

		if endY < (height-buffer):
			endY = endY + buffer + buffer

		newImage = image[startY:endY, startX:endX]
		images.append(newImage)

	return images


def gray_out(image, startX, startY, endX, endY):
	"""
	Grays out inverse of detected boxes, perserving image sizes

	:param image: source cv image matrix
	:type image: mat
	:param startX: first coordinates for x axis
	:type startX: int
	:param startY: first coordinates for y axis
	:type startY: int
	:param endX: second coordinates for x axis
	:type endX: int
	:param endY: second coordinates for y axis
	:type endY: int
	:return: grayed out image cv image matrix
	:rtype: mat
	"""
	stencil = np.zeros(image.shape).astype(image.dtype)
	contours = [np.array([[startX, startY],[startX,endY],[endX,endY],[endX,startY]])]
	color = [255, 255, 255]
	cv2.fillPoly(stencil, contours, color)
	result = cv2.bitwise_and(image, stencil)

	return result


def saveBoxes(boxes, image, basename='test'):
	"""
	Saves detected boxes to disk

	:param boxes: array of boxes coordinates in order of startX, startY, endX, endY
	:type boxes: int[[]]
	:param image: source cv image matrix
	:type image: mat
	:param basename: filename prefix
	:type basename: str
	"""
	counter = 0
	buffer = 4
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		if startY > buffer:
			startY = startY - buffer

		if startX > buffer:
			startX = startX - buffer

		if endX < (width-buffer):
			endX = endX + buffer

		if endY < (height-buffer):
			endY = endY + buffer + buffer

		newImage = image[startY:endY, startX:endX]

		if image is not None:
			cv2.imwrite(outputFolder.format(basename, counter), newImage)
			counter += 1


def check_image(image, startX, startY, endX, endY, buffer=0):
	"""
	Function that returns true or false, for if an image exists within the bounding box

	:param image: source cv image matrix
	:type image: mat
	:param startX: first coordinates for x axis
	:type startX: int
	:param startY: first coordinates for y axis
	:type startY: int
	:param endX: second coordinates for x axis
	:type endX: int
	:param endY: second coordinates for y axis
	:type endY: int
	:param buffer: image detection pixel buffer size
	:type buffer: int
	:return: wether text was found
	:rtype: bool
	"""

	startX -= buffer
	startY -= buffer
	endX += buffer
	endY += buffer

	stencil = np.zeros(image.shape).astype(image.dtype)
	contours = [np.array([[startX, startY],[startX,endY],[endX,endY],[endX,startY]])]
	color = [255, 255, 255]
	cv2.fillPoly(stencil, contours, color)
	result = cv2.bitwise_and(image, stencil)

	boxes = detect(result)
	# show the output image
	cv2.imshow("Text Detection", result)
	cv2.waitKey(0)
	if boxes.__len__() > 0:
		return True
	else:
		return False

# Detection method
def detect(image):
	"""
	Function to perform detection on an image

	:param image: source cv image matrix to perform detection
	:type image: mat
	:return: detected boxes list by it's coordinates in order of startX, startY, endX, endY
	:rtype: int[[]]
	"""
	global rW
	global rH

	orig = image.copy()
	(H, W) = image.shape[:2]

	# resize
	(newW, newH) = (width, height)
	rW = W / float(newW)
	rH = H / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	# load the pre-trained EAST text detector
	net = cv2.dnn.readNet('C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/opencv-text-detection/frozen_east_text_detection.pb')

	# construct a blob from the image and then perform a forward pass of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):

			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < conf:
				continue

			# print("Found! Conf:{}".format(scoresData[x]))

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	# drawBoxes(boxes, orig)
	return boxes


"""
Example Use Code


video = cv2.VideoCapture('C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/candycrush.mp4')
counter = 0
iterator = 0
skip = 30
# Read until video is completed
while(video.isOpened()):
	# Capture frame-by-frame
	ret, frame = video.read()

	if iterator <= skip:
		iterator += 1
		continue

	# Use Example
	if ret is True:
		if check_image(frame, 269, 232, 359, 254, 6) is True:
			print("Found!")

		'''
		boxes = detect(frame)
		check_image(frame, boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
		# images = cutBoxes(boxes, frame)
		# saveBoxes(boxes, frame, counter)
		print("Checked image {}, {} text boxes was found.".format(counter, len(boxes)))
		'''

		counter += 1
		iterator = 0
	else:
		break

# When everything done, release the video capture object
video.release()

# Closes all the frame
cv2.destroyAllWindows()
"""