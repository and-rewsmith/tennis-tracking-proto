# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

class CustomFrame:
	def __init__(self, frame, is_hit):
		self.frame = frame
		self.is_hit = is_hit 
		self.show_anyway = False 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# lower and upper boundaries of the "green" in HSV
# TODO: why is this rgb????
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# initialize the list of tracked points, to draw the ball trayectory
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, use the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

frame_state = list()

# infinite loop
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# cv2.imshow("hsv", hsv)
	# input()

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	fgmask = fgbg.apply(frame)
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.bitwise_and(mask, mask, mask=fgmask)

	# cv2.imshow("mask", mask)
	# input()

	# mask = cv2.dilate(mask, None, iterations=1)
	# cv2.imshow("dilate", mask)
	# input()

	# mask = cv2.erode(mask, None, iterations=1)
	# cv2.imshow("erode", mask)
	# input()

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST,
		cv2.CHAIN_APPROX_NONE)[-2]
	center = None

	# print(len(cnts))

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		# print(M)

		desired_radius = .1

		# break if moment is 0
		if M["m00"] != 0:
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > desired_radius:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
		elif radius > desired_radius:
			# print(c)
			center = (c[0][0][0], c[0][0][1])
		# else:
			# print(radius)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	# cv2.imshow("Frame", frame)
	# print(frame)
	# cv2.imshow("Frame", mask)

	is_hit = center != None
	frame_state.append(CustomFrame(frame.copy(), is_hit))

	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# TODO: can reduce time complexity if we care
FRAME_BUFFER = 30 
prev = None
for i, state in enumerate(frame_state):
	if state.is_hit and prev != None and prev.is_hit == True:
		index = min(i + FRAME_BUFFER // 2, len(frame_state)-1)
		count = 0 
		while count != FRAME_BUFFER and index >= 0:
			frame_state[index - 1].show_anyway = True 
			index -= 1
			count += 1
	prev = state

for state in frame_state:
	if state.is_hit or state.show_anyway:
		cv2.imshow("test", state.frame)
		cv2.waitKey(100);


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()