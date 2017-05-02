import imutils
import numpy as np
import argparse
import cv2
import math
import os
from random import randint

face_cascade = cv2.CascadeClassifier(
    '/Users/emreylmz/PycharmProjects/OpenCVExample/haarcascade_frontalface_default.xml')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    else:
        camera = cv2.VideoCapture(args["video"])

    ret, frame = camera.read()
    if ret is True:
        run = True
    else:
        run = False

    firstFrame = None

    isMove = True
    captureCount = 5
    waitTime = 3
    startCapture = False
    width = 720
    # if a video path was not supplied, grab the reference
    # to the gray

    while run:

        # grab the current frame
        (grabbed, frame) = camera.read()

        frame = setupFrame(frame, width)


        #cv2.normalize(frame, frame, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        originalFrame = frame.copy()
        skinMask = frame.copy()
        isFace, userIndicators = detectFace(frame)


        # if the 'r' key is pressed, reset the movement first frame
        if cv2.waitKey(1) & 0xFF == ord("r"):
            firstFrame = None

        if cv2.waitKey(1) & 0xFF == ord("c"):
            print "Frame Changed"
            isMove = not isMove

        if isMove:
            movement, firstFrame = detectMovement(originalFrame, firstFrame)

            frameForSkin = movement
        else:
            frameForSkin = originalFrame

        if isFace:


            skinMask = extractSkin(frameForSkin)

            for idx, faceArea in enumerate(userIndicators[0]):
                personFrame = originalFrame.copy()
                personSkin = skinMask.copy()

                #skin = dimFaces(skin, faceArea)
                handArea = userIndicators[1][idx]
                personSkin = dimFaces(personSkin, faceArea, handArea)
                #skinMask = detectHandArea(skinMask, userIndicators[1][idx])

                #cv2.imshow("Skin Mask", skinMask)

                personFrame, fingerCount, handPosition = findHand(personFrame, personSkin)

                personStr = "Person " + str(idx + 1)

                skin = cv2.bitwise_and(personFrame, personFrame, mask=personSkin)
                cv2.imshow(personStr, np.hstack([personFrame, skin]))
                if fingerCount == 2:
                    captureCount, waitTime, startCapture = captureFrame(cv2.flip(camera.read()[1], 1), True, captureCount, waitTime, startCapture, faceArea, handPosition)
                else:
                    captureCount, waitTime, startCapture = captureFrame(cv2.flip(camera.read()[1], 1), False, captureCount, waitTime, startCapture, faceArea, handPosition)



        # show the skin in the image along with the mask
        cv2.imshow("images", frame)


        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


def remove_bg(frame):
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    fg_mask = cv2.createBackgroundSubtractorMOG2(0, 10).apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    cv2.imshow("Removed", frame)
    return frame


"""class BackGroundSubtractor:
	# When constructing background subtractor, we
	# take in two arguments:
	# 1) alpha: The background learning factor, its value should
	# be between 0 and 1. The higher the value, the more quickly
	# your program learns the changes in the background. Therefore,
	# for a static background use a lower value, like 0.001. But if
	# your background has moving trees and stuff, use a higher value,
	# maybe start with 0.01.
	# 2) firstFrame: This is the first frame from the video/webcam.
	def __init__(self,alpha,firstFrame):
		self.alpha  = alpha
		self.backGroundModel = firstFrame

	def getForeground(self,frame):
		# apply the background averaging formula:
		# NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
		self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.

		return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)"""


def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame


def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    bitWise = cv2.bitwise_and(d1, d2)
    return cv2.threshold(bitWise, 25, 255, cv2.THRESH_BINARY)[1]

def setupFrame(frame, width):

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    #if args.get("video") and not grabbed:
    #    break

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    frame = imutils.resize(frame, width=width)

    frame = cv2.flip(frame, 1)

    return frame



def detectMovement(frame, firstFrame):
    gray_out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_out = cv2.bilateralFilter(gray_out, 5, 50, 100)
    #gray_out = cv2.GaussianBlur(gray_out, (3, 3), 0)
    #gray_out = cv2.medianBlur(gray_out, 11, 0)
    gray_out = denoise(gray_out)

    if firstFrame is None:
        print "Movement Reseted"
        firstFrame = gray_out

    frameDelta = cv2.absdiff(firstFrame, gray_out)
    deltaThres = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    #deltaThres = cv2.threshold(deltaThres, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
    deltaThres = cv2.erode(deltaThres, kernel, iterations=1)
    deltaThres = cv2.dilate(deltaThres, kernel, iterations=1)

    #deltaThres = cv2.GaussianBlur(deltaThres, (15, 15), 0)

    deltaThres = cv2.medianBlur(deltaThres, 11, 0)

    moveSkin = cv2.bitwise_and(frame, frame, mask=deltaThres)

    cv2.imshow("Movement Skin", moveSkin)

    #cv2.normalize(deltaThres, deltaThres, 0, 255, cv2.NORM_MINMAX)


    return moveSkin, firstFrame

def extractSkin(frame):
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    # lower = np.array([0, 10, 60], dtype = "uint8")
    # upper = np.array([20, 150, 255], dtype = "uint8")

    # Ycb cr
    lower = np.array([0, 133, 77], dtype="uint8")
    upper = np.array([255, 173, 127], dtype="uint8")

    # Ycb cr
    # lower = np.array([54,131,110], dtype = "uint8")
    # upper = np.array([163,157,135], dtype = "uint8")

    # HSV
    # lower = np.array([0, 55, 90], dtype = "uint8")
    # upper = np.array([28, 175, 230], dtype = "uint8")

    frameTemp = cv2.GaussianBlur(frame, (11, 11), 0)
    frameTemp = cv2.medianBlur(frameTemp, 11, 0)

    converted = cv2.cvtColor(frameTemp, cv2.COLOR_BGR2YCR_CB)

    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (1, 1))
    skinMask = cv2.erode(skinMask, kernel, iterations=3)
    skinMask = cv2.dilate(skinMask, kernel, iterations=3)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    # skinMask = cv2.medianBlur(skinMask, 3)
    #skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    return skinMask

def detectFace(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    isFace = False
    faceAreas = []
    handAreas = []

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        isFace = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, ((x - 2 * w), (y - h)), ((x + 3 * w), (y + 3 * h)), (255, 0, 0), 2)

        face = np.array([[x, y], [x, y + h*15], [x + w, y + h*15], [x + w, y]])
        handArea = np.array(
            [[(x - 2 * w), (y - h)], [(x - 2 * w), (y + 3 * h)], [(x + 3 * w), (y + 3 * h)], [(x + 3 * w), (y - h)]])

        faceAreas.append(face)
        handAreas.append(handArea)


    userIndicators = [faceAreas, handAreas]

    return isFace, userIndicators

def dimFaces(frame, faceArea, handArea):

    #for faceArea in faceAreas:

    cv2.fillPoly(frame, pts=[faceArea], color=(0, 0, 0))

    maskk = np.zeros_like(frame)
    cv2.fillPoly(maskk, pts=[handArea], color=(255, 255, 255))
    frame = cv2.bitwise_and(frame, frame, mask=maskk)

    return frame

def detectHandArea(frame, handAreas):

    for handArea in handAreas:

        maskk = np.zeros_like(frame)
        cv2.fillPoly(maskk, pts=[handArea], color=(255, 255, 255))
        frame = cv2.bitwise_and(frame, frame, mask=maskk)

    return frame

def findHand(frame, skinMask):
    # Do contour detection on skin region

    _, contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    fingerCount = -1

    handPosition = []
    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(cnt)
        handPosition = np.array([x, y, w, h])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(skinMask.shape, np.uint8)

        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        #cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(frame, far, 1, [255, 255, 255], 10)
                # dist = cv2.pointPolygonTest(cnt,far,True)
                #cv2.line(frame, start, end, [0, 0, 0], 20)
                # cv2.circle(crop_img,far,5,[0,0,255],-1)
            if count_defects == 1:
                cv2.putText(frame, "Two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            elif count_defects == 2:
                cv2.putText(frame, "Three", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            elif count_defects == 3:
                cv2.putText(frame, "Four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

            elif count_defects == 4:
                cv2.putText(frame, "Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            else:
                cv2.putText(frame, "Hello World!!!", (50, 50), \
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            fingerCount = count_defects

            # Draw the contour on the source image
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > 1000:
                    cv2.drawContours(frame, contours, i, (0, 255, 0), 0)

    return frame, fingerCount, handPosition

def cropImage(frame, facePosition, handPosition):

    originalWidth = frame.shape[1]/720.0
    crAreaSize = int((facePosition[3][0] - facePosition[0][0]) * 3 * originalWidth)

    faceX = int((facePosition[0][0] + ((facePosition[3][0] - facePosition[0][0]) / 2)) * originalWidth)
    handMiddleX = int((handPosition[0] + (handPosition[2] / 2)) * originalWidth)
    handMiddleY = int((handPosition[1] + (handPosition[3]/2)) * originalWidth)
    handX = int(handPosition[0] * originalWidth)
    handY = int(handPosition[1] * originalWidth)
    handW = int(handPosition[2] * originalWidth)
    handH = int(handPosition[3] * originalWidth)
    frameW = frame.shape[1]
    frameH = frame.shape[0]

    if handMiddleX <= faceX:
        if handX > crAreaSize:
            #if handMiddleY > crAreaSize/2:
            cropped = frame[0:frameH, handX - crAreaSize:handX]
            #else:
            #   cropped = frame[0:frameH, handX - crAreaSize:handX]
        else:
            cropped = frame[0:frameH, 0:handX]
        print "Left"
        print handPosition[0]
    else:
        if (frameW - handX) > crAreaSize:
            cropped = frame[0:frame.shape[0], handX + handW:handX + handW + crAreaSize]
        else:
            cropped = frame[0:frame.shape[0], handX + handW:frameW]
        print "Right"
        print frame.shape[1]
    print facePosition
    print handPosition
    print frame.shape[0]
    print frame.shape[1]
    return cropped


def captureFrame(frame, capture, captureCount, waitTime, startCapture, facePos, handPos):

    if len(facePos) > 0 and len(handPos) > 0:
        if capture:
            captureCount = captureCount - 1
            print "Captured"
        else:
            captureCount = 5
            #print "Non-Captured"

        if captureCount == 0:
            startCapture = True

        if startCapture:
            waitTime = waitTime - 1
            print "Capturing Started"
            if waitTime == 0:
                print "Capturing Finished"

                cropped = cropImage(frame, facePos, handPos)

                QueueDIR = os.path.dirname(os.path.realpath(__file__)) + "/Queue"
                OriginalDIR = os.path.dirname(os.path.realpath(__file__)) + "/Original"

                fileNumber = randint(10000, 100000)
                fileName = "%d.png" % fileNumber
                originalFileName = "original_" + fileName
                cv2.imwrite(os.path.join(QueueDIR, fileName), cropped)
                cv2.imwrite(os.path.join(OriginalDIR, originalFileName), frame)
                captureCount = 5
                waitTime = 3
                startCapture = False

    return captureCount, waitTime, startCapture


main()