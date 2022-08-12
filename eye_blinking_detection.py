import cv2
import imutils
from imutils import face_utils
import numpy as np
import time
import dlib
from scipy.spatial import distance as dist
from threading import Thread

def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])

    ear= (A+B)/(2.0*C)
    return ear

ear_limit=0.28 #can we modifty accouding to me
blink_time_limit=3

count=0
total=0

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cam=cv2.VideoCapture(0)
fileStream=None
time.sleep(1.0)

while cam.isOpened():
    ret,frame=cam.read()
    image=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,0)

    for (i, face) in enumerate(rects):

        shape=predictor(gray,face)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEar=eye_aspect_ratio(leftEye)
        rightEar=eye_aspect_ratio(rightEye)

        ear=(leftEar+rightEar)/2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if(ear < ear_limit):
            count+=1

        else:
            if (count>=blink_time_limit):
                total+=1
            count=0
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("video",frame)
    key=cv2.waitKey(1)
    if(key==27):
        break
cam.release()
cv2.destroyAllWindows()



