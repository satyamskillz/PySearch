# visualize_facial_landmarks(image, shape, colors=None, alpha=0.75) is not part of imutils liberary

import dlib
import cv2
import numpy as np
import imutils
from imutils import face_utils

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam=cv2.VideoCapture(0)
while cam.isOpened():
    ret,frame=cam.read()
    image=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,1)

    for (i,face) in enumerate(rects):
        shape=predictor(gray,face)
        shape=face_utils.shape_to_np(shape)

        output=face_utils.visualize_facial_landmarks(image,shape)
        cv2.imshow("sdsd",output)

        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_68_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                cv2.imshow("video",roi)
                cv2.imshow("video1", clone)
            cv2.waitKey(0)
    c=cv2.waitKey(1)
    if(c==27):
        break
cam.release()
cv2.destroyAllWindows()