from imutils import face_utils
import dlib
import imutils
import numpy as np
import cv2

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam=cv2.VideoCapture(0)
while cam.isOpened():
    ret,frame=cam.read()
    image=imutils.resize(frame,width=500)
    image=cv2.flip(image,+1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,1)
    for (i,rect) in enumerate(rects):
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)
        (x,y,h,w)=face_utils.rect_to_bb(rect)
        cropped=image[y:y+h, x:x+w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow("video",image) #fast
        cv2.imshow("face",cropped)

    # cv2.imshow("video1", image) #slow
    # cv2.imshow("face2", cropped)
    c=cv2.waitKey(1)
    if(c==27):
        break
cam.release()
cv2.destroyAllWindows()