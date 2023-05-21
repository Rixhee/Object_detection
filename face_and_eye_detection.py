import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np

image = cv.imread("unnamed.jpg")
fix_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

eye_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_eye.xml")
face_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_frontalface_default.xml")

faces = face_classifier.detectMultiScale(fix_image, 1.3, 5)

if faces is ():
    print("No face found")

def detect_eye_face(fix_image):
    face_rect = face_classifier.detectMultiScale(fix_image)
    eye_rect = eye_classifier.detectMultiScale(fix_image)
    
    for (x,y,w,h) in face_rect:
        cv.rectangle(fix_image, (x,y), (x+w, y+h), (255, 0, 0), 4)

        for (ix,iy,iw,ih) in eye_rect:
            cv.rectangle(fix_image, (ix, iy), (ix+iw, iy+ih), (255, 0, 255), 5)
    
    return fix_image

result = detect_eye_face(fix_image)
plt.imshow(result)
plt.show() 