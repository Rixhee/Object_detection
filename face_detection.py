import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("unnamed.jpg")
fix_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

face_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_frontalface_default.xml")

faces = face_classifier.detectMultiScale(fix_image, 1.3, 5)

if faces is ():
    print("No face found")

def detect_face(fix_image):
    face_rect = face_classifier.detectMultiScale(fix_image)

    for (x, y, w, h) in face_rect:
        cv.rectangle(fix_image, (x,y), (x+w, y+h), (255, 0, 0), 4)

    return fix_image

result = detect_face(fix_image)
plt.imshow(result)
plt.show() 



