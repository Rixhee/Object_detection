import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

image = cv.imread("unnamed.jpg")
fix_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

eyes_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_eye.xml")

def detect_eyes(fix_image):
    eyes_rect = eyes_classifier.detectMultiScale(fix_image)

    for (x,y,w,h) in eyes_rect:
        cv.rectangle(fix_image, (x,y), (x+w, y+h), (255, 255, 0), 3)

    return fix_image

result = detect_eyes(fix_image)
plt.imshow(result)
plt.show()