import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv

image = cv.imread("jpg.jpg")
fix_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

car_plate_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_russian_plate_number.xml")

def detect_car_plate(fix_image):
    car_plate_rectangle = car_plate_classifier.detectMultiScale(fix_image)

    for (x,y,w,h) in car_plate_rectangle:
        cv.rectangle(fix_image, (x,y), (x+w, y+h), (255, 255, 0), 5)

    return fix_image

result = detect_car_plate(fix_image)
plt.imshow(result)
plt.show()
