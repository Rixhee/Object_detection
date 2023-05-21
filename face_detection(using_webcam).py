import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

face_classifier = cv.CascadeClassifier("haarcascade_files/haar-cascade-files-master/haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 5)

    cv.imshow("faces", frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

 
