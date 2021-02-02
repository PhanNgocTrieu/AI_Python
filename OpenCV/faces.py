import numpy as np
import cv2
import pickle

#### Implementation Recognizer

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()  #adding "pip install opencv-contrib-python"
recognizer.read("trainner.yml")

labels = {"person": 1}
#load labels name from pickel
with open("D:/File_Documents/OpenCV_Python/Face Detect/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   
    face = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=5)
    # write a picture with face


    #reading position of face:
    for (x,y,w,h) in face:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]   #(ycord_start, cyord_end)
        roi_color = frame[y:y + h, x:x + w]

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])


        img_item = "Image/my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0,255,0) #BGB 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y), (end_cord_x,end_cord_y),color=color,thickness=stroke)

    #Displaying
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Display result
cv2.release()
cv2.destroyAllWindows()
