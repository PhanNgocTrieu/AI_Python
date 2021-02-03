import os
import numpy as np
from PIL import Image
import cv2 # interest in training data
import pickle #using pickle to save Label IDs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'Training Image')

face_cascades = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()  #adding "pip install opencv-contrib-python"

current_id = 0
label_ids = {}
y_lables = []
x_train = []


# reading all of image file in folder name of image_dir
# we can change the direction of image_dir
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            #label from dictionaries
            #label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(label,path)

            #creating id for labels:
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            # not doing this much
            #y_lables.append(label) # some number
            #x_train.append(path) # verify this image, turn in to NUMPY array, GRAY

            # training image to number array: meaning color pixel
            pil_image = Image.open(path).convert("L") #grayscale
            image_array = np.array(pil_image, "uint8") #convert
            #print(image_array)


            #the region of interest in training data
            faces = face_cascades.detectMultiScale(image_array, scaleFactor= 1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_lables.append(id_) #add lable id for lables
            # Creating Training labels: we know the lables but have no ID for labels so now we do

#print(y_lables)
#print(x_train)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_lables))
recognizer.save("trainner.yml")