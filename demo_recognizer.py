import cv2
import numpy as np
import os 
import time
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def run_recognizer(img_path):
    before = time.time()
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read('trainer_trash.yml')

    # recognizer = cv2.face.FisherFaceRecognizer_create()
    # recognizer.read('trainer_trash.yml')
    
    # cascadePath = 'haarcascade_frontalface_default.xml'
    # faceCascade = cv2.CascadeClassifier(cascadePath)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    names = ['Abdullah Gul','Jennifer Aniston','Jennifer Garner','omg']
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=1,
    minSize=(30, 30)
    )

    id = 0
    find_max = []
    max_conf = 0
    max_id = 0

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (640, 640)))
        print(id, confidence)
        if confidence >= max_conf:
            max_conf = confidence
            max_id = id 
            
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
        cv2.putText(img,str(names[id]), (x-70, y+110),font,1,(0,255,0),2)
        # cv2.putText(img,str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)
        img = cv2.resize(img, (640,480))
        cv2.imwrite("result/result3.jpg", img)
        break
            
    name = names[max_id]
    after = time.time()
    print(after - before)
    return name
        
print(run_recognizer('./test/Jennifer Garner.jpg'))
