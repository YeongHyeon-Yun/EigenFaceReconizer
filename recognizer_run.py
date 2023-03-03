import cv2
import numpy as np

# recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer.read('trainer_add_oyr_yyh.yml')

recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read('trainer_trash.yml')

# cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

count = 0
test_count = 1
miss = 0

names = ['adg','jfa','jfp','omg']

cam = cv2.VideoCapture('people_vid.mp4')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret, img = cam.read()
    if ret == False:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.2,
    #     minNeighbors=6,
    #     minSize=(int(minW), int(minH))
    # )

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=1,
        minSize=(300, 300)
    )
    
    test_count += 1
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
        # cv2.imshow('hey', gray[y:y+h, x:x+w])
        id, confidence = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (640, 640)))
        
        if id != 3:
            # cv2.imwrite("/tf/hey_brother/kds_wrong%d.jpg" % miss, img)
            miss += 1
        
        if id == 3 and test_count % 50 == 0:
            # cv2.imwrite("/tf/hey_brother/kds_correct%d.jpg" % count, img)
            count += 1
        
        print('ID =', id, 'confidence=', confidence)
        
        if confidence < 55 :
            id = names[id]
        else:
            id = "unknown"
        
        confidence = "  {0}%".format(round(100-confidence))
        
        # print('AfterID =', id, 'confidence=', confidence)
        
        cv2.putText(img,str(id), (x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)
        
    
    # cv2.imshow('camera',img)
    # if(int(cam.get(1)) % 50 == 0):
    #     cv2.imwrite("/tf/hey_brother/yyh%d.jpg" % count, img)
    #     count += 1
    
    
    if cv2.waitKey(1) > 0 : break
    
acc = 100 * (1 - (miss/test_count))
print('전체 이미지: {}장 중,  잘못잡은 얼굴: {}회'.format(test_count, miss))
print('정확도 >>>', acc,'%')

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()