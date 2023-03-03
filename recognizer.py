import cv2
import numpy as np
from PIL import Image #python imaging library
import os

path = './dataset' #경로 (dataset 폴더)
recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = ['adg','jfa','jfg','omg']


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #listdir : 해당 디렉토리 내 파일 리스트
    #path + file Name : 경로 list 만들기

    faceSamples = []
    ids = []
    for imagePath in imagePaths: #각 파일마다
        #흑백 변환
        PIL_img = Image.open(imagePath).convert('L') #L : 8 bit pixel, bw
        img_numpy = np.array(PIL_img, 'uint8')
        
        # print('여기',os.path.split(imagePath)[-1].split("."))
        
        #user id
        # id = int(os.path.split(imagePath)[-1].split(".")[0][:3])#마지막 index : -1
        # 사진 파일 형식: omg69.jpg
        for idx, i in enumerate(names):
            if i == os.path.split(imagePath)[-1].split(".")[0][:3]:    # 이름값만 추출하여 라벨과 비교
                id = idx
                # print(id, i)
                
        #얼굴 샘플
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(cv2.resize(img_numpy[y:y+h,x:x+w], (640, 640)))
            ids.append(id)

    return faceSamples, ids

print('\n [INFO] Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)

recognizer.train(faces,np.array(ids)) #학습

recognizer.write('trainer_trash.yml')
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))