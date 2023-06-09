import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


section = 0
time=0
id=0

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

print("Before")
print("Image List: ")
print(images)
print("ClassNames List: ")
print(classNames)
print("---")

print("First myList after reading directory...")
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("After")
print("images: ")
print(images)
print("ClassNames: ")
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name,time,id,section):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},\t{time}{dtString},\t{id},\t{section}')

markAttendance('Name','     Time','Id','Section')

 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
   
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        #print(faceDis)
        matchIndex = np.argmin(faceDis)
    
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
          
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            
            if (name == "Raihan Sikdar".upper()):
                markAttendance(name,time, 133, 2)
                cv2.putText(img,name+" ID 133",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            elif (name == "Lionel Messi".upper()):
                markAttendance(name,time, 128, 2)
                cv2.putText(img,name+" ID 128",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            elif (name == "Akib Khan".upper()):
                markAttendance(name,time,  219, 2)
                cv2.putText(img,name+" ID 219",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            elif (name == "Kazi Sarwar".upper()):
                markAttendance(name,time, 231, 2)
                cv2.putText(img,name+" ID 281",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            elif (name == "Sifat".upper()):
                markAttendance(name,time,  281, 2)
                cv2.putText(img,name+" ID 281",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            elif (name == "Peal".upper()):
                markAttendance(name,time,  181, 2)
                cv2.putText(img,name+" ID 181",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name,time,id,section)
        else:
            name = 'Unknown'
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)