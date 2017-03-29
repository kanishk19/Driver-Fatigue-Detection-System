# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 01:06:54 2017

@author: WIN8
"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
faceCascade = cv2.CascadeClassifier('C:\\Users\\WIN8\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gray = gray[ y:y+h , x:x+w ]  
    gray = cv2.resize(gray,(100,100), interpolation = cv2.INTER_AREA)
    eye = gray[20:50,10:90]
    mouth = gray[60:100,25:75]
    eye = cv2.equalizeHist(eye)
    eye = cv2.medianBlur(eye,7)
    eye = cv2.medianBlur(eye,3)
    kernel = np.ones((5,5),np.float32)/25
    mouth = cv2.filter2D(mouth,-1,kernel)
    A = eye.ravel()
    B = mouth.ravel()
    C = np.concatenate((A,B))
    
    x_chk=pd.DataFrame(data=C)
    df=pd.read_csv('df.csv')
    
    x=df.iloc[:,:-1]
    y1=df.iloc[:,-1]
    
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    x=sc.fit_transform(x)
    x_chk=sc.fit_transform(x_chk)
    
    from sklearn.decomposition import PCA
    pca=PCA(n_components=50)
    x_pca=pca.fit_transform(x)
    x_chk_pca=pca.transform(x_chk)
    
    from sklearn.cross_validation import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_pca,y1,test_size=0.20)
    
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_pca,y1)
    print(svc.predict(x_chk_pca))
    

    # Display the resulting frame
    cv2.imshow('Video', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()