# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 02:15:01 2017

@author: WIN8
"""


import cv2
import numpy as np
import pandas as pd

faceCascade = cv2.CascadeClassifier('C:\\Users\\WIN8\\Anaconda2\\Library\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_default.xml')
# Read the image
#iteration
e= np.empty([1,4400])
for i in range(1,133):
    image = cv2.imread('C:\\Users\\WIN8\\Desktop\\DFDS\\New\\IMG-20170121-WA00{}.jpg'.format(i))
    print(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
                                         gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(5,5),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                        )

    print("Found {} faces!".format(len(faces)))
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #cut out the face part in the image
        gray = gray[ y:y+h,x:x+w]
    print(gray.shape)
    
    #scale down the image
    gray2 = cv2.resize(gray,(100,100), interpolation = cv2.INTER_AREA)
    
    #gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #for eye and nose
    cv2.rectangle(gray2, (10,20),(90,50),255,2)
    cv2.rectangle(gray2, (25,60),(75,100),255,2)
    
    #for filtering
    eye = gray2[20:50,10:90]
    mouth = gray2[60:100,25:75]
    #for eye
    eye = cv2.equalizeHist(eye)
    eye = cv2.medianBlur(eye,7)
    eye = cv2.medianBlur(eye,3)
    print(eye.shape)
    print(eye)
    #for mouth
    kernel = np.ones((5,5),np.float32)/25
    mouth = cv2.filter2D(mouth,-1,kernel)
    
    #cv2.imshow("Faces found", eye)
    
    #cv2.imshow("Faces found", mouth)
    A = eye.ravel()
    print(A.shape)
    print(A)
    B = mouth.ravel()
    C = np.concatenate((A,B))
    print(C.shape)
    print(C)
    #d = C.tolist()
    e=np.vstack((e,C))
   # e.append(d) 
    #np.delete(C)
print (e.shape)
#e = e[2:4,:]
x=pd.DataFrame(data=e)
x = x.drop(x.index[[0]])


print(x.shape)
y=[-1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,
   -1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,
   1,-1,1,1,-1,-1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,1,-1,
   -1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1]
   
x[4400] = np.asarray(y) 

x.to_csv('df.csv',index=False)

#C.to_xls("dataset")
cv2.waitKey(0)
