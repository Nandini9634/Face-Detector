#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=10)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
    return face_img

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frame = cv2.imread('group.jpg')
frame = detect_face(frame)

cv2.imshow('Face Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




