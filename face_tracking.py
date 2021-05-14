# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:05:21 2021

@author: ragha
"""
import cv2
from landmark_detection import Detector 

dec = Detector()
cap = cv2.VideoCapture(0)
    
while True:
  success, img_BGR = cap.read()
  dec.get_face_landmarks(img_BGR)  
  cv2.imshow('result', img_BGR)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()