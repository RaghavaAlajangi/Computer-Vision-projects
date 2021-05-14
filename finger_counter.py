# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:55:13 2021

@author: raghava
"""

import cv2
import time
from landmark_detection import Detector 

dec = Detector()
finger_tips = [8, 12, 16, 20]
web_cam = cv2.VideoCapture(0)
t1 = time.time()

while True:
    succ, img_BGR = web_cam.read()
    t2 = time.time()
    frame_rate = int(1/(t2-t1))
    t1 = t2
    finger_ids = dec.get_hand_landmarks(img_BGR, draw=True)
    #face_ids = dec.get_face_landmarks(img_BGR, draw=True)
    
    if len(finger_ids)!= 0:
        finger_count = []
        # Thumb detection
        if finger_ids[4][1] >= finger_ids[4-2][1]:
            finger_count.append(1)
        else:
            finger_count.append(0)
        # Femaining fingers
        for i in finger_tips:
            if finger_ids[i][2] <= finger_ids[i-2][2]:
                finger_count.append(1)
            else:
                finger_count.append(0)
        total_count = finger_count.count(1)
        cv2.putText(img_BGR, str(total_count), (45, 140), cv2.FONT_HERSHEY_PLAIN, 8, (255,0,0), 20)
        
    cv2.rectangle(img_BGR, (20, 20), (150, 160), (0,255,0), 4)
    cv2.putText(img_BGR, str(frame_rate), (550, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 5)
    cv2.imshow('img', img_BGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
web_cam.release()
cv2.destroyAllWindows()