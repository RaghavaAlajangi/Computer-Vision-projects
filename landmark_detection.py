# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:39:38 2021

@author: raghava
"""
import cv2
import mediapipe as mp

class Detector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hand_obj = self.mp_hands.Hands()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_obj = self.mp_face_mesh.FaceMesh()

    def get_face_landmarks(self, img_BGR, draw=True):
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        results = self.face_obj.process(cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB))
        face_keypoints = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image=img_BGR,
                                          landmark_list=face_landmarks,
                                          connections=self.mp_face_mesh.FACE_CONNECTIONS,
                                          landmark_drawing_spec=drawing_spec,
                                          connection_drawing_spec=drawing_spec)
                for idx, lm in enumerate(face_landmarks.landmark):
                    h, w, c = img_BGR.shape
                    lm_x, lm_y = int(lm.x*w), int(lm.y*h)
                    face_keypoints.append([idx, lm_x, lm_y])
        return face_keypoints
          
    def get_hand_landmarks(self, img_BGR, draw=True):
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        results = self.hand_obj.process(img_RGB)
        finger_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img_BGR, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img_BGR.shape
                    lm_x, lm_y = int(lm.x*w), int(lm.y*h)
                    finger_keypoints.append([idx, lm_x, lm_y])
        return finger_keypoints