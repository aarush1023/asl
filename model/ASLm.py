# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:15:33 2024

@author: Aarush Jain
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from joblib import dump

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

started = False
counter = 0
alphabet = 27
coords = []
alpha = [chr(i) for i in range(65, 91)] + ["None"]

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            #  print(hand_landmarks_list,len(hand_landmarks_list))
            handedness_list = detection_result.handedness
            #  print(handedness_list,len(handedness_list))
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    hand_landmarks])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            return annotated_image
    except Exception as e:
        print(e)
        print(3)
        return rgb_image
    

def draw_rect(image, result):
    maxx = 0
    minx = 5000000000
    maxy = 0
    miny = 5000000000
    for landmark_list in result.hand_landmarks:
        for landmark in landmark_list:
            if landmark.x > maxx:
                maxx = landmark.x
            if landmark.x < minx:
                minx = landmark.x
            if landmark.y  > maxy:
                maxy = landmark.y
            if landmark.y < miny:
                miny = landmark.y
    new_image = np.copy(image)
    # cv2.rectangle(new_image, (minx*image.shape[0], miny*image.shape[1]), (maxx*image.shape[0], maxy*image.shape[1]), (255, 0, 0))
    if result.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                # Extract bounding box coordinates
            bounding_box = cv2.boundingRect(np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark]))

    # Normalize coordinates
            x, y, w, h = bounding_box
            x_min, y_min, x_max, y_max = x / image_width, y / image_height, (x + w) / image_width, (y + h) / image_height

# Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return new_image
    
    
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    global HandLandmarkerResult
    HandLandmarkerResult = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    
    
    min_hand_detection_confidence=0.5,  # lower than value to get predictions more often
    min_hand_presence_confidence=0.5,  # lower than value to get predictions more often
    min_tracking_confidence=0.5,
    result_callback=print_result)

landmarker = HandLandmarker.create_from_options(options)

df = pd.read_csv('output1.csv')
x = df.loc[:, df.columns != 'OUT']
y = df['OUT']
model = LogisticRegression(max_iter=100000)
model.fit(x, y)
dump(model, 'ASLm.joblib')

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    if not success:
        print("ignoring empty camera frame")
        continue
    
    if started:
        if len(HandLandmarkerResult.hand_landmarks) > 0:
            counter += 1 
            for landmark_list in HandLandmarkerResult.hand_landmarks:
                for landmark in landmark_list:
                    coords.append(landmark.x)
                    coords.append(landmark.y)
                    coords.append(landmark.z)
    if counter == 40:
        pred = model.predict([coords])
        coords = []
        print(pred)
        started = False
        counter= 0
        alphabet = pred[0]
    image = cv2.flip(image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    landmarker.detect_async(image=mp_image, timestamp_ms=int(time.time() * 1000))
    # if started:
    #     counter += 1
    image = draw_landmarks_on_image(image, HandLandmarkerResult)
    cv2.putText(img=image, text=alpha[alphabet-1], org=(0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('k'):
        started = True
        print('started')

cv2.destroyAllWindows()
