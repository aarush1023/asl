import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pandas as pd
from mediapipe import solutions
import csv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# data = {str(n): [] for n in range(21*3*40)}
# data_copy = pd.DataFrame(data)
# data_copy.to_csv('output.csv')
started = False
counter = 0
alphabet = 1
coords = []

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

# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # global image

    # print('hand landmarker result: {}'.format(result))
    # draw the landmarks on the image
    # image = draw_landmarks_on_image(output_image,w result)
    # if cv2.waitKey(1) == ord('e'):
    #     for landmark_list in result.hand_landmarks:
    #         for landmark in landmark_list:
    #             file1 = open('output.csv', 'a')
    #             print(landmark.x, landmark.y, landmark.z, file=file1)
    #             file1.close()

    # image = draw_landmarks_on_image(output_image.numpy_view(), result)
    global HandLandmarkerResult
    HandLandmarkerResult = result
    # global started
    # if started:
    #     if len(result.hand_landmarks) > 0:
    #         write_to_csv(result)
            # global counter
            # counter += 1
            # print("check")
            # global alphabet
            # print(result)
            # for landmark_list in result.hand_landmarks:
            #     print("for")
            #     for landmark in landmark_list:
            #         global data
            #         print(landmark.x)
            #         data[str(counter*3)].append(landmark.x)
            #         print(1)
            #         data[str(counter*3 + 1)].append(landmark.y)
            #         print(2)
            #         data[str(counter*3 + 2)].append(landmark.z)
            #         print("after")


def write_to_csv(result):
    # df = pd.read_csv('output1.csv')
    # counter2 = 0
    global coords
    global alphabet
    # print(result.hand_landmarks)
    # for landmark_list in result.hand_landmarks:
    #     for landmark in landmark_list:
    #         data[str(counter*63 + 3*counter2)].append(landmark.x)
    #         data[str(counter*63 + 3*counter2 + 1)].append(landmark.y)
    #         data[str(counter*63 + 3*counter2 + 2)].append(landmark.z)
    #         counter2 += 1
    for landmark_list in result.hand_landmarks:
        for landmark in landmark_list:
            coords.append(landmark.x)
            coords.append(landmark.y)
            coords.append(landmark.z)
    if counter == 39:
        coords.append(alphabet)
        with open('output1.csv', mode='a', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(coords)
            output.close()
            coords = []
        # print(len(coords))
        # df2 = pd.DataFrame([coords], columns=df.columns)
        # df = pd.concat([df, df2])
        # print(df.head())
        # df.to_csv('output1.csv')
        # coords = []
    # with open('output.csv', 'a', newline='') as file:
    #     fieldnames = data.keys()
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     if file.tell() == 0:
    #         writer.writeheader()
            
    #     writer.writerow(data)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.5,  # lower than value to get predictions more often
    min_hand_presence_confidence=0.5,  # lower than value to get predictions more often
    min_tracking_confidence=0.5,
    result_callback=print_result)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

#    with HandLandmarker.create_from_options(options) as landmarker:
    image = cv2.flip(image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    landmarker.detect_async(image=mp_image, timestamp_ms=int(time.time() * 1000))
    # if started:
    #     counter += 1
    image = draw_landmarks_on_image(image, HandLandmarkerResult)
    
    if started:
        if len(HandLandmarkerResult.hand_landmarks) > 0:
            write_to_csv(HandLandmarkerResult)
            counter += 1

    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1)
    if key == ord('j'):
        alphabet -= 1
        print(alphabet)
    if key == ord('k'):
        started = True
        print('started')
    if counter == 40:
        print(alphabet)
        started = False
        counter = 0
    if key == ord('l'):
        alphabet+=1
        print(alphabet)
    if key == ord('q'):
        break

# data = pd.DataFrame(data)
# data.to_csv('output.csv')
cv2.waitKey()
cv2.destroyAllWindows()
