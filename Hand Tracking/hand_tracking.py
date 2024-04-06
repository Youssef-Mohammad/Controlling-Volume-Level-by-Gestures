import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

previous_time = 0

while True:
    success, img = cap.read()

    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(img_RGB)

    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for hand_land_marks in result.multi_hand_landmarks:
            for id, land_mark in enumerate(hand_land_marks.landmark):
                height, width, color_channels = img.shape
                x_center, y_center = int(land_mark.x * width), int(land_mark.y * height)
                cv.putText(img, str(id), (x_center, y_center), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)


            mp_draw.draw_landmarks(img, hand_land_marks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv.imshow("Image", img)
    if cv.waitKey(1) == 27: # Escape
        break