import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initializing Sound Configuration
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

######################################################
camera_width, camera_height = 640, 480 # Video dimensions
######################################################

# Setting the camera
cap = cv.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

# To calculate the FPS later
previous_time = 0

# Initialization of the volume and the volume bar
vol_bar = round(volume.GetMasterVolumeLevelScalar() * 100)
vol = volume.GetMasterVolumeLevelScalar()

# Creating an instance of the hand detector module
# Captures only hands that are detected by confidence at least 0.7
detector = htm.HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()

    # Detecting Hands
    img = detector.FindHands(img)

    # Storing hands' landmarks
    landmarks_list = detector.find_position(img, draw=False)

    if len(landmarks_list) != 0:
        # Keeping the landmarks of the index and thumb only
        x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]

        # Calculating a center point
        x_center, y_center = (x1 + x2) // 2, (y1 + y2) //2

        # Marking them with circles
        cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x_center, y_center), 10, (255, 0, 255), cv.FILLED)

        # make a line between the index and the thumb to measure its length
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Euclidean distance
        length = np.hypot(x1 - x2, y1 - y2)

        # Just some coloring :)
        if length < 30:
            cv.circle(img, (x_center, y_center), 10, (0, 255, 0), cv.FILLED)
        
        if length > 250:
            cv.circle(img, (x_center, y_center), 10, (0, 0, 255), cv.FILLED)

        # Calculating the value of the new sound level
        vol = np.interp(length, [30, 250], [0.0, 1.0])

        # The height of the interactive bar
        vol_bar = int(np.interp(length, [30, 250], [400, 150]))

        # Setting the sound value
        volume.SetMasterVolumeLevelScalar(vol, None)

    # Creating an interactive bar
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv.FILLED)

    # Calculating the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Show the sound level and FPS on the screen
    cv.putText(img, "FPS: " + str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.putText(img, str(round(volume.GetMasterVolumeLevelScalar() * 100)) + '%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv.imshow("Image", img)

    if cv.waitKey(1) == 27:
        break