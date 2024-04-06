import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self. max_hands,
                                        min_detection_confidence=self.detection_confidence,
                                        min_tracking_confidence=self.tracking_confidence)

    def FindHands(self, img, draw=True):
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(img_RGB)

        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for hand_land_marks in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_land_marks, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def find_position(self, img, hand_number=0, draw=True):
        landmarks_list=[]
        
        if self.result.multi_hand_landmarks:
            hand_land_marks = self.result.multi_hand_landmarks[hand_number]

            for id, land_mark in enumerate(hand_land_marks.landmark):
                height, width, _ = img.shape
                x_center, y_center = int(land_mark.x * width), int(land_mark.y * height)

                landmarks_list.append([id, x_center, y_center])

                if draw:
                    cv.putText(img, str(id), (x_center, y_center), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            
        return landmarks_list

def main():
    previous_time = 0
    current_time = 0  

    img = detector = HandDetector() 

    cap = cv.VideoCapture(0)

    while True:
        success, img = cap.read() 

        detector.FindHands(img)
        landmarks_list = detector.find_position(img)

        if len(landmarks_list) != 0:
            print(landmarks_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv.imshow("Image", img)
        if cv.waitKey(1) == 27: # Escape
            break

if __name__ == "__main__":
    main()