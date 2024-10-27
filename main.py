import cv2
import mediapipe as mp
import pyautogui

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, hands_landmarks, hands.HAND_CONNECTIONS)
            landmarks = hands_landmarks.landmark
            if landmarks[8]:
                screen_x = screen_w * landmarks[8].x
                screen_y = screen_h * landmarks[8].y
                pyautogui.moveTo(screen_x, screen_y)
            for landmark in landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                cv2.circle(frame, (x, y), 5, (0, 255, 0))

    cv2.imshow('Hand Gestures', frame)
    cv2.waitKey(1)
