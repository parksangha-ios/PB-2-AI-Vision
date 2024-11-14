import cv2
import mediapipe as mp
import time
import math
import numpy as np

def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
img = cv2.imread("lecture.png")  # 초기에는 강의 자료를 표시

def resize_to_match(frame, img):
    frame_height, frame_width = frame.shape[:2]
    img_resized = cv2.resize(img, (frame_width, frame_height))
    return img_resized

hands = mp_hands.Hands(max_num_hands=1)

# 상태 및 타이머 변수 초기화
mode = 'lecture'  # 현재 모드: 'lecture' 또는 'break'
timer_running = False
start_time = None
total_time = 10  # 타이머 시간 (10초)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        points = hand_landmarks.landmark

        fingers = [0, 0, 0, 0, 0]

        # 엄지손가락
        if points[4].x < points[3].x:
            fingers[0] = 1  # 오른손 기준
        else:
            fingers[0] = 0

        # 나머지 손가락
        tips = [8, 12, 16, 20]
        for i, tip in enumerate(tips):
            if points[tip].y < points[tip - 2].y:
                fingers[i+1] = 1
            else:
                fingers[i+1] = 0

        if sum(fingers) == 0:
            # 주먹 동작 인식
            if not timer_running:
                start_time = time.time()
                timer_running = True
    else:
        # 손이 인식되지 않음
        pass

    if timer_running:
        elapsed_time = time.time() - start_time
        if mode == 'lecture':
            # 강의 모드에서 타이머는 0부터 10초까지 증가
            display_time = int(elapsed_time)
            if elapsed_time >= total_time:
                mode = 'break'
                start_time = time.time()  # 타이머 재설정
        elif mode == 'break':
            # 쉬는 시간 모드에서 타이머는 10부터 0까지 감소
            display_time = int(total_time - elapsed_time)
            if display_time <= 0:
                mode = 'lecture'
                timer_running = False  # 타이머 종료

        # 남은 시간을 프레임에 표시
        cv2.putText(
            frame,
            f"Timer: {display_time} sec",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    # 현재 모드에 따라 이미지 변경
    if mode == 'lecture':
        img = cv2.imread("lecture.png")
    elif mode == 'break':
        img = cv2.imread("breaktime.png")

    img_resized = resize_to_match(frame, img)
    combined = cv2.hconcat([frame, img_resized])
    cv2.imshow("lecture", combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
