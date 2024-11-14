import cv2 as cv
import mediapipe as mp
import glob
import numpy as np

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 슬라이드 이미지 불러오기 및 리사이즈
img_files = sorted(glob.glob('./images/*.jpg'))  # 슬라이드 경로
if not img_files:
    print("슬라이드 이미지가 없습니다.")
    exit()

current_slide = 0
total_slides = len(img_files)
slide_image = cv.imread(img_files[current_slide])
slide_image = cv.resize(slide_image, (800, 800))  # 슬라이드 이미지를 800x800으로 리사이즈

# 가상 판서용 레이어 초기화
pen_layer = np.zeros_like(slide_image)
drawing = False
prev_x, prev_y = None, None
color = (0, 0, 255)  # 기본 펜 색상
thickness = 2

# 레이저 포인터 설정
laser_pointer_color = (0, 255, 0)  # 레이저 포인터 색상 (녹색)
laser_pointer_radius = 5  # 레이저 포인터 크기 (Reduced from 10 to 5)
is_laser_active = False  # 레이저 포인터 활성화 상태

def calculate_distance(landmark1, landmark2, frame_width, frame_height):
    x1, y1 = int(landmark1.x * frame_width), int(landmark1.y * frame_height)
    x2, y2 = int(landmark2.x * frame_width), int(landmark2.y * frame_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def toggle_laser():
    global is_laser_active
    is_laser_active = not is_laser_active
    print("레이저 포인터 활성화 상태:", is_laser_active)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (800, 800))  # 웹캠 프레임을 800x800으로 리사이즈
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    frame_height, frame_width, _ = frame.shape
    slide_image_copy = slide_image.copy()

    laser_pointer_x, laser_pointer_y = None, None

    # 손의 랜드마크 추적
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 검지 손가락 끝 좌표
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            laser_pointer_x = int(index_finger_tip.x * slide_image_copy.shape[1])
            laser_pointer_y = int(index_finger_tip.y * slide_image_copy.shape[0])

            # 중지 손가락 끝 좌표
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            distance_between_fingers = calculate_distance(index_finger_tip, middle_finger_tip, frame_width, frame_height)

            # 두 손가락 간 거리로 레이저 포인터 활성화/비활성화
            if distance_between_fingers > 50:  # 두 손가락이 벌어져 있을 때
                toggle_laser()

            # 레이저 포인터 표시
            if is_laser_active and laser_pointer_x and laser_pointer_y:
                cv.circle(slide_image_copy, (laser_pointer_x, laser_pointer_y),
                          laser_pointer_radius, laser_pointer_color, -1)

            # 판서 모드
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = calculate_distance(index_finger_tip, thumb_tip, frame_width, frame_height)

            if distance < 40:  # 펜 모드
                drawing = True
                x = laser_pointer_x
                y = laser_pointer_y

                if prev_x is not None and prev_y is not None:
                    cv.line(pen_layer, (prev_x, prev_y), (x, y), color, thickness)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    # 판서 및 레이저 포인터 표시된 슬라이드 표시
    # blended_slide = cv.addWeighted(slide_image_copy, 1, pen_layer, 0.6, 0)

    # Create a mask from the pen_layer where there is drawing
    pen_gray = cv.cvtColor(pen_layer, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(pen_gray, 10, 255, cv.THRESH_BINARY)

    # Copy the pen_layer onto the slide_image_copy where the mask is not zero
    cv.copyTo(src=pen_layer, dst=slide_image_copy, mask=mask)

    # Display the updated slide image
    cv.imshow('Lecture', slide_image_copy)

    # 웹캠 영상 표시
    cv.imshow('Webcam', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):  # 레이저 포인터 활성화/비활성화 토글
        toggle_laser()
    elif key == ord('u'):  # 다음 슬라이드
        if current_slide < total_slides - 1:
            current_slide += 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, (800, 800))  # 슬라이드 이미지를 800x800으로 리사이즈
            pen_layer = np.zeros_like(slide_image)
    elif key == ord('d'):  # 이전 슬라이드
        if current_slide > 0:
            current_slide -= 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, (800, 800))  # 슬라이드 이미지를 800x800으로 리사이즈
            pen_layer = np.zeros_like(slide_image)

cap.release()
cv.destroyAllWindows()
