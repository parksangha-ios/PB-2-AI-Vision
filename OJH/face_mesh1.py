import cv2 as cv
import mediapipe as mp
import glob
import time
import numpy as np

mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

mesh = mp_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# 이미지 슬라이드 설정
img_files = sorted(glob.glob('./images/*.jpg'))  # 슬라이드 경로 확인
if not img_files:
    print("No slide images found.")
    exit()
current_slide = 0
total_slides = len(img_files)
slide_image = cv.imread(img_files[current_slide])
slide_image = cv.resize(slide_image, (800, 800))

eye_closed_time = None  # 눈 감은 시간 기록
EYE_CLOSED_THRESHOLD = 3  # 눈 감은 상태 유지 시간 (초)

def is_mouth_open(landmarks, threshold=0.07):
    mouth_top = landmarks.landmark[13]
    mouth_bottom = landmarks.landmark[14]
    mouth_ratio = abs(mouth_top.y - mouth_bottom.y)
    return mouth_ratio > threshold  # 임계값 초과 시 입 벌림 감지

def apply_mosaic(frame, face_landmarks, padding=50):
    landmark_points = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in face_landmarks.landmark]

    #사각형 영역 설정
    x_min = min(point[0] for point in landmark_points) - padding
    y_min = min(point[1] for point in landmark_points) - padding
    x_max = max(point[0] for point in landmark_points) + padding
    y_max = max(point[1] for point in landmark_points) + padding

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, frame.shape[1])
    y_max = min(y_max, frame.shape[0])

    #모자이크 효과
    mosaic_frame = cv.blur(frame, (50, 50))  # 블러 강도 조절 가능

    # 얼굴 사각형 영역만 원본으로 복사
    mosaic_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]

    return mosaic_frame

def is_eye_closed(landmarks):
    # 왼쪽 눈 및 오른쪽 눈의 상단-하단 거리 계산
    left_eye_top = landmarks.landmark[159]
    left_eye_bottom = landmarks.landmark[145]
    left_eye_ratio = abs(left_eye_top.y - left_eye_bottom.y)
    
    right_eye_top = landmarks.landmark[386]
    right_eye_bottom = landmarks.landmark[374]
    right_eye_ratio = abs(right_eye_top.y - right_eye_bottom.y)
    
    # 눈 감은 상태 판단
    if left_eye_ratio < 0.01 and right_eye_ratio < 0.01:
        return True
    return False

is_mosaic_active = False
last_mouth_open_time = None  # 마지막으로 입을 벌린 시간
MOSAIC_DURATION = 3  # 모자이크가 활성화된 후 3초 후 비활성화

cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            # 입 벌림 상태 확인
            if is_mouth_open(landmarks):
                if not is_mosaic_active:
                    is_mosaic_active = True  # 모자이크 활성화
                    last_mouth_open_time = time.time()  # 입을 벌린 시간 기록
                elif last_mouth_open_time is not None:  # last_mouth_open_time 유효성 확인
                    # 3초 후 다시 입을 벌리면 모자이크 해제
                    if time.time() - last_mouth_open_time >= MOSAIC_DURATION:
                        is_mosaic_active = False  # 모자이크 비활성화
            else:
                # 입을 닫았을 때 모자이크 상태 변경
                last_mouth_open_time = None  # 입을 닫으면 타이머 초기화


            # 눈 감은 상태 체크
            if is_eye_closed(landmarks):
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                elif time.time() - eye_closed_time >= EYE_CLOSED_THRESHOLD:
                    cap.release()  # 카메라 종료
                    cv.destroyAllWindows()
                    exit()  # 프로그램 종료
            else:
                eye_closed_time = None  # 눈을 뜨면 초기화

    # 모자이크 적용
    if is_mosaic_active:
        frame = apply_mosaic(frame, landmarks)

    # 슬라이드와 카메라 화면 합치기
    frame_resized = cv.resize(frame, (800, 800))
    combined_image = cv.hconcat([frame_resized, slide_image])  # 이미지 가로로 합치기
    
    cv.imshow('Camera & Slide', combined_image)  # 합친 이미지 출력
    
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
