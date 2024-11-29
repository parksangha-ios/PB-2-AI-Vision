import cv2 as cv
import mediapipe as mp
import glob
import numpy as np
import time

# Mediapipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hand_landmark = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2
)

# 슬라이드 이미지 로드
img_files = sorted(glob.glob('team2/OJH/lecture.png'))  # 슬라이드 경로를 조정하세요
if not img_files:
    print("No slide images found.")
    exit()

current_slide = 0
total_slides = len(img_files)
slide_image = cv.imread(img_files[current_slide])
slide_image = cv.resize(slide_image, (800, 800))  # 슬라이드를 800x800으로 크기 조정

# 그림을 그리기 위한 레이어 생성
pen_layer = np.zeros_like(slide_image)  # 그리기용
pointer_layer = np.zeros_like(slide_image)  # 레이저 포인터용
highlight_layer = np.zeros_like(slide_image)  # 형광펜용

# 그리기 도구를 위한 변수들
ml = 150  # 도구의 왼쪽 여백
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
prevx, prevy = 0, 0

# 도구별 두께 설정
thick_pen = 4          # 펜 도구 두께
thick_highlighter = 20 # 형광펜 두께
thick_erase = 30       # 지우개 크기

# 도구 이미지 로드
tools = cv.imread("team2/tools.png")
tools = tools.astype('uint8')
tools = cv.resize(tools, (max_x - ml, max_y))  # 슬라이드에 맞게 도구 이미지 크기 조정

note_layer = np.zeros((200, 800, 3), dtype=np.uint8)  # 핵심노트 레이어 (높이 200픽셀)
is_note_area = False

current_cropped_image = None  # 현재 선택된 크롭 이미지
rotation_angle = 0  # 현재 회전 각도

def detect_thumb_direction(hand_landmarks):
    # 엄지손가락의 방향 감지
    thumb_tip = hand_landmarks.landmark[4]  # 엄지손가락 끝
    thumb_base = hand_landmarks.landmark[2]  # 엄지손가락 둘째 마디
    
    # x 좌표 차이로 방향 판단
    diff_x = thumb_tip.x - thumb_base.x
    
    if diff_x > 0.04:  # 임계값 설정
        return "right"
    elif diff_x < -0.04:
        return "left"
    return "neutral"

def getTool(x):
    if x < 50 + ml:
        return "highlighter"  # 'line'을 'highlighter'로 변경
    elif x < 100 + ml:
        return "crop"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "shrink"
    else:
        return "erase"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

cap = cv.VideoCapture(0)
cropped_images = []

start_x, start_y = 0, 0  # 시작점 좌표
drawing = False          # 그리기 시작 여부
while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, (800, 800))   # 웹캠 프레임을 800x800으로 크기 조정

    rgb = cv.cvtColor(frm, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    slide_image_copy = slide_image.copy()
    pointer_layer = np.zeros_like(slide_image)  # 레이저 포인터 레이어 초기화

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            # 웹캠 프레임에 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frm, i, mp_hands.HAND_CONNECTIONS)

            x, y = int(i.landmark[8].x * 800), int(i.landmark[8].y * 800)  # 검지 끝부분

            # 레이저 포인터 기능: 슬라이드에 손가락 위치 표시
            cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원

            # 도구 선택 영역
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv.circle(slide_image_copy, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Your current tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40
            
            x, y = int(i.landmark[8].x * 800), int(i.landmark[8].y * 800)
            
            # 핵심노트 영역 확인 (화면 하단 200픽셀)
            is_note_area = y >= 600  # 800(전체 높이) - 200(노트 높이)
            
            xi, yi = int(i.landmark[12].x * 800), int(i.landmark[12].y * 800)
            y9 = int(i.landmark[9].y * 800)

            if curr_tool == "draw":
                if index_raised(yi, y9):
                    if prevx != 0 and prevy != 0:
                        if is_note_area:
                            # 노트 영역에 그리기
                            note_y = y - 600  # 노트 레이어의 상대 좌표로 변환
                            prev_note_y = prevy - 600
                            cv.line(note_layer, (prevx, prev_note_y), (x, note_y), (0, 0, 255), thick_pen)
                        else:
                            # 기존 슬라이드 영역에 그리기
                            cv.line(pen_layer, (prevx, prevy), (x, y), (0, 0, 255), thick_pen)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = 0, 0
            
            elif curr_tool == "highlighter":
                if index_raised(yi, y9):
                    if prevx != 0 and prevy != 0:
                        # 형광펜 레이어에 선 그리기
                        cv.line(highlight_layer, (prevx, prevy), (x, y), (0, 255, 255), thick_highlighter)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = 0, 0
            
            # crop 도구 부분 수정
            elif curr_tool == "crop":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv.rectangle(slide_image_copy, (xii, yii), (x, y), (0, 255, 255), 2)
                else:
                    if var_inits:
                        x_start, x_end = min(xii, x), max(xii, x)
                        y_start, y_end = min(yii, y), max(yii, y)
                        selected_area = slide_image[y_start:y_end, x_start:x_end]

                        if selected_area.size != 0:
                            current_cropped_image = selected_area.copy()  # 현재 크롭된 이미지 저장
                            rotation_angle = 0  # 회전 각도 초기화
                            cropped_images.append(current_cropped_image)
                            cv.imshow("Additional Materials", current_cropped_image)
                        var_inits = False

                # # 회전 제스처 감지
                # if current_cropped_image is not None:
                #     thumb_direction = detect_thumb_direction(i)
                    
                #     if thumb_direction == "right":
                #         # 시계 방향 90도 회전
                #         rotation_angle = (rotation_angle + 90) % 360
                #         matrix = cv.getRotationMatrix2D((current_cropped_image.shape[1]/2, 
                #                                     current_cropped_image.shape[0]/2), 
                #                                     -rotation_angle, 1.0)
                #         rotated = cv.warpAffine(current_cropped_image, matrix, 
                #                             (current_cropped_image.shape[1], 
                #                             current_cropped_image.shape[0]))
                #         cv.imshow("Additional Materials", rotated)
                #         # 마지막 크롭 이미지 업데이트
                #         cropped_images[-1] = rotated
                #         time.sleep(0.5)  # 회전 간 딜레이
                        
                #     elif thumb_direction == "left":
                #         # 반시계 방향 90도 회전
                #         rotation_angle = (rotation_angle - 90) % 360
                #         matrix = cv.getRotationMatrix2D((current_cropped_image.shape[1]/2, 
                #                                     current_cropped_image.shape[0]/2), 
                #                                     -rotation_angle, 1.0)
                #         rotated = cv.warpAffine(current_cropped_image, matrix, 
                #                             (current_cropped_image.shape[1], 
                #                             current_cropped_image.shape[0]))
                #         cv.imshow("Additional Materials", rotated)
                #         # 마지막 크롭 이미지 업데이트
                #         cropped_images[-1] = rotated
                #         time.sleep(0.5)  # 회전 간 딜레이

            elif curr_tool == "shrink":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv.rectangle(slide_image_copy, (xii, yii), (x, y), (255, 255, 0), 2)
                else:
                    if var_inits:
                        # slide_image에서 선택된 영역 추출
                        x_start, x_end = min(xii, x), max(xii, x)
                        y_start, y_end = min(yii, y), max(yii, y)
                        selected_area = slide_image[y_start:y_end, x_start:x_end]

                        if selected_area.size != 0:
                            # 선택된 영역 축소
                            shrunk_area = cv.resize(selected_area, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

                            # 'Additional Materials' 창에 축소된 영역 표시
                            cv.imshow("Additional Materials", shrunk_area)

                        var_inits = False

            elif curr_tool == "erase":
                if index_raised(yi, y9):
                    cv.circle(pen_layer, (x, y), thick_erase, (0, 0, 0), -1)
                    cv.circle(highlight_layer, (x, y), thick_erase, (0, 0, 0), -1)  # 형광펜도 지우기

    # 슬라이드 이미지와 pen_layer, highlight_layer를 결합 (흰색 배경 제외)
    # pen_layer에서 그림이 있는 곳의 마스크 생성
    pen_gray = cv.cvtColor(pen_layer, cv.COLOR_BGR2GRAY)
    _, mask_pen = cv.threshold(pen_gray, 10, 255, cv.THRESH_BINARY)

    # highlight_layer에서 형광펜의 마스크 생성
    highlight_gray = cv.cvtColor(highlight_layer, cv.COLOR_BGR2GRAY)
    _, mask_highlight = cv.threshold(highlight_gray, 10, 255, cv.THRESH_BINARY)

    # pointer_layer에서 레이저 포인터의 마스크 생성
    pointer_gray = cv.cvtColor(pointer_layer, cv.COLOR_BGR2GRAY)
    _, mask_pointer = cv.threshold(pointer_gray, 10, 255, cv.THRESH_BINARY)

    # 마스크가 0이 아닌 곳에 pen_layer를 slide_image_copy에 복사
    cv.copyTo(src=pen_layer, dst=slide_image_copy, mask=mask_pen)

    # 형광펜 레이어를 slide_image_copy에 반투명하게 합성
    # 투명도(alpha)를 0.4로 설정하여 형광펜 효과 적용
    alpha = 0.4  # 투명도 조절 (0.0 투명 ~ 1.0 불투명)
    highlight_color = (0, 255, 255)  # 형광펜 색상 (노란색)
    highlight_indices = np.where(mask_highlight == 255)
    for c in range(3):  # B, G, R 채널 각각에 대해
        slide_image_copy[highlight_indices[0], highlight_indices[1], c] = (
            slide_image_copy[highlight_indices[0], highlight_indices[1], c] * (1 - alpha) +
            highlight_color[c] * alpha
        )

    # 마스크가 0이 아닌 곳에 pointer_layer를 slide_image_copy에 복사
    cv.copyTo(src=pointer_layer, dst=slide_image_copy, mask=mask_pointer)

    # 슬라이드 이미지에 도구 오버레이
    slide_image_copy[:max_y, ml:max_x] = cv.addWeighted(
        tools, 0.7, slide_image_copy[:max_y, ml:max_x], 0.3, 0
    )

    cv.putText(
        slide_image_copy,
        curr_tool,
        (270 + ml, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        )
    # 슬라이드와 노트 결합
    combined_image = np.vstack([slide_image_copy, note_layer])
    cv.imshow("Lecture", combined_image)

    # 웹캠 피드 표시
    cv.imshow("Webcam", frm)

    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break
    elif key == ord('u'):  # 다음 슬라이드
        if current_slide < total_slides - 1:
            current_slide += 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, (800, 800))
            pen_layer = np.zeros_like(slide_image)
            highlight_layer = np.zeros_like(slide_image)  # 형광펜 레이어 초기화
    elif key == ord('d'):  # 이전 슬라이드
        if current_slide > 0:
            current_slide -= 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, (800, 800))
            pen_layer = np.zeros_like(slide_image)
            highlight_layer = np.zeros_like(slide_image)  # 형광펜 레이어 초기화
    elif key == ord('s'):  # 's' 키를 누르면 Summary 창 표시
        if cropped_images:
            target_height = max(img.shape[0] for img in cropped_images)
            resized_images = []
            
            for img in cropped_images:
                # 비율 유지하면서 높이 조정
                aspect_ratio = img.shape[1] / img.shape[0]
                target_width = int(target_height * aspect_ratio)
                resized = cv.resize(img, (target_width, target_height))
                resized_images.append(resized)
            
            # 전체 너비 계산
            total_width = sum(img.shape[1] for img in resized_images)
            
            # Summary 이미지 생성
            summary_image = np.zeros((target_height, total_width, 3), dtype=np.uint8)
            
            # 이미지들을 가로로 붙이기
            current_x = 0
            for img in resized_images:
                width = img.shape[1]
                summary_image[0:target_height, current_x:current_x + width] = img
                current_x += width
            
            # Summary 창 표시
            cv.imshow("Summary", summary_image)
            
    # 키 입력 처리 부분에 추가
    elif key == ord('c'):  # 'c' 키를 누르면 핵심노트 전체 지우기
        note_layer = np.zeros((200, 800, 3), dtype=np.uint8)
        
cap.release()
cv.destroyAllWindows()
