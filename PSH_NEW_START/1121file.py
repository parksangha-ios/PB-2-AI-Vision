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
img_files = sorted(glob.glob('./images/*.jpg'))  # 슬라이드 경로를 조정하세요
if not img_files:
    print("No slide images found.")
    exit()

current_slide = 0
total_slides = len(img_files)
slide_image = cv.imread(img_files[current_slide])
slide_image = cv.resize(slide_image, (800, 800))  # 슬라이드를 800x800으로 크기 조정

# 그림을 그리기 위한 레이어 생성
pen_layer = np.zeros_like(slide_image)  # 그리기용
highlight_layer = np.zeros_like(slide_image)  # 형광펜용

# 그리기 도구를 위한 변수들
ml = 150  # 도구의 왼쪽 여백
max_x, max_y = 250 + ml, 50
curr_tool = "laser pointer"  # 초기 도구를 'laser pointer'로 설정
time_init = True
rad = 40
var_inits = False
prevx, prevy = 0, 0

# 도구별 두께 설정
thick_pen = 4          # 펜 도구 두께
thick_highlighter = 20 # 형광펜 두께
thick_erase = 30       # 지우개 크기

# 선택 영역 관련 변수들
selection_done = False
selected_area = None
selected_mask = None
selected_pos = None
x_offset = 0
y_offset = 0

# 도구 이미지 로드
tools = cv.imread("tools1.jpg")
tools = tools.astype('uint8')
tools = cv.resize(tools, (max_x - ml, max_y))  # 슬라이드에 맞게 도구 이미지 크기 조정

def getTool(x):
    if x < 50 + ml:
        return "highlighter"  # 'line'을 'highlighter'로 변경
    elif x < 100 + ml:
        return "enlarge"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "select_move"  # 'shrink'를 'select_move'로 변경
    else:
        return "erase"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

def fingers_up(hand):
    # 검지와 중지가 펴져 있는지 확인
    y_index_tip = hand.landmark[8].y
    y_index_mcp = hand.landmark[5].y
    y_middle_tip = hand.landmark[12].y
    y_middle_mcp = hand.landmark[9].y

    if (y_index_mcp - y_index_tip) > 0.02 and (y_middle_mcp - y_middle_tip) > 0.02:
        return True
    return False

def middle_finger_folded(hand):
    # 중지가 접혀 있는지 확인
    y_middle_tip = hand.landmark[12].y
    y_middle_mcp = hand.landmark[9].y
    if (y_middle_tip - y_middle_mcp) > 0.02:
        return True
    return False

cap = cv.VideoCapture(0)
while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, (800, 800))  # 웹캠 프레임을 800x800으로 크기 조정

    rgb = cv.cvtColor(frm, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    slide_image_copy = slide_image.copy()
    # 포인터 레이어를 흰색 배경으로 초기화
    pointer_layer = np.full_like(slide_image, 255)  # 흰색 배경

    if op.multi_hand_landmarks:
        for hand in op.multi_hand_landmarks:
            # 웹캠 프레임에 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frm, hand, mp_hands.HAND_CONNECTIONS)

            x, y = int(hand.landmark[8].x * 800), int(hand.landmark[8].y * 800)  # 검지 끝부분

            xi, yi = int(hand.landmark[12].x * 800), int(hand.landmark[12].y * 800)
            y9 = int(hand.landmark[9].y * 800)

            # 손가락 상태 확인
            fingers_up_status = fingers_up(hand)
            middle_folded_status = middle_finger_folded(hand)

            # 도구 선택 영역
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv.circle(slide_image_copy, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    selected_tool = getTool(x)
                    if curr_tool == selected_tool:
                        curr_tool = "laser pointer"
                        print("Tool deactivated. Switched back to laser pointer.")
                    else:
                        curr_tool = selected_tool
                        print("Your current tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            # 현재 선택된 도구에 따른 포인터 시각화
            if curr_tool == "laser pointer":
                cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원
            elif curr_tool == "erase":
                size = thick_erase  # 지우개 크기 사용
                cv.rectangle(pointer_layer, (x - size // 2, y - size // 2),
                             (x + size // 2, y + size // 2), (0, 0, 0), -1)  # 검은색 사각형
            elif curr_tool == "highlighter":
                cv.circle(pointer_layer, (x, y), thick_highlighter // 2, (0, 255, 255), 2)  # 노란색 원 테두리
            elif curr_tool == "draw":
                cv.circle(pointer_layer, (x, y), thick_pen // 2, (0, 0, 255), 2)  # 빨간색 원 테두리
            elif curr_tool == "enlarge" or curr_tool == "select_move":
                size_rect = 10  # 사각형 크기
                cv.rectangle(pointer_layer, (x - size_rect // 2, y - size_rect // 2),
                             (x + size_rect // 2, y + size_rect // 2), (0, 255, 255), 2)  # 노란색 사각형 테두리

            # 그리기 도구 구현
            if curr_tool == "draw":
                if index_raised(yi, y9):
                    if prevx != 0 and prevy != 0:
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

            elif curr_tool == "enlarge":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv.rectangle(slide_image_copy, (xii, yii), (x, y), (0, 255, 255), 2)
                else:
                    if var_inits:
                        # slide_image에서 선택된 영역 추출
                        x_start, x_end = min(xii, x), max(xii, x)
                        y_start, y_end = min(yii, y), max(yii, y)
                        selected_area = slide_image[y_start:y_end, x_start:x_end]

                        if selected_area.size != 0:
                            # 선택된 영역 확대
                            enlarged_area = cv.resize(selected_area, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)

                            # 'Additional Materials' 창에 확대된 영역 표시
                            cv.imshow("Additional Materials", enlarged_area)

                        var_inits = False

            elif curr_tool == "select_move":
                if not selection_done:
                    # 선택 모드
                    if fingers_up_status:
                        # 준비 상태에서 영역 선택
                        if not var_inits:
                            x1, y1 = x, y
                            var_inits = True
                        # 점선 사각형 그리기
                        rect_start = (x1, y1)
                        rect_end = (x, y)
                        # 점선 사각형을 그리기 위한 코드
                        thickness = 2
                        line_type = cv.LINE_AA
                        color = (255, 255, 0)
                        dash_length = 10
                        x_min, x_max = min(x1, x), max(x1, x)
                        y_min, y_max = min(y1, y), max(y1, y)
                        for i in range(x_min, x_max, dash_length * 2):
                            cv.line(slide_image_copy, (i, y_min), (i + dash_length, y_min), color, thickness, line_type)
                            cv.line(slide_image_copy, (i, y_max), (i + dash_length, y_max), color, thickness, line_type)
                        for i in range(y_min, y_max, dash_length * 2):
                            cv.line(slide_image_copy, (x_min, i), (x_min, i + dash_length), color, thickness, line_type)
                            cv.line(slide_image_copy, (x_max, i), (x_max, i + dash_length), color, thickness, line_type)
                    elif middle_folded_status:
                        # 중지를 접으면 선택 확정
                        if var_inits:
                            # pen_layer에서 선택된 영역 추출
                            x_start, x_end = min(x1, x), max(x1, x)
                            y_start, y_end = min(y1, y), max(y1, y)
                            selected_area = pen_layer[y_start:y_end, x_start:x_end].copy()
                            selected_mask = np.zeros_like(pen_layer[:, :, 0])
                            gray_selected_area = cv.cvtColor(selected_area, cv.COLOR_BGR2GRAY)
                            _, selected_mask = cv.threshold(gray_selected_area, 10, 255, cv.THRESH_BINARY)
                            x_offset = x_start
                            y_offset = y_start
                            selection_done = True
                            var_inits = False
                            print("Selection completed. Switched to move mode.")
                            # 원본 위치의 그림 지우기
                            pen_layer[y_start:y_end, x_start:x_end] = 0
                    else:
                        pass  # 아무 동작 없음
                else:
                    # 이동 모드
                    if middle_folded_status:
                        # 중지를 접은 상태로 이동
                        x_offset = x - selected_area.shape[1] // 2
                        y_offset = y - selected_area.shape[0] // 2
                    elif fingers_up_status:
                        # 중지를 펴면 위치 확정
                        if selected_area is not None and selected_mask is not None:
                            # 새로운 위치에 선택 영역을 pen_layer에 합성
                            h, w = selected_area.shape[:2]
                            x_end = x_offset + w
                            y_end = y_offset + h

                            # 경계 체크
                            x1_dst = max(0, x_offset)
                            y1_dst = max(0, y_offset)
                            x2_dst = min(800, x_end)
                            y2_dst = min(800, y_end)

                            x1_src = max(0, -x_offset)
                            y1_src = max(0, -y_offset)
                            x2_src = x1_src + (x2_dst - x1_dst)
                            y2_src = y1_src + (y2_dst - y1_dst)

                            selected_area_moved = np.zeros_like(pen_layer)
                            selected_area_moved[y1_dst:y2_dst, x1_dst:x2_dst] = selected_area[y1_src:y2_src, x1_src:x2_src]

                            pen_layer = cv.bitwise_or(pen_layer, selected_area_moved)
                            # 초기화
                            selected_area = None
                            selected_mask = None
                            selection_done = False
                            var_inits = False
                            print("Position confirmed.")
                    else:
                        pass  # 아무 동작 없음

            elif curr_tool == "erase":
                if index_raised(yi, y9):
                    cv.circle(pen_layer, (x, y), thick_erase, (0, 0, 0), -1)
                    cv.circle(highlight_layer, (x, y), thick_erase, (0, 0, 0), -1)  # 형광펜도 지우기

        # 선택된 영역 이동 및 합성
        if selection_done and selected_area is not None:
            # 이동된 위치의 마스크와 영역 생성
            h, w = selected_area.shape[:2]
            x_end = x_offset + w
            y_end = y_offset + h

            # 경계 체크
            x1_dst = max(0, x_offset)
            y1_dst = max(0, y_offset)
            x2_dst = min(800, x_end)
            y2_dst = min(800, y_end)

            x1_src = max(0, -x_offset)
            y1_src = max(0, -y_offset)
            x2_src = x1_src + (x2_dst - x1_dst)
            y2_src = y1_src + (y2_dst - y1_dst)

            selected_area_moved = np.zeros_like(pen_layer)
            selected_area_moved[y1_dst:y2_dst, x1_dst:x2_dst] = selected_area[y1_src:y2_src, x1_src:x2_src]

            # 이동 중인 선택 영역을 slide_image_copy에 합성
            selected_mask_moved = np.zeros_like(pen_layer[:, :, 0])
            gray_selected_area = cv.cvtColor(selected_area_moved[y1_dst:y2_dst, x1_dst:x2_dst], cv.COLOR_BGR2GRAY)
            _, selected_mask_moved[y1_dst:y2_dst, x1_dst:x2_dst] = cv.threshold(
                gray_selected_area, 10, 255, cv.THRESH_BINARY)
            cv.copyTo(src=selected_area_moved, dst=slide_image_copy, mask=selected_mask_moved)

            # 선택 영역 주위에 점선 사각형 그리기
            thickness = 2
            line_type = cv.LINE_AA
            color = (0, 255, 0)
            dash_length = 10
            x_min, x_max = x1_dst, x2_dst
            y_min, y_max = y1_dst, y2_dst
            for i in range(x_min, x_max, dash_length * 2):
                cv.line(slide_image_copy, (i, y_min), (i + dash_length, y_min), color, thickness, line_type)
                cv.line(slide_image_copy, (i, y_max), (i + dash_length, y_max), color, thickness, line_type)
            for i in range(y_min, y_max, dash_length * 2):
                cv.line(slide_image_copy, (x_min, i), (x_min, i + dash_length), color, thickness, line_type)
                cv.line(slide_image_copy, (x_max, i), (x_max, i + dash_length), color, thickness, line_type)

    # 슬라이드 이미지와 pen_layer, highlight_layer를 결합 (흰색 배경 제외)
    # pen_layer에서 그림이 있는 곳의 마스크 생성
    pen_gray = cv.cvtColor(pen_layer, cv.COLOR_BGR2GRAY)
    _, mask_pen = cv.threshold(pen_gray, 10, 255, cv.THRESH_BINARY)

    # highlight_layer에서 형광펜의 마스크 생성
    highlight_gray = cv.cvtColor(highlight_layer, cv.COLOR_BGR2GRAY)
    _, mask_highlight = cv.threshold(highlight_gray, 10, 255, cv.THRESH_BINARY)

    # pointer_layer에서 포인터의 마스크 생성 (흰색 배경 제외)
    pointer_gray = cv.cvtColor(pointer_layer, cv.COLOR_BGR2GRAY)
    _, mask_pointer = cv.threshold(pointer_gray, 254, 255, cv.THRESH_BINARY_INV)

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

    # 강의 슬라이드와 주석을 표시
    cv.imshow("Lecture", slide_image_copy)
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

cap.release()
cv.destroyAllWindows()
