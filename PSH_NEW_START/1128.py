import cv2 as cv
import mediapipe as mp
import glob
import numpy as np
import time
import math  # 거리 계산을 위해 추가

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

# 그림을 그리기 위한 레이어 생성 (알파 채널 포함)
pen_layer = np.zeros((800, 800, 4), dtype=np.uint8)          # 그리기용
highlight_layer = np.zeros((800, 800, 4), dtype=np.uint8)    # 형광펜용

# 그리기 도구를 위한 변수들
ml = 150  # 도구의 왼쪽 여백
max_x, max_y = 250 + ml, 50
curr_tool = "laser pointer"  # 초기 도구를 'laser pointer'로 설정
time_init = True
rad = 40
var_inits = False
prevx, prevy = 0, 0

ml2 = 600  # 도구의 오른쪽 여백
max_x2, max_y2 = 800, 50

# 도구별 두께 설정
thick_pen = 4          # 펜 도구 두께
thick_highlighter = 20 # 형광펜 두께
thick_erase = 30       # 지우개 크기

# 선택 영역 관련 변수들 (select_move 기능 추가)
selection_done = False
selected_area_pen = None        # pen_layer에서 선택된 영역
selected_area_highlight = None  # highlight_layer에서 선택된 영역
x_offset = 0
y_offset = 0
x_start = y_start = x_end = y_end = 0  # 선택 영역의 시작과 끝 좌표를 전역 변수로 선언
selection_cooldown = 0  # 선택 쿨다운 초기화

# 도구 이미지 로드
tools = cv.imread("tools1.jpg")
tools = tools.astype('uint8')
tools = cv.resize(tools, (max_x - ml, max_y))  # 슬라이드에 맞게 도구 이미지 크기 조정

def index_below_middle(yi_middle, y9, xi, x8, yi, y8):
    # 검지와 중지가 붙어 있는지 확인
    distance = np.sqrt((x8 - xi) ** 2 + (y8 - yi) ** 2)
    fingers_together = distance < 40

    # 중지와 검지가 손바닥 중심보다 아래로 향했는지 확인
    below_condition = (yi_middle > y9) and (yi > y9) and (y8 > y9)

    return fingers_together and below_condition

def index_horizontal(yi, y9, xi, x8, yi_middle, y8):
    # 검지와 중지가 붙어 있는지 확인
    distance = np.sqrt((x8 - xi) ** 2 + (y8 - yi) ** 2)
    fingers_together = distance < 40

    # 검지와 중지가 손바닥 중심 높이와 가까운지 확인
    horizontal_condition = abs(yi - y9) < 50 and abs(y8 - y9) < 50 

    return fingers_together and horizontal_condition

def getTool(x):
    if x < 50 + ml:
        return "highlighter"
    elif x < 100 + ml:
        return "enlarge"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "select_move"
    else:
        return "erase"
    
COLOR_SECTIONS = {
    "red": (600, 640),
    "yellow": (640, 680),
    "green": (680, 720),
    "blue": (720, 760),
    "black": (760, 800),
}   

def getTool2(x):
    for color, (start, end) in COLOR_SECTIONS.items():
        if start <= x < end:
            return color
    return None

selected_color = (0, 0, 255)  # 초기 색상 (빨간색)

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

# 손가락이 펴져 있는지 확인하는 함수
def fingers_up(hand):
    # 검지와 중지가 펴져 있는지 확인
    y_index_tip = hand.landmark[8].y
    y_index_mcp = hand.landmark[5].y
    y_middle_tip = hand.landmark[12].y
    y_middle_mcp = hand.landmark[9].y

    if (y_index_mcp - y_index_tip) > 0.02 and (y_middle_mcp - y_middle_tip) > 0.02:
        return True
    return False

# 중지가 접혀 있는지 확인하는 함수
def middle_finger_folded(hand):
    y_middle_tip = hand.landmark[12].y
    y_middle_mcp = hand.landmark[9].y
    if (y_middle_tip - y_middle_mcp) > 0.02:
        return True
    return False

# 주먹 동작 인식 함수 추가
def is_fist(hand):
    fingers_folded = 0
    # 검지, 중지, 약지, 새끼손가락
    for id in [8, 12, 16, 20]:
        if hand.landmark[id].y > hand.landmark[id - 2].y:
            fingers_folded += 1
    # 엄지손가락은 옆으로 접혀 있는지 확인
    if abs(hand.landmark[4].x - hand.landmark[3].x) < 0.02:
        fingers_folded += 1
    if fingers_folded >= 4:
        return True
    else:
        return False

cap = cv.VideoCapture(0)
cap.release()
cap = cv.VideoCapture(0)

# 모드 및 타이머 변수 초기화
mode = 'lecture'  # 현재 모드: 'lecture' 또는 'break'
timer_running = False
start_time = None
lecture_duration = 10  # 강의 모드에서의 타이머 시간 (초)
break_duration = 10     # 쉬는 시간 모드 시간 (초)
display_time = lecture_duration  # display_time 초기화

while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, (800, 800))

    rgb = cv.cvtColor(frm, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    # 현재 모드에 따라 슬라이드 이미지 설정
    if mode == 'lecture':
        slide_image_current = slide_image.copy()
    elif mode == 'break':
        slide_image_current = cv.imread("breaktime.png")
        slide_image_current = cv.resize(slide_image_current, (800, 800))

    slide_image_copy = slide_image_current.copy()
    pointer_layer = np.full_like(slide_image_current, 255)

    # 타이머 실행 및 모드 전환
    if mode == 'lecture':
        if timer_running:
            elapsed_time = time.time() - start_time
            display_time = int(lecture_duration - elapsed_time)
            if elapsed_time >= lecture_duration:
                mode = 'break'
                timer_running = True
                start_time = time.time()
                print("Lecture time over. Switching to break mode.")
        else:
            display_time = lecture_duration  # 타이머가 시작되지 않았을 때는 전체 시간을 표시
    elif mode == 'break':
        elapsed_time = time.time() - start_time
        display_time = int(break_duration - elapsed_time)
        if elapsed_time >= break_duration:
            mode = 'lecture'
            timer_running = False
            start_time = None
            display_time = lecture_duration
            print("Break time over. Back to lecture mode.")

    # 타이머가 음수가 되지 않도록 보정
    display_time = max(display_time, 0)

    # 손 동작 인식 및 도구 기능 구현
    if op.multi_hand_landmarks:
        for hand in op.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frm, hand, mp_hands.HAND_CONNECTIONS)
            x, y = int(hand.landmark[8].x * 800), int(hand.landmark[8].y * 800)
            xi, yi = int(hand.landmark[12].x * 800), int(hand.landmark[12].y * 800)
            y9 = int(hand.landmark[9].y * 800)
            yi_middle = yi
            y8 = y
            x8 = x

            fingers_up_status = fingers_up(hand)
            middle_folded_status = middle_finger_folded(hand)
            fist_status = is_fist(hand)

            if mode == 'lecture':
                # 주먹 동작 인식하여 타이머 시작
                if fist_status and not timer_running:
                    start_time = time.time()
                    timer_running = True
                    print("Lecture timer started.")

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
                # 색깔 선택
                if 600 <= x <= 800 and y <= max_y2:
                    selected_tool = getTool2(x)
                    if selected_tool:
                        # 색상 매핑: 빨강, 노랑, 초록, 파랑, 검정
                        color_map = {
                            "red": (0, 0, 255),
                            "yellow": (0, 255, 255),
                            "green": (0, 255, 0),
                            "blue": (255, 0, 0),
                            "black": (0, 0, 0),
                        }
                        selected_color = color_map[selected_tool]
                        curr_tool = "color selection"
                    cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원
                # 현재 선택된 도구에 따른 포인터 시각화
                if curr_tool == "laser pointer" or curr_tool == "color selection":
                    cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원
                elif curr_tool == "erase":
                    size = thick_erase
                    cv.rectangle(pointer_layer, (x - size // 2, y - size // 2),
                                 (x + size // 2, y + size // 2), (0, 0, 0), -1)
                elif curr_tool == "highlighter":
                    cv.circle(pointer_layer, (x, y), thick_highlighter // 2, (0, 255, 255), 2)
                elif curr_tool == "draw":
                    cv.circle(pointer_layer, (x, y), thick_pen // 2, (0, 0, 255), 2)
                elif curr_tool == "enlarge" or curr_tool == "select_move":
                    size_rect = 10
                    cv.rectangle(pointer_layer, (x - size_rect // 2, y - size_rect // 2),
                                 (x + size_rect // 2, y + size_rect // 2), (0, 255, 255), 2)

                # 그리기 도구 구현
                if curr_tool == "draw":
                    if index_raised(yi, y9) or index_below_middle(yi_middle, y9, xi, x8, yi, y8) or index_horizontal(yi, y9, xi, x8, yi_middle, y8):
                        if prevx != 0 and prevy != 0:
                            color_with_alpha = selected_color + (255,)  # 알파 채널 추가
                            cv.line(pen_layer, (prevx, prevy), (x, y), color_with_alpha, thick_pen)
                        prevx, prevy = x, y
                    else:
                        prevx, prevy = 0, 0

                elif curr_tool == "highlighter":
                    if index_raised(yi, y9) or index_below_middle(yi_middle, y9, xi, x8, yi, y8) or index_horizontal(yi, y9, xi, x8, yi_middle, y8):
                        if prevx != 0 and prevy != 0:
                            color_with_alpha = selected_color + (int(255 * 0.4),)  # 알파 채널 추가 (투명도 적용)
                            cv.line(highlight_layer, (prevx, prevy), (x, y), color_with_alpha, thick_highlighter)
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
                            x_start, x_end = min(xii, x), max(xii, x)
                            y_start, y_end = min(yii, y), max(yii, y)
                            selected_area_img = slide_image[y_start:y_end, x_start:x_end]

                            if selected_area_img.size != 0:
                                enlarged_area = cv.resize(selected_area_img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
                                cv.imshow("Additional Materials", enlarged_area)

                            var_inits = False

                elif curr_tool == "select_move":
                    if not selection_done:
                        # 선택 모드
                        if time.time() > selection_cooldown:
                            if fingers_up_status:
                                # 영역 선택 시작
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
                                # 선택 확정
                                if var_inits:
                                    x_start, x_end = min(x1, x), max(x1, x)
                                    y_start, y_end = min(y1, y), max(y1, y)
                                    # pen_layer에서 선택된 영역 추출
                                    selected_area_pen = pen_layer[y_start:y_end, x_start:x_end].copy()
                                    # highlight_layer에서 선택된 영역 추출
                                    selected_area_highlight = highlight_layer[y_start:y_end, x_start:x_end].copy()
                                    x_offset = x_start
                                    y_offset = y_start
                                    selection_done = True
                                    var_inits = False
                                    print("Selection completed. Switched to move mode.")
                                    # 원본 위치의 그림 지우기
                                    pen_layer[y_start:y_end, x_start:x_end] = 0
                                    highlight_layer[y_start:y_end, x_start:x_end] = 0
                            else:
                                pass
                        else:
                            pass
                    else:
                        # 이동 모드
                        if middle_folded_status:
                            # 중지를 접은 상태로 이동
                            # 선택 영역의 크기를 계산
                            w = selected_area_pen.shape[1] if selected_area_pen is not None else selected_area_highlight.shape[1]
                            h = selected_area_pen.shape[0] if selected_area_pen is not None else selected_area_highlight.shape[0]

                            x_offset = x - w // 2
                            y_offset = y - h // 2
                        elif fingers_up_status:
                            # 위치 확정
                            # pen_layer에 합성
                            if selected_area_pen is not None:
                                h_pen, w_pen = selected_area_pen.shape[:2]
                                x1_dst_pen = max(0, x_offset)
                                y1_dst_pen = max(0, y_offset)
                                x2_dst_pen = min(800, x_offset + w_pen)
                                y2_dst_pen = min(800, y_offset + h_pen)

                                x1_src_pen = x1_dst_pen - x_offset
                                y1_src_pen = y1_dst_pen - y_offset
                                x2_src_pen = x1_src_pen + (x2_dst_pen - x1_dst_pen)
                                y2_src_pen = y1_src_pen + (y2_dst_pen - y1_dst_pen)

                                # 이동된 선택 영역 추출
                                selected_area_moved_pen = selected_area_pen[y1_src_pen:y2_src_pen, x1_src_pen:x2_src_pen]

                                # 채널 분리
                                b_pen, g_pen, r_pen, a_pen = cv.split(selected_area_moved_pen)
                                overlay_color_pen = cv.merge((b_pen, g_pen, r_pen))
                                mask_pen = a_pen

                                # ROI 정의
                                roi_pen = pen_layer[y1_dst_pen:y2_dst_pen, x1_dst_pen:x2_dst_pen]

                                # 알파 채널을 마스크로 사용하여 합성
                                cv.copyTo(selected_area_moved_pen, mask_pen, roi_pen)

                            # highlight_layer에 합성
                            if selected_area_highlight is not None:
                                h_highlight, w_highlight = selected_area_highlight.shape[:2]
                                x1_dst_highlight = max(0, x_offset)
                                y1_dst_highlight = max(0, y_offset)
                                x2_dst_highlight = min(800, x_offset + w_highlight)
                                y2_dst_highlight = min(800, y_offset + h_highlight)

                                x1_src_highlight = x1_dst_highlight - x_offset
                                y1_src_highlight = y1_dst_highlight - y_offset
                                x2_src_highlight = x1_src_highlight + (x2_dst_highlight - x1_dst_highlight)
                                y2_src_highlight = y1_src_highlight + (y2_dst_highlight - y1_dst_highlight)

                                # 이동된 선택 영역 추출
                                selected_area_moved_highlight = selected_area_highlight[y1_src_highlight:y2_src_highlight, x1_src_highlight:x2_src_highlight]

                                # 채널 분리
                                b_highlight, g_highlight, r_highlight, a_highlight = cv.split(selected_area_moved_highlight)
                                overlay_color_highlight = cv.merge((b_highlight, g_highlight, r_highlight))
                                mask_highlight = a_highlight

                                # ROI 정의
                                roi_highlight = highlight_layer[y1_dst_highlight:y2_dst_highlight, x1_dst_highlight:x2_dst_highlight]

                                # 알파 채널을 마스크로 사용하여 합성
                                cv.copyTo(selected_area_moved_highlight, mask_highlight, roi_highlight)

                            # 초기화
                            selected_area_pen = None
                            selected_area_highlight = None
                            selection_done = False
                            var_inits = False
                            selection_cooldown = time.time() + 0.5
                            print("Position confirmed.")
                        else:
                            pass

                elif curr_tool == "erase":
                    if index_raised(yi, y9) or index_below_middle(yi_middle, y9, xi, x8, yi, y8) or index_horizontal(yi, y9, xi, x8, yi_middle, y8):
                        cv.circle(pen_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)
                        cv.circle(highlight_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)

    # 선택된 영역 이동 및 합성
    if selection_done and curr_tool == "select_move":
        # pen_layer의 선택 영역 이동 및 합성
        if selected_area_pen is not None:
            h_pen, w_pen = selected_area_pen.shape[:2]
            x1_dst_pen = max(0, x_offset)
            y1_dst_pen = max(0, y_offset)
            x2_dst_pen = min(800, x_offset + w_pen)
            y2_dst_pen = min(800, y_offset + h_pen)

            x1_src_pen = x1_dst_pen - x_offset
            y1_src_pen = y1_dst_pen - y_offset
            x2_src_pen = x1_src_pen + (x2_dst_pen - x1_dst_pen)
            y2_src_pen = y1_src_pen + (y2_dst_pen - y1_dst_pen)

            # 이동된 선택 영역 추출
            selected_area_moved_pen = selected_area_pen[y1_src_pen:y2_src_pen, x1_src_pen:x2_src_pen]

            # 채널 분리
            b_pen, g_pen, r_pen, a_pen = cv.split(selected_area_moved_pen)
            overlay_color_pen = cv.merge((b_pen, g_pen, r_pen))
            mask_pen = a_pen

            # ROI 정의
            roi_pen = slide_image_copy[y1_dst_pen:y2_dst_pen, x1_dst_pen:x2_dst_pen]

            # 알파 채널을 마스크로 사용하여 합성
            cv.copyTo(overlay_color_pen, mask_pen, roi_pen)

        # highlight_layer의 선택 영역 이동 및 합성
        if selected_area_highlight is not None:
            h_highlight, w_highlight = selected_area_highlight.shape[:2]
            x1_dst_highlight = max(0, x_offset)
            y1_dst_highlight = max(0, y_offset)
            x2_dst_highlight = min(800, x_offset + w_highlight)
            y2_dst_highlight = min(800, y_offset + h_highlight)

            x1_src_highlight = x1_dst_highlight - x_offset
            y1_src_highlight = y1_dst_highlight - y_offset
            x2_src_highlight = x1_src_highlight + (x2_dst_highlight - x1_dst_highlight)
            y2_src_highlight = y1_src_highlight + (y2_dst_highlight - y1_dst_highlight)

            # 이동된 선택 영역 추출
            selected_area_moved_highlight = selected_area_highlight[y1_src_highlight:y2_src_highlight, x1_src_highlight:x2_src_highlight]

            # 채널 분리
            b_highlight, g_highlight, r_highlight, a_highlight = cv.split(selected_area_moved_highlight)
            overlay_color_highlight = cv.merge((b_highlight, g_highlight, r_highlight))
            mask_highlight = a_highlight

            # ROI 정의
            roi_highlight = slide_image_copy[y1_dst_highlight:y2_dst_highlight, x1_dst_highlight:x2_dst_highlight]

            # 알파 채널을 마스크로 사용하여 합성
            cv.copyTo(overlay_color_highlight, mask_highlight, roi_highlight)

        # 선택 영역 주위에 점선 사각형 그리기
        thickness = 2
        line_type = cv.LINE_AA
        color = (0, 255, 0)
        dash_length = 10

        w = selected_area_pen.shape[1] if selected_area_pen is not None else selected_area_highlight.shape[1]
        h = selected_area_pen.shape[0] if selected_area_pen is not None else selected_area_highlight.shape[0]

        x_min, x_max = x_offset, x_offset + w
        y_min, y_max = y_offset, y_offset + h

        for i in range(x_min, x_max, dash_length * 2):
            cv.line(slide_image_copy, (i, y_min), (i + dash_length, y_min), color, thickness, line_type)
            cv.line(slide_image_copy, (i, y_max), (i + dash_length, y_max), color, thickness, line_type)
        for i in range(y_min, y_max, dash_length * 2):
            cv.line(slide_image_copy, (x_min, i), (x_min, i + dash_length), color, thickness, line_type)
            cv.line(slide_image_copy, (x_max, i), (x_max, i + dash_length), color, thickness, line_type)

    cv.rectangle(slide_image_copy, (750, 50), (790, 70), selected_color, -1)

    # pen_layer를 slide_image_copy에 복사
    if mode == 'lecture':
        b, g, r, a = cv.split(pen_layer)
        overlay_color = cv.merge((b, g, r))
        mask_pen = a
        cv.copyTo(overlay_color, mask_pen, slide_image_copy)

        # highlight_layer를 slide_image_copy에 블렌딩
        alpha_value = 0.4  # 투명도 설정
        b_h, g_h, r_h, a_h = cv.split(highlight_layer)
        overlay_highlight = cv.merge((b_h, g_h, r_h))
        # 마스크 생성 (alpha 채널을 이진화)
        _, mask_highlight = cv.threshold(a_h, 0, 255, cv.THRESH_BINARY)

        # 슬라이드 이미지에서 하이라이트 영역 추출
        slide_highlight_region = cv.bitwise_and(slide_image_copy, slide_image_copy, mask=mask_highlight)

        # 하이라이트 레이어와 슬라이드 이미지의 하이라이트 영역을 블렌딩
        highlight_blend = cv.addWeighted(overlay_highlight, alpha_value, slide_highlight_region, 1 - alpha_value, 0)

        # 마스크의 역으로 슬라이드 이미지에서 하이라이트 영역 제거
        inv_mask_highlight = cv.bitwise_not(mask_highlight)
        slide_image_copy = cv.bitwise_and(slide_image_copy, slide_image_copy, mask=inv_mask_highlight)

        # 블렌딩된 하이라이트를 슬라이드 이미지에 추가
        slide_image_copy = cv.add(slide_image_copy, highlight_blend)

        # 포인터 레이어 합성
        pointer_gray = cv.cvtColor(pointer_layer, cv.COLOR_BGR2GRAY)
        _, mask_pointer = cv.threshold(pointer_gray, 254, 255, cv.THRESH_BINARY_INV)
        cv.copyTo(pointer_layer, mask_pointer, slide_image_copy)

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

    # 타이머와 모드 정보를 웹캠 이미지에 표시
    cv.putText(
        frm,
        f"Mode: {mode.capitalize()}",
        (20, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if mode == 'lecture' else (0, 0, 255),
        2,
    )
    cv.putText(
        frm,
        f"Timer: {display_time} sec",
        (20, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
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
            # pen_layer와 highlight_layer를 초기화하지 않습니다.
    elif key == ord('d'):  # 이전 슬라이드
        if current_slide > 0:
            current_slide -= 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, (800, 800))
            # pen_layer와 highlight_layer를 초기화하지 않습니다.

cap.release()
cv.destroyAllWindows()
