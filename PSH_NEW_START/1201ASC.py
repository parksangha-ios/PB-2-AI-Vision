import cv2 as cv
import mediapipe as mp
import glob
import numpy as np
import time

# Mediapipe 손 인식 초기화
mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

# 손 랜드마크 설정 (신뢰도 향상)
hand_landmark = mp_hands.Hands(
    min_detection_confidence=0.8,  # 손 인식 최소 신뢰도 증가
    min_tracking_confidence=0.8,   # 손 추적 최소 신뢰도 증가
    max_num_hands=1                # 최대 손 개수를 1로 설정하여 성능 향상
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

# 그림을 그리기 위한 레이어 생성 (알파 채널 포함)
canvas_size = (800, 800)
pen_layer = np.zeros((*canvas_size, 4), dtype=np.uint8)          # 그리기용
highlight_layer = np.zeros((*canvas_size, 4), dtype=np.uint8)    # 형광펜용

# 그리기 도구를 위한 변수들
tool_margin_left = 150  # 도구의 왼쪽 여백
tool_max_x, tool_max_y = 250 + tool_margin_left, 50
curr_tool = "laser pointer"  # 초기 도구를 'laser pointer'로 설정
time_init = True
rad = 40
var_inits = False
prevx, prevy = 0, 0

color_margin_left = 600  # 색상 선택 영역의 왼쪽 여백
color_max_x, color_max_y = 800, 50

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
selection_cooldown = 0  # 선택 쿨다운 초기화

# 도구 이미지 로드
tools = cv.imread("team2/tools1.jpg")
tools = cv.resize(tools, (tool_max_x - tool_margin_left, tool_max_y))  # 슬라이드에 맞게 도구 이미지 크기 조정

# 색상 선택 영역 정의
COLOR_SECTIONS = {
    "red": (600, 640),
    "yellow": (640, 680),
    "green": (680, 720),
    "blue": (720, 760),
    "black": (760, 800),
}

def get_tool(x):
    if x < 50 + tool_margin_left:
        return "highlighter"
    elif x < 100 + tool_margin_left:
        return "crop"
    elif x < 150 + tool_margin_left:
        return "draw"
    elif x < 200 + tool_margin_left:
        return "select_move"
    else:
        return "erase"

def get_color(x):
    for color, (start, end) in COLOR_SECTIONS.items():
        if start <= x < end:
            return color
    return None

# 전역 변수 추가/수정
summary_images = []  # summary_images를 summary_images로 변경
last_pinch_time = None
pinch_count = 0
selected_summary_index = -1  # summary 이미지 선택을 위한 인덱스

def check_summary_area(x, y):
    """summary 영역의 어떤 이미지 위에 있는지 확인"""
    if y < 600 or y > 800:  # summary 영역의 y좌표 범위
        return -1
    
    section_width = 800 // 4
    index = x // section_width
    return index if index < len(summary_images) else -1

def process_crop_tool(hand, x, y):
    global summary_images, last_pinch_time, pinch_count, selected_summary_index
    global current_cropped_image, var_inits, previous_x, previous_y
    global xii, yii, rotation_angle
    
    if not var_inits:
        xii, yii = x, y
        var_inits = True
        previous_x = x
        previous_y = y
    
    current_time = time.time()
    pinch_detected = check_pinch_gesture(hand)
    thumb_direction = detect_thumb_direction(hand, 4, 2)
    
    # summary 영역에서의 인덱스 확인
    if y > 600:
        selected_summary_index = check_summary_area(x, y)
    else:
        selected_summary_index = -1

    if current_cropped_image is not None:
        if thumb_direction != "neutral":
            # 회전 처리
            angle = 90 if thumb_direction == "right" else -90
            matrix = cv.getRotationMatrix2D(
                (current_cropped_image.shape[1]/2, current_cropped_image.shape[0]/2),
                angle, 1.0
            )
            rotated = cv.warpAffine(
                current_cropped_image, matrix,
                (current_cropped_image.shape[1], current_cropped_image.shape[0])
            )
            current_cropped_image = rotated.copy()
            
    if pinch_detected:
        if not last_pinch_time or (current_time - last_pinch_time) > 0.3:
            pinch_count += 1
            last_pinch_time = current_time
            
            if pinch_count == 2:
                if selected_summary_index != -1:
                    if y > previous_y + 50:
                        summary_images.pop(selected_summary_index)
                    elif x > previous_x + 50 and selected_summary_index < 3:
                        if selected_summary_index < len(summary_images) - 1:
                            summary_images[selected_summary_index], summary_images[selected_summary_index + 1] = \
                            summary_images[selected_summary_index + 1], summary_images[selected_summary_index]
                    elif x < previous_x - 50 and selected_summary_index > 0:
                        summary_images[selected_summary_index], summary_images[selected_summary_index - 1] = \
                        summary_images[selected_summary_index - 1], summary_images[selected_summary_index]
                else:
                    if len(summary_images) >= 4:
                        warning_img = np.zeros((200, 400, 3), dtype=np.uint8)
                        cv.putText(warning_img, "최대 4개까지 핵심노트에", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv.putText(warning_img, "저장할 수 있습니다.", (50, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv.imshow("Warning", warning_img)
                    else:
                        summary_images.append(current_cropped_image.copy())
                
                pinch_count = 0
            previous_x = x
            previous_y = y


current_cropped_image = None  # 현재 선택된 크롭 이미지

selected_color = (0, 0, 255)  # 초기 색상 (빨간색)
selected_tool = None  # 선택된 도구 초기화

# 엄지손가락의 방향 감지
def detect_thumb_direction(hand, finger_tip_id, finger_pip_id):
    thumb_tip = hand.landmark[finger_tip_id]  # 엄지손가락 끝
    thumb_base = hand.landmark[finger_pip_id]  # 엄지손가락 둘째 마디
    
    if thumb_tip.x - thumb_base.x > 0.04:  # 임계값 설정
        return "right"
    elif thumb_tip.x - thumb_base.x < -0.04:
        return "left"
    return "neutral"

# 손가락이 펴져 있는지 확인하는 함수
def is_finger_up(hand, finger_tip_id, finger_pip_id):
    return hand.landmark[finger_tip_id].y < hand.landmark[finger_pip_id].y

# 중지가 접혀 있는지 확인하는 함수
def is_finger_folded(hand, finger_tip_id, finger_pip_id):
    return hand.landmark[finger_tip_id].y > hand.landmark[finger_pip_id].y

# 주먹 제스처를 인식하는 함수
def is_fist(hand):
    # 모든 손가락이 접혀 있는지 확인
    fingers_folded = True
    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:  # 검지, 중지, 약지, 새끼
        if hand.landmark[tip_id].y < hand.landmark[pip_id].y:
            fingers_folded = False
            break
    return fingers_folded


# =========================================================
# 손가락이 펴져 있는지 접혀 있는지 확인하는 함수(오른손만 사용)
def get_finger_status(hand):
    fingers = []

    if hand.landmark[4].x < hand.landmark[2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    tip_id = [8, 12, 16, 20]
    pip_id = [6, 10, 14, 18]
    for tip, pip in zip(tip_id, pip_id):
        if hand.landmark[tip].y < hand.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def recognize_gesture(fingers_status):
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers_status == [0, 1, 0, 0, 0]:
        return 'point'
    elif fingers_status == [0, 1, 1, 0, 0]:
        return 'peace'
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'


def rotate_image(image, direction):
    """이미지 회전 함수"""
    h, w = image.shape[:2]
    center = (w/2, h/2)
    angle = 90 if direction == "right" else -90
    matrix = cv.getRotationMatrix2D(center, -angle, 1.0)
    return cv.warpAffine(image, matrix, (w, h))

def check_pinch_gesture(hand):
    """엄지와 검지의 핀치 제스처 확인"""
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05  # 임계값 조정 가능


# =========================================================



# 전역 변수 초기화
ctime = None
ptime = None
timer_running = False
start_time = None

# x1, y1, xii, yii 변수 초기화
x1, y1 = 0, 0   # select_move 용
xii, yii = 0, 0 # enlarge 용

def process_frame(frame):
    global curr_tool, time_init, rad, var_inits, prevx, prevy
    global selection_done, selected_area_pen, selected_area_highlight
    global x_offset, y_offset, selection_cooldown
    global ctime, ptime
    global timer_running, start_time
    global selected_color, selected_tool
    global x1, y1, xii, yii  # x1, y1, xii, yii를 전역 변수로 선언
    global slide_image_copy  # slide_image_copy를 전역 변수로 선언
    global show_summary, last_gesture_time
    global last_thumb_direction, rotation_start_time, is_rotating
    global pinch_count, last_pinch_time

    pinch_count = 0
    last_pinch_time = None
    # key_notes = []  # 핵심노트 저장 리스트
    rotation_start_time = 0
    is_rotating = False
    last_thumb_direction = "neutral"
    
    current_time = time.time()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    pointer_layer = np.full((*canvas_size, 3), 255, dtype=np.uint8)

    if op.multi_hand_landmarks:
        for hand in op.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frm, hand, mp_hands.HAND_CONNECTIONS)
            # 주먹 제스처 인식하여 타이머 시작
            if is_fist(hand) and not timer_running:
                start_time = time.time()
                timer_running = True
                print("Lecture timer started.")

            # 손가락 상태 확인
            fingers_status = get_finger_status(hand)
            gesture = recognize_gesture(fingers_status)
            
            thumb_direction = detect_thumb_direction(hand, 4, 2)   # 엄지 방향
            index_up = is_finger_up(hand, 8, 6)    # 검지
            middle_up = is_finger_up(hand, 12, 10) # 중지
            fingers_up_status = index_up and middle_up  # 검지와 중지가 펴진 상태
            middle_folded_status = not middle_up        # 중지가 접힌 상태

            x = int(hand.landmark[8].x * canvas_size[0])  # 검지 끝 좌표
            y = int(hand.landmark[8].y * canvas_size[1])
            
            # standby 제스처 감지 시 summary 토글
            current_time = time.time()
            if gesture == 'standby' and (current_time - last_gesture_time) > 0.5:
                show_summary = not show_summary
                last_gesture_time = current_time

            # 도구 선택 영역
            if tool_margin_left < x < tool_max_x and y < tool_max_y:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv.circle(slide_image_copy, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    selected_tool = get_tool(x)
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

            # 색깔 선택 영역
            if color_margin_left <= x <= color_max_x and y <= color_max_y:
                selected_tool = get_color(x)
                if selected_tool:
                    # 색상 매핑
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
            elif curr_tool == "crop" or curr_tool == "select_move":
                size_rect = 10
                cv.rectangle(pointer_layer, (x - size_rect // 2, y - size_rect // 2),
                             (x + size_rect // 2, y + size_rect // 2), (0, 255, 255), 2)

            # 그리기 도구 구현
            if curr_tool == "draw":
                if fingers_up_status:
                    if prevx != 0 and prevy != 0:
                        color_with_alpha = selected_color + (255,)  # 알파 채널 추가
                        cv.line(pen_layer, (prevx, prevy), (x, y), color_with_alpha, thick_pen)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = 0, 0

            elif curr_tool == "highlighter":
                if fingers_up_status:
                    if prevx != 0 and prevy != 0:
                        color_with_alpha = selected_color + (int(255 * 0.4),)  # 알파 채널 추가 (투명도 적용)
                        cv.line(highlight_layer, (prevx, prevy), (x, y), color_with_alpha, thick_highlighter)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = 0, 0

            # crop 도구 부분 수정
            elif curr_tool == "crop":
                process_crop_tool(hand, x, y)

            elif curr_tool == "select_move":
                if not selection_done:
                    # 선택 모드
                    if time.time() > selection_cooldown:
                        if fingers_up_status:
                            # 영역 선택 시작
                            if not var_inits:
                                x1, y1 = x, y
                                var_inits = True
                            else:
                                # 점선 사각형 그리기
                                rect_start = (x1, y1)
                                rect_end = (x, y)
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
                    # 이동 모드
                    if middle_folded_status:
                        # 중지를 접은 상태로 이동
                        # 선택 영역의 크기를 계산
                        if selected_area_pen is not None:
                            w = selected_area_pen.shape[1]
                            h = selected_area_pen.shape[0]
                        elif selected_area_highlight is not None:
                            w = selected_area_highlight.shape[1]
                            h = selected_area_highlight.shape[0]
                        else:
                            w = h = 0

                        x_offset = x - w // 2
                        y_offset = y - h // 2
                    elif fingers_up_status:
                        # 위치 확정
                        # pen_layer에 합성
                        if selected_area_pen is not None:
                            h_pen, w_pen = selected_area_pen.shape[:2]
                            x1_dst_pen = max(0, x_offset)
                            y1_dst_pen = max(0, y_offset)
                            x2_dst_pen = min(canvas_size[0], x_offset + w_pen)
                            y2_dst_pen = min(canvas_size[1], y_offset + h_pen)

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
                            x2_dst_highlight = min(canvas_size[0], x_offset + w_highlight)
                            y2_dst_highlight = min(canvas_size[1], y_offset + h_highlight)

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
                if fingers_up_status:
                    cv.circle(pen_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)
                    cv.circle(highlight_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)

    return pointer_layer

show_summary = False
last_gesture_time = 0
cap = cv.VideoCapture(0)

# 모드 및 타이머 변수 초기화
mode = 'lecture'  # 현재 모드: 'lecture' 또는 'break'
lecture_duration = 100  # 강의 모드에서의 타이머 시간 (초)
break_duration = 5     # 쉬는 시간 모드 시간 (초)
display_time = lecture_duration  # display_time 초기화

while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, canvas_size)

    # 현재 모드에 따라 슬라이드 이미지 설정
    if mode == 'lecture':
        slide_image_current = slide_image.copy()
    elif mode == 'break':
        slide_image_current = cv.imread("breaktime.png")
        slide_image_current = cv.resize(slide_image_current, canvas_size)

    slide_image_copy = slide_image_current.copy()

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

    # 프레임 처리
    pointer_layer = process_frame(frm)

    # 선택된 영역 이동 및 합성
    if selection_done and curr_tool == "select_move":
        # pen_layer의 선택 영역 이동 및 합성
        if selected_area_pen is not None:
            h_pen, w_pen = selected_area_pen.shape[:2]
            x1_dst_pen = max(0, x_offset)
            y1_dst_pen = max(0, y_offset)
            x2_dst_pen = min(canvas_size[0], x_offset + w_pen)
            y2_dst_pen = min(canvas_size[1], y_offset + h_pen)

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
            x2_dst_highlight = min(canvas_size[0], x_offset + w_highlight)
            y2_dst_highlight = min(canvas_size[1], y_offset + h_highlight)

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
        slide_image_copy[:tool_max_y, tool_margin_left:tool_max_x] = cv.addWeighted(
            tools, 0.7, slide_image_copy[:tool_max_y, tool_margin_left:tool_max_x], 0.3, 0
        )

        cv.putText(
            slide_image_copy,
            curr_tool,
            (270 + tool_margin_left, 30),
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

    if show_summary and summary_images:
        # 슬라이드 이미지 리사이즈 (600*800)
        slide_display = cv.resize(slide_image_copy, (800, 600))
        
        # summary 영역 생성 (200*800)
        summary_height = 200
        summary_width = 800
        summary_area = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
        
        # 각 크롭 이미지의 너비 계산
        section_width = summary_width // 4
        
        # 최대 4개의 이미지 표시
        for idx, img in enumerate(summary_images[-4:]):  # 최근 4개만 표시
            if img is None:
                continue
                
            # 비율 유지하면서 리사이즈
            h, w = img.shape[:2]
            ratio = min(section_width/w, summary_height/h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            # 이미지 리사이즈
            resized_img = cv.resize(img, (new_w, new_h))
            
            # 중앙 정렬을 위한 좌표 계산
            x_offset = idx * section_width + (section_width - new_w) // 2
            y_offset = (summary_height - new_h) // 2
            
            # summary 영역에 이미지 삽입
            summary_area[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        # 최종 디스플레이 이미지 생성
        final_display = np.vstack([slide_display, summary_area])
        
        # 강의 슬라이드 표시
        cv.imshow("Lecture", final_display)
    else:
        # summary 없이 원본 크기로 표시
        cv.imshow("Lecture", slide_image_copy)
    
    # 웹캠 피드 표시
    cv.imshow("Webcam", frm)
    key = cv.waitKey(5) & 0xFF  # waitKey 값을 증가시켜 CPU 부하 감소
    if key == 27:  # ESC 키
        break
    elif key == ord('u'):  # 다음 슬라이드
        if current_slide < total_slides - 1:
            current_slide += 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, canvas_size)
            # pen_layer와 highlight_layer를 초기화하지 않습니다.
    elif key == ord('d'):  # 이전 슬라이드
        if current_slide > 0:
            current_slide -= 1
            slide_image = cv.imread(img_files[current_slide])
            slide_image = cv.resize(slide_image, canvas_size)
            # pen_layer와 highlight_layer를 초기화하지 않습니다.


cap.release()
cv.destroyAllWindows()
