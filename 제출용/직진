import cv2 as cv
import mediapipe as mp
import glob
import numpy as np
import time
import math

# Mediapipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hand_landmark = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)

# 슬라이드 이미지 로드
img_files = sorted(glob.glob('./images/*.jpg'))
if not img_files:
    print("No slide images found.")
    exit()

current_slide = 0
total_slides = len(img_files)
canvas_size = (800, 800)
slide_image = cv.imread(img_files[current_slide])
slide_image = cv.resize(slide_image, canvas_size)

slide_pen_layers = [np.zeros((*canvas_size, 4), dtype=np.uint8) for _ in range(total_slides)]
slide_highlight_layers = [np.zeros((*canvas_size, 4), dtype=np.uint8) for _ in range(total_slides)]

pen_layer = slide_pen_layers[current_slide].copy()
highlight_layer = slide_highlight_layers[current_slide].copy()

# temp_layer: 굿노트식 직선 보정용 임시 레이어
temp_layer = np.zeros((*canvas_size, 4), dtype=np.uint8)

tool_margin_left = 150
tool_max_x, tool_max_y = 250 + tool_margin_left, 50
curr_tool = "laser pointer"
time_init = True
rad = 40
var_inits = False
prevx, prevy = 0, 0

thick_pen = 4
thick_highlighter = 30
thick_erase = 25

selection_done = False
selected_area_pen = None
selected_area_highlight = None
x_offset = 0
y_offset = 0
selection_cooldown = 0

tools = cv.imread("tools1.jpg")
tools = cv.resize(tools, (tool_max_x - tool_margin_left, tool_max_y))

color_margin_left = 400
color_max_x, color_max_y = 800, 50

COLOR_SECTIONS = {
    "red": ((420, 25), (0, 0, 255)),
    "yellow": ((460, 25), (0, 255, 255)),
    "green": ((500, 25), (0, 255, 0)),
    "blue": ((540, 25), (255, 0, 0)),
    "black": ((580, 25), (0, 0, 0))
}


def draw_color_circles(image):
    for color, (center, bgr_color) in COLOR_SECTIONS.items():
        cv.circle(image, center, 20, bgr_color, -1)
draw_color_circles(slide_image)


def calculate_thumb_angle(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]  # 엄지 끝
    thumb_mcp = hand_landmarks.landmark[2]  # 엄지 관절
    dx = thumb_tip.x - thumb_mcp.x
    dy = thumb_tip.y - thumb_mcp.y
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)


last_slide_time = 0
slow_delay = 1.3
fast_delay = 0.1


def get_tool(x):
    if x < 50 + tool_margin_left:
        return "highlighter"
    elif x < 100 + tool_margin_left:
        return "select mode"
    elif x < 150 + tool_margin_left:
        return "draw"
    elif x < 200 + tool_margin_left:
        return "select_move"
    else:
        return "erase"

def get_color(x):
    for color, (center, bgr_color) in COLOR_SECTIONS.items():
        if abs(center[0] - x) <= 20:
            return color
    return None

summary_images = []
image_sizes = []
summary_operations = []
last_pinch_time = None
pinch_count = 0
selected_summary_index = -1
moving_summary = False
moving_summary_index = -1

crop_start_point = None
crop_end_point = None
cropping = False

rotation_mode = False
rotation_angle = 0
rotation_start_time = None
rotation_delay = 0.5
current_cropped_image = None

selected_color = (0, 0, 255)
selected_tool = None
current_transformation_tool = None

def detect_thumb_direction(hand, finger_tip_id, finger_pip_id):
    thumb_tip = hand.landmark[finger_tip_id]
    thumb_base = hand.landmark[finger_pip_id]

    if thumb_tip.x - thumb_base.x > 0.04:
        return "right"
    elif thumb_tip.x - thumb_base.x < -0.04:
        return "left"
    return "neutral"

middle_finger_up_start = 0
middle_finger_delay = 0.5
middle_finger_confirmed = False

def is_finger_up(hand, finger_tip_id, finger_pip_id):
    return hand.landmark[finger_tip_id].y < hand.landmark[finger_pip_id].y

def check_middle_finger_status(middle_up):
    global middle_finger_up_start, middle_finger_confirmed
    current_time = time.time()
    if middle_up:
        if middle_finger_up_start == 0:
            middle_finger_up_start = current_time
        elif current_time - middle_finger_up_start >= middle_finger_delay:
            middle_finger_confirmed = True
    else:
        middle_finger_up_start = 0
        middle_finger_confirmed = False
    return middle_finger_confirmed

def is_all_fingers_up(hand):
    for tip_id, pip_id in [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]:
        if not is_finger_up(hand, tip_id, pip_id):
            return False
    return True

def is_fist(hand):
    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand.landmark[tip_id].y < hand.landmark[pip_id].y:
            return False
    return True

def is_one(hand):
    index_up = is_finger_up(hand, 8, 6)
    middle_up = not is_finger_up(hand, 12, 10)
    ring_up = not is_finger_up(hand, 16, 14)
    pinky_up = not is_finger_up(hand, 20, 18)
    return index_up and middle_up and ring_up and pinky_up

def is_two(hand):
    index_up = is_finger_up(hand, 8, 6)
    middle_up = is_finger_up(hand, 12, 10)
    ring_up = not is_finger_up(hand, 16, 14)
    pinky_up = not is_finger_up(hand, 20, 18)
    index_x = hand.landmark[8].x
    middle_x = hand.landmark[12].x
    return index_up and middle_up and ring_up and pinky_up and abs(index_x - middle_x) > 0.05

def is_three(hand):
    index_up = is_finger_up(hand, 8, 6)
    middle_up = is_finger_up(hand, 12, 10)
    ring_up = is_finger_up(hand, 16, 14)
    pinky_up = not is_finger_up(hand, 20, 18)
    return index_up and middle_up and ring_up and pinky_up

def check_pinch_gesture(hand):
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    pinch = distance < 0.05
    if pinch:
        print("Pinch gesture detected")
    return pinch

def check_summary_area(x, y):
    if y < 600 or y > 800:
        return -1
    section_width = 800 // 4
    index = x // section_width
    return index if index < len(summary_images) else -1

def process_crop_tool(hand, x, y, fx=1.0, fy=1.0, tool_type="crop"):
    global summary_images, last_pinch_time, pinch_count, selected_summary_index
    global current_cropped_image, var_inits, prevx, prevy
    global x_offset, y_offset, rotation_angle
    global crop_start_point, crop_end_point, cropping
    global slide_image_copy
    global rotation_mode
    global rotation_angle, current_cropped_image, rotation_start_time
    global should_close_rotating_window
    global current_transformation_tool

    current_time = time.time()
    pinch_detected = check_pinch_gesture(hand)
    thumb_direction = detect_thumb_direction(hand, 4, 2)

    if curr_tool != tool_type and not rotation_mode:
        should_close_rotating_window = True
    if pinch_detected:
        if not last_pinch_time or (current_time - last_pinch_time) > 0.3:
            pinch_count += 1
            last_pinch_time = current_time

            if pinch_count == 1 and not cropping:
                crop_start_point = (x, y)
                cropping = True
                print(f"{tool_type.capitalize()} started at: {crop_start_point}")
            elif pinch_count == 2 and cropping:
                crop_end_point = (x, y)
                cropping = False
                pinch_count = 0
                print(f"{tool_type.capitalize()} ended at: {crop_end_point}")

                x_start, y_start = min(crop_start_point[0], crop_end_point[0]), min(crop_start_point[1], crop_end_point[1])
                x_end, y_end = max(crop_start_point[0], crop_end_point[0]), max(crop_start_point[1], crop_end_point[1])

                if 0 <= y_start < y_end <= canvas_size[1] and 0 <= x_start < x_end <= canvas_size[0]:
                    combined_image = slide_image_copy.copy()
                    
                    b_pen, g_pen, r_pen, a_pen = cv.split(pen_layer)
                    overlay_color_pen = cv.merge((b_pen, g_pen, r_pen))
                    mask_pen = a_pen
                    cv.copyTo(overlay_color_pen, mask_pen, combined_image)

                    alpha_value = 0.4
                    b_h, g_h, r_h, a_h = cv.split(highlight_layer)
                    overlay_highlight = cv.merge((b_h, g_h, r_h))
                    _, mask_highlight = cv.threshold(a_h, 0, 255, cv.THRESH_BINARY)
                    slide_highlight_region = cv.bitwise_and(combined_image, combined_image, mask=mask_highlight)
                    highlight_blend = cv.addWeighted(overlay_highlight, alpha_value, slide_highlight_region, 1 - alpha_value, 0)
                    inv_mask_highlight = cv.bitwise_not(mask_highlight)
                    combined_image = cv.bitwise_and(combined_image, combined_image, mask=inv_mask_highlight)
                    combined_image = cv.add(combined_image, highlight_blend)
                    
                    cropped_area = combined_image[y_start:y_end, x_start:x_end].copy()

                    if cropped_area.size == 0:
                        print("Invalid area selection.")
                    else:
                        if fx != 1.0 or fy != 1.0:
                            processed_image = cv.resize(cropped_area, None, fx=fx, fy=fy,
                                                        interpolation=cv.INTER_LINEAR if fx > 1 else cv.INTER_AREA)
                        else:
                            processed_image = cropped_area

                        current_cropped_image = processed_image
                        current_transformation_tool = tool_type
                        
                        rotation_mode = True
                        rotation_angle = 0
                        rotation_start_time = time.time()
                        print(f"{tool_type.capitalize()} completed. Entering rotation mode.")
                else:
                    print("Selection coordinates out of bounds.")

    if cropping and crop_start_point is not None:
        cv.rectangle(slide_image_copy, crop_start_point, (x, y), (255, 0, 0), 2)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv.getRotationMatrix2D(center, -angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv.warpAffine(image, matrix, (new_w, new_h), borderValue=(255, 255, 255))

def get_finger_status(hand):
    fingers = []
    if hand.landmark[4].x < hand.landmark[2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    tip_id = [8, 12, 16, 20]
    pip_id = [6, 10, 14, 18]
    for tip, pip_ in zip(tip_id, pip_id):
        if hand.landmark[tip].y < hand.landmark[pip_].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

ctime = None
ptime = None
timer_running = False
start_time = None

mode_select_time = None
select_mode = None

x1, y1 = 0, 0
xii, yii = 0, 0
max_H, max_W = 0, 0

should_close_rotating_window = False

erase_click_count = 0

smoothed_x = None
smoothed_y = None
alpha = 0.2

# 직선 보정 관련 변수 추가
drawing = False
line_points = []
line_start_time = None
no_movement_start = None
last_position = None
pause_threshold = 2.0

def straighten_line(tool_mode):
    global line_points, pen_layer, highlight_layer, selected_color, thick_pen, thick_highlighter, temp_layer

    if len(line_points) < 2:
        line_points = []
        return

    start_point = line_points[0]
    end_point = line_points[-1]

    # temp_layer 초기화
    temp_layer[:, :] = 0

    # 펜/형광펜 모드에 따라 pen_layer나 highlight_layer에 직선 그리기
    if tool_mode == "draw":
        thickness = thick_pen
        color_with_alpha = selected_color + (255,)
        cv.line(pen_layer, start_point, end_point, color_with_alpha, thickness)
    else:
        thickness = thick_highlighter
        color_with_alpha = selected_color + (int(255 * 0.4),)
        cv.line(highlight_layer, start_point, end_point, color_with_alpha, thickness)

    line_points = []

def process_frame(frame):
    global curr_tool, time_init, rad, var_inits, prevx, prevy
    global selection_done, selected_area_pen, selected_area_highlight
    global x_offset, y_offset, selection_cooldown
    global ctime, ptime
    global timer_running, start_time
    global selected_color, selected_tool
    global x1, y1, xii, yii
    global slide_image_copy
    global rotation_mode, rotation_angle, current_cropped_image
    global summary_images, summary_operations
    global pinch_count, last_pinch_time
    global crop_start_point, crop_end_point, cropping
    global rotation_start_time, rotation_delay
    global should_close_rotating_window
    global moving_summary, moving_summary_index
    global pen_layer, highlight_layer, temp_layer
    global mode_select_time, select_mode
    global erase_click_count
    global current_time, last_slide_time, slide_delay
    global current_slide, slide_image
    global smoothed_x, smoothed_y, alpha
    global max_H, max_W
    global current_transformation_tool
    global drawing, line_points, line_start_time, no_movement_start, last_position, pause_threshold

    current_time = time.time()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    pointer_layer = np.full((*canvas_size, 3), 255, dtype=np.uint8)

    if rotation_mode:
        if op.multi_hand_landmarks:
            for hand in op.multi_hand_landmarks:
                if current_time - rotation_start_time > rotation_delay:
                    thumb_direction = detect_thumb_direction(hand, 4, 2)
                    if thumb_direction == "left":
                        rotation_angle -= 2
                    elif thumb_direction == "right":
                        rotation_angle += 2
                    if is_all_fingers_up(hand):
                        rotated_image = rotate_image(current_cropped_image, rotation_angle)
                        if len(summary_images) >= 4:
                            summary_images.pop(0)
                            image_sizes.pop(0)
                            summary_operations.pop(0)

                        h, w = rotated_image.shape[:2]
                        image_sizes.append((h, w))
                        if current_transformation_tool is not None:
                            final_operation = current_transformation_tool + "+rotated"
                        else:
                            final_operation = "rotated"

                        summary_images.append(rotated_image.copy())
                        summary_operations.append(final_operation)

                        print(f"Rotated image added to summary. Total summary images: {len(summary_images)}")
                        rotation_mode = False
                        should_close_rotating_window = True
                        curr_tool = 'laser pointer'
                        print("Rotation confirmed. Switched back to laser pointer.")

        if rotation_mode and current_cropped_image is not None:
            rotated_image = rotate_image(current_cropped_image, rotation_angle)
            cv.imshow("Rotating Image", rotated_image)

        return pointer_layer

    if op.multi_hand_landmarks:
        for hand in op.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if is_fist(hand) and not timer_running:
                start_time = time.time()
                timer_running = True
                print("Lecture timer started.")

            x = int(hand.landmark[8].x * canvas_size[0])
            y = int(hand.landmark[8].y * canvas_size[1])

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
                        if selected_tool == "erase":
                            pen_layer[:, :] = 0
                            highlight_layer[:, :] = 0
                            temp_layer[:, :] = 0
                            curr_tool = "laser pointer"
                            print("Erase all drawings. Switched back to laser pointer.")
                        else:
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

            if color_margin_left <= x <= color_max_x and y <= color_max_y:
                selected_color_name = get_color(x)
                if selected_color_name:
                    color_map = {
                        "red": (0, 0, 255),
                        "yellow": (0, 255, 255),
                        "green": (0, 255, 0),
                        "blue": (255, 0, 0),
                        "black": (0, 0, 0),
                    }
                    selected_color = color_map[selected_color_name]
                    print(f"Selected color: {selected_color_name}")
                cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)

            index_up = is_finger_up(hand, 8, 6)
            middle_up = is_finger_up(hand, 12, 10)
            middle_confirmed = check_middle_finger_status(middle_up)
            fingers_up_status = index_up and middle_up
            fingers_up_status_highlighter = index_up and middle_confirmed

            if curr_tool == "laser pointer":
                cv.circle(pointer_layer, (x, y), 5, (0, 0, 255), -1)
            elif curr_tool == "erase":
                size = thick_erase
                cv.rectangle(pointer_layer, (x - size // 2, y - size // 2),
                             (x + size // 2, y + size // 2), (0, 0, 0), -1)
                if fingers_up_status:
                    cv.circle(pen_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)
                    cv.circle(highlight_layer, (x, y), thick_erase, (0, 0, 0, 0), -1)

            elif curr_tool == "draw":
                cv.circle(pointer_layer, (x, y), thick_pen // 2, selected_color, 2)
                # 여기서 기존 direct draw 부분을 temp_layer로 변경
                if fingers_up_status:
                    if not drawing:
                        drawing = True
                        line_points = []
                        line_start_time = time.time()
                        no_movement_start = None
                        last_position = None
                        prevx, prevy = 0, 0
                        temp_layer[:, :] = 0

                    if smoothed_x is None or smoothed_y is None:
                        smoothed_x, smoothed_y = x, y
                    else:
                        smoothed_x = int(alpha * x + (1 - alpha) * smoothed_x)
                        smoothed_y = int(alpha * y + (1 - alpha) * smoothed_y)

                    line_points.append((smoothed_x, smoothed_y))

                    # temp_layer에 곡선 표시
                    if prevx != 0 and prevy != 0:
                        color_with_alpha = selected_color + (255,)
                        cv.line(temp_layer, (prevx, prevy), (smoothed_x, smoothed_y), color_with_alpha, thick_pen)

                    prevx, prevy = smoothed_x, smoothed_y

                    # 움직임 체크
                    if last_position is not None:
                        dist = math.hypot(smoothed_x - last_position[0], smoothed_y - last_position[1])
                        if dist < 2:
                            if no_movement_start is None:
                                no_movement_start = time.time()
                            else:
                                if time.time() - no_movement_start > pause_threshold:
                                    # 직선 보정
                                    straighten_line("draw")
                                    drawing = False
                                    prevx, prevy = 0, 0
                                    smoothed_x, smoothed_y = None, None
                                    no_movement_start = None
                                    line_start_time = None
                                    last_position = None
                        else:
                            no_movement_start = None
                    last_position = (smoothed_x, smoothed_y)
                else:
                    if drawing:
                        drawing = False
                        prevx, prevy = 0, 0
                        smoothed_x, smoothed_y = None, None
                        if line_points:
                            straighten_line("draw")
                        line_points = []
                        line_start_time = None
                        no_movement_start = None
                        last_position = None

            elif curr_tool == "highlighter":
                cv.circle(pointer_layer, (x, y), thick_highlighter // 2, selected_color, 2)
                if fingers_up_status_highlighter:
                    if not drawing:
                        drawing = True
                        line_points = []
                        line_start_time = time.time()
                        no_movement_start = None
                        last_position = None
                        prevx, prevy = 0, 0
                        temp_layer[:, :] = 0

                    if smoothed_x is None or smoothed_y is None:
                        smoothed_x, smoothed_y = x, y
                    else:
                        smoothed_x = int(alpha * x + (1 - alpha) * smoothed_x)
                        smoothed_y = int(alpha * y + (1 - alpha) * smoothed_y)

                    line_points.append((smoothed_x, smoothed_y))

                    # temp_layer에 곡선 표시
                    if prevx != 0 and prevy != 0:
                        color_with_alpha = selected_color + (int(255 * 0.4),)
                        cv.line(temp_layer, (prevx, prevy), (smoothed_x, smoothed_y), color_with_alpha, thick_highlighter)

                    prevx, prevy = smoothed_x, smoothed_y

                    if last_position is not None:
                        dist = math.hypot(smoothed_x - last_position[0], smoothed_y - last_position[1])
                        if dist < 2:
                            if no_movement_start is None:
                                no_movement_start = time.time()
                            else:
                                if time.time() - no_movement_start > pause_threshold:
                                    straighten_line("highlighter")
                                    drawing = False
                                    prevx, prevy = 0, 0
                                    smoothed_x, smoothed_y = None, None
                                    no_movement_start = None
                                    line_start_time = None
                                    last_position = None
                        else:
                            no_movement_start = None
                    last_position = (smoothed_x, smoothed_y)
                else:
                    if drawing:
                        drawing = False
                        prevx, prevy = 0, 0
                        smoothed_x, smoothed_y = None, None
                        if line_points:
                            straighten_line("highlighter")
                        line_points = []
                        line_start_time = None
                        no_movement_start = None
                        last_position = None

            elif curr_tool == "select mode":
                size_rect = 10
                cv.rectangle(pointer_layer, (x - size_rect // 2, y - size_rect // 2),
                             (x + size_rect // 2, y + size_rect // 2), (0, 255, 255), 2)
                if time.time() > selection_cooldown:
                    if is_one(hand):
                        if mode_select_time is None:
                            mode_select_time = time.time()
                        elif time.time() - mode_select_time > 0.8:
                            select_mode = "crop"
                            curr_tool = "crop"
                            print("Mode selected: crop")
                            mode_select_time = None
                            selection_cooldown = time.time() + 1.0
                            var_inits = False
                    elif is_two(hand):
                        if mode_select_time is None:
                            mode_select_time = time.time()
                        elif time.time() - mode_select_time > 0.8:
                            select_mode = "enlarge"
                            curr_tool = "enlarge"
                            print("Mode selected: enlarge")
                            mode_select_time = None
                            selection_cooldown = time.time() + 1.0
                            var_inits = False
                    elif is_three(hand):
                        if mode_select_time is None:
                            mode_select_time = time.time()
                        elif time.time() - mode_select_time > 0.8:
                            select_mode = "shrink"
                            curr_tool = "shrink"
                            print("Mode selected: shrink")
                            mode_select_time = None
                            selection_cooldown = time.time() + 1.0
                            var_inits = False
                    else:
                        mode_select_time = None
                else:
                    mode_select_time = None
            elif curr_tool == "crop":
                process_crop_tool(hand, x, y, tool_type="crop")
            elif curr_tool == "enlarge":
                process_crop_tool(hand, x, y, fx=2.0, fy=2.0, tool_type="enlarge")
            elif curr_tool == "shrink":
                process_crop_tool(hand, x, y, fx=0.6, fy=0.6, tool_type="shrink")
            elif curr_tool == "select_move":
                # 선택 이동 로직 기존 유지
                if not selection_done:
                    if time.time() > selection_cooldown:
                        if fingers_up_status:
                            if not var_inits:
                                x1, y1 = x, y
                                var_inits = True
                            else:
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
                        elif index_up and not middle_up:
                            if var_inits:
                                x_start, x_end = min(x1, x), max(x1, x)
                                y_start, y_end = min(y1, y), max(y1, y)
                                selected_area_pen = pen_layer[y_start:y_end, x_start:x_end].copy()
                                selected_area_highlight = highlight_layer[y_start:y_end, x_start:x_end].copy()
                                x_offset = x_start
                                y_offset = y_start
                                selection_done = True
                                var_inits = False
                                print("Selection completed. Switched to move mode.")
                                pen_layer[y_start:y_end, x_start:x_end] = 0
                                highlight_layer[y_start:y_end, x_start:x_end] = 0
                else:
                    if index_up and not middle_up:
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

                        # pen 복원
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

                            selected_area_moved_pen = selected_area_pen[y1_src_pen:y2_src_pen, x1_src_pen:x2_src_pen]
                            b_pen, g_pen, r_pen, a_pen = cv.split(selected_area_moved_pen)
                            mask_pen = a_pen
                            cv.copyTo(selected_area_moved_pen, mask_pen, pen_layer[y1_dst_pen:y2_dst_pen, x1_dst_pen:x2_dst_pen])

                        # highlight 복원
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

                            selected_area_moved_highlight = selected_area_highlight[y1_src_highlight:y2_src_highlight, x1_src_highlight:x2_src_highlight]
                            b_highlight, g_highlight, r_highlight, a_highlight = cv.split(selected_area_moved_highlight)
                            mask_highlight = a_highlight
                            cv.copyTo(selected_area_moved_highlight, mask_highlight, highlight_layer[y1_dst_highlight:y2_dst_highlight, x1_dst_highlight:x2_dst_highlight])

                        selected_area_pen = None
                        selected_area_highlight = None
                        selection_done = False
                        var_inits = False
                        selection_cooldown = time.time() + 0.8
                        print("Position confirmed.")

            if curr_tool == "laser pointer" and check_pinch_gesture(hand):
                summary_index = check_summary_area(x, y)
                if summary_index != -1 and len(summary_images) > summary_index:
                    if not moving_summary:
                        moving_summary = True
                        moving_summary_index = summary_index
                        print(f"Selected summary {moving_summary_index + 1} for moving.")
                    else:
                        target_index = summary_index
                        if target_index != moving_summary_index:
                            summary_images[moving_summary_index], summary_images[target_index] = summary_images[target_index], summary_images[moving_summary_index]
                            image_sizes[moving_summary_index], image_sizes[target_index] = image_sizes[target_index], image_sizes[moving_summary_index]
                            summary_operations[moving_summary_index], summary_operations[target_index] = summary_operations[target_index], summary_operations[moving_summary_index]
                            print(f"Swapped summary {moving_summary_index + 1} with summary {target_index + 1}.")
                        moving_summary = False
                        moving_summary_index = -1

    return pointer_layer

cap = cv.VideoCapture(0)

mode = 'lecture'
lecture_duration = 100
break_duration = 5
display_time = lecture_duration

while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, canvas_size)
    op = hand_landmark.process(cv.cvtColor(frm, cv.COLOR_BGR2RGB))
    
    if op.multi_hand_landmarks and op.multi_handedness and not rotation_mode:
        for i, hand_landmarks in enumerate(op.multi_hand_landmarks):
            handedness_label = op.multi_handedness[i].classification[0].label
            if handedness_label == "Left":
                thumb_angle = calculate_thumb_angle(hand_landmarks)
                slide_delay = slow_delay if 135 >= thumb_angle >= 45 else fast_delay

                if time.time() - last_slide_time > slide_delay:
                    thumb_direction = hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x

                    if thumb_direction > 0.05:
                        if current_slide < len(img_files) - 1:
                            slide_pen_layers[current_slide] = pen_layer.copy()
                            slide_highlight_layers[current_slide] = highlight_layer.copy()
                            current_slide += 1
                            slide_image = cv.imread(img_files[current_slide])
                            slide_image = cv.resize(slide_image, canvas_size)
                            pen_layer = slide_pen_layers[current_slide].copy()
                            highlight_layer = slide_highlight_layers[current_slide].copy()
                            print(f"왼손 인식: 오른쪽 슬라이드로 이동 (슬라이드 {current_slide + 1}/{len(img_files)})")
                            last_slide_time = time.time()
                            draw_color_circles(slide_image)

                    elif thumb_direction < -0.05:
                        if current_slide > 0:
                            slide_pen_layers[current_slide] = pen_layer.copy()
                            slide_highlight_layers[current_slide] = highlight_layer.copy()
                            current_slide -= 1
                            slide_image = cv.imread(img_files[current_slide])
                            slide_image = cv.resize(slide_image, canvas_size)
                            pen_layer = slide_pen_layers[current_slide].copy()
                            highlight_layer = slide_highlight_layers[current_slide].copy()
                            print(f"왼손 인식: 왼쪽 슬라이드로 이동 (슬라이드 {current_slide + 1}/{len(img_files)})")
                            last_slide_time = time.time()
                            draw_color_circles(slide_image)
            else:
                print("오른손 인식: 슬라이드 동작 없음")

    if mode == 'lecture':
        slide_image_current = slide_image.copy()
    else:
        slide_image_current = cv.imread("breaktime.png")
        slide_image_current = cv.resize(slide_image_current, canvas_size)

    slide_image_copy = slide_image_current.copy()

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
            display_time = lecture_duration
    else:
        elapsed_time = time.time() - start_time
        display_time = int(break_duration - elapsed_time)
        if elapsed_time >= break_duration:
            mode = 'lecture'
            timer_running = False
            start_time = None
            display_time = lecture_duration
            print("Break time over. Back to lecture mode.")

    display_time = max(display_time, 0)

    pointer_layer = process_frame(frm)

    if selection_done and curr_tool == "select_move":
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

            selected_area_moved_pen = selected_area_pen[y1_src_pen:y2_src_pen, x1_src_pen:x2_src_pen]
            b_pen, g_pen, r_pen, a_pen = cv.split(selected_area_moved_pen)
            overlay_color_pen = cv.merge((b_pen, g_pen, r_pen))
            mask_pen = a_pen
            roi_pen = slide_image_copy[y1_dst_pen:y2_dst_pen, x1_dst_pen:x2_dst_pen]
            cv.copyTo(overlay_color_pen, mask_pen, roi_pen)

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

            selected_area_moved_highlight = selected_area_highlight[y1_src_highlight:y2_src_highlight, x1_src_highlight:x2_src_highlight]
            b_highlight, g_highlight, r_highlight, a_highlight = cv.split(selected_area_moved_highlight)
            overlay_color_highlight = cv.merge((b_highlight, g_highlight, r_highlight))
            mask_highlight = a_highlight
            roi_highlight = slide_image_copy[y1_dst_highlight:y2_dst_highlight, x1_dst_highlight:x2_dst_highlight]
            cv.copyTo(overlay_color_highlight, mask_highlight, roi_highlight)

        thickness = 2
        line_type = cv.LINE_AA
        color = (0, 255, 0)
        dash_length = 10

        if selected_area_pen is not None:
            w = selected_area_pen.shape[1]
            h = selected_area_pen.shape[0]
        elif selected_area_highlight is not None:
            w = selected_area_highlight.shape[1]
            h = selected_area_highlight.shape[0]
        else:
            w = h = 0

        x_min, x_max = x_offset, x_offset + w
        y_min, y_max = y_offset, y_offset + h

        for i in range(x_min, x_max, dash_length * 2):
            cv.line(slide_image_copy, (i, y_min), (i + dash_length, y_min), color, thickness, line_type)
            cv.line(slide_image_copy, (i, y_max), (i + dash_length, y_max), color, thickness, line_type)
        for i in range(y_min, y_max, dash_length * 2):
            cv.line(slide_image_copy, (x_min, i), (x_min, i + dash_length), color, thickness, line_type)
            cv.line(slide_image_copy, (x_max, i), (x_max, i + dash_length), color, thickness, line_type)

    cv.rectangle(slide_image_copy, (750, 50), (790, 70), selected_color, -1)

    if mode == 'lecture':
        b, g, r, a = cv.split(pen_layer)
        overlay_color = cv.merge((b, g, r))
        mask_pen = a
        cv.copyTo(overlay_color, mask_pen, slide_image_copy)

        alpha_value = 0.4
        b_h, g_h, r_h, a_h = cv.split(highlight_layer)
        overlay_highlight = cv.merge((b_h, g_h, r_h))
        _, mask_highlight = cv.threshold(a_h, 0, 255, cv.THRESH_BINARY)
        slide_highlight_region = cv.bitwise_and(slide_image_copy, slide_image_copy, mask=mask_highlight)
        highlight_blend = cv.addWeighted(overlay_highlight, alpha_value, slide_highlight_region, 1 - alpha_value, 0)
        inv_mask_highlight = cv.bitwise_not(mask_highlight)
        slide_image_copy = cv.bitwise_and(slide_image_copy, slide_image_copy, mask=inv_mask_highlight)
        slide_image_copy = cv.add(slide_image_copy, highlight_blend)

        # temp_layer 적용 (현재 그리고 있는 선)
        b_t, g_t, r_t, a_t = cv.split(temp_layer)
        overlay_temp = cv.merge((b_t, g_t, r_t))
        mask_temp = a_t
        cv.copyTo(overlay_temp, mask_temp, slide_image_copy)

        pointer_gray = cv.cvtColor(pointer_layer, cv.COLOR_BGR2GRAY)
        _, mask_pointer = cv.threshold(pointer_gray, 254, 255, cv.THRESH_BINARY_INV)
        cv.copyTo(pointer_layer, mask_pointer, slide_image_copy)

        slide_image_copy[:tool_max_y, tool_margin_left:tool_max_x] = cv.addWeighted(
            tools, 0.7, slide_image_copy[:tool_max_y, tool_margin_left:tool_max_x], 0.3, 0
        )

        cv.putText(
            slide_image_copy,
            curr_tool,
            (30, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

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

    if summary_images:
        slide_display = cv.resize(slide_image_copy, (800, 600))
        summary_height = 200
        summary_width = 800
        summary_area = np.full((summary_height, summary_width, 3), 255, dtype=np.uint8)
        section_width = summary_width // 4

        separator = np.zeros((2, summary_width, 3), dtype=np.uint8)
        
        for idx, (img, size) in enumerate(zip(summary_images[-4:], image_sizes[-4:])):
            if img is None:
                continue
            h, w = size

            max_h = max(s[0] for s in image_sizes)
            max_w = max(s[1] for s in image_sizes)

            if h >= max_h or w >= max_w:
                scale = summary_height / h
                new_h = summary_height
                new_w = int(w * scale)
                if new_w > section_width:
                    scale = section_width / w
                    new_w = section_width
                    new_h = int(h * scale)
            else:
                scale = min(section_width/max_w, summary_height/max_h)
                new_w = int(w * scale)
                new_h = int(h * scale)

            resized_img = cv.resize(img, (new_w, new_h))
            x_offset = idx * section_width + (section_width - new_w) // 2
            y_offset = (summary_height - new_h) // 2
            summary_area[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        final_display = np.vstack([slide_display, separator, summary_area])
        cv.imshow("Lecture", final_display)
    else:
        cv.imshow("Lecture", slide_image_copy)

    cv.imshow("Webcam", frm)
    key = cv.waitKey(5) & 0xFF
    if key == 27:
        break
    
    if should_close_rotating_window:
        cv.destroyWindow("Rotating Image")
        should_close_rotating_window = False

cap.release()
cv.destroyAllWindows()
