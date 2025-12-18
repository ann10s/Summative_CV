import cv2
import mediapipe as mp
import numpy as np


def clicked(x_mouse, y_mouse, x, y, w, h):
    return x <= x_mouse <= x + w and y <= y_mouse <= y + h


def click_event(event, x_mouse, y_mouse, flags, params):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        if state == "menu":
            if clicked(x_mouse, y_mouse, 200, 300, 240, 70):
                state = "running"
            elif clicked(x_mouse, y_mouse, 560, 20, 60, 60):
                state = "tutorial"
        elif state == "tutorial":
            if clicked(x_mouse, y_mouse, 10, 10, 150, 40):
                state = "menu"
        else:
            if clicked(x_mouse, y_mouse, 10, 10, 150, 40):
                state = "menu"


def drawing_button(bg, x, y, w, h):
    cv2.rectangle(bg, (x, y), (x + w, y + h), (0, 189, 207), -1)    # filling
    cv2.rectangle(bg, (x, y), (x + w, y + h), (255, 255, 255), 2)


def drawing_mask(resized_mask, x, y):
    x_start = max(0, x)    # where?
    y_start = max(0, y)
    x_end = min(w, x + new_w)
    y_end = min(h, y + new_h)

    mask_x_start = max(0, -x)    # which part?
    mask_y_start = max(0, -y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    mask = resized_mask[:, :, :3]
    alpha_mask = resized_mask[:, :, 3] / 255.0
    alpha = alpha_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    for color in range(3):
        flipped[y_start:y_end, x_start:x_end, color] = (1 - alpha) * flipped[y_start:y_end, x_start:x_end, color] + alpha * mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end, color]


dog_mask = cv2.imread("images/tongue.png", cv2.IMREAD_UNCHANGED)    # loading with transparency
detective_mask = cv2.imread("images/detective.png", cv2.IMREAD_UNCHANGED)
christmas_mask = cv2.imread("images/hat.png", cv2.IMREAD_UNCHANGED)
laser_mask = cv2.imread("images/laser.png", cv2.IMREAD_UNCHANGED)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

min_mouth_distance = 0.01
basic_eye_distance = 100
min_eye_distance = 0.3
max_eye_distance = 0.35
max_head_tilt = 0.07

screenshot_counter = 0
frame_counter = 0
prev = None
now = None
state = "menu"

while(cap.isOpened()):
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or not ret:
        break

    flipped = np.fliplr(frame).copy()
    h, w, _ = flipped.shape
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(flippedRGB)

    if state == "menu":    # main menu
        hsv = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] // 4    # darkened background
        dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.putText(dark, "AnnChat", (190, 100), font, 2, (255, 255, 255), 3)
        drawing_button(dark, 560, 20, 60, 60)
        cv2.putText(dark, "i", (583, 69), font, 1.7, (255, 255, 255), 4)
        drawing_button(dark, 200, 300, 240, 70)
        cv2.putText(dark, "Start", (280, 345), font, 1, (255, 255, 255), 2)
        cv2.putText(dark, "Change your facial expression", (150, 160), font, 0.7, (255, 255, 255, 2))    # description
        cv2.putText(dark, "and see what happens. If you", (150, 190), font, 0.7, (255, 255, 255, 2))
        cv2.putText(dark, "like it, you can take a screenshot", (150, 220), font, 0.7, (255, 255, 255, 2))
        cv2.putText(dark, "by pressing the 'S' key. Good luck!", (150, 250), font, 0.7, (255, 255, 255, 2))
        cv2.imshow("AnnChat", dark)
        cv2.setMouseCallback("AnnChat", click_event)

    elif state == "tutorial":    # tutorial
        hsv = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] // 4    # darkened background
        dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.putText(dark, "How to play", (180, 110), font, 1.5, (255, 255, 255), 3)
        cv2.putText(dark, "* Opened mouth - dog mask", (130, 170), font, 0.8, (255, 255, 255), 1)
        cv2.putText(dark, "* Squinted eyes - detective hat", (120, 210), font, 0.8, (255, 255, 255), 1)
        cv2.putText(dark, "* Bulging eyes - lasers from eyes", (110, 250), font, 0.8, (255, 255, 255), 1)
        cv2.putText(dark, "* Normal expression - christmas hat", (100, 290), font, 0.8, (255, 255, 255), 1)
        cv2.putText(dark, "Keep your head straight!", (170, 330), font, 0.8, (55, 22, 255), 2)
        cv2.putText(dark, "Press 'S' to take a screenshot", (130, 370), font, 0.8, (224, 171, 79), 2)
        drawing_button(dark, 10, 10, 150, 40)    # back to main menu
        cv2.putText(dark, "Main menu", (16, 40), font, 0.8, (255, 255, 255), 2)
        cv2.imshow("AnnChat", dark)

    else:    # running
        if results.multi_face_landmarks is not None:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_eye_x, left_eye_y = face_landmarks[33].x * w, face_landmarks[33].y * h
            right_eye_x, right_eye_y = face_landmarks[263].x * w, face_landmarks[263].y * h
            eye_distance = np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)
            straight_head = abs(left_eye_y - right_eye_y) < max_head_tilt * (eye_distance / basic_eye_distance) * h

            if straight_head:    # head tilt check
                mouth_top = face_landmarks[13].y
                mouth_bottom = face_landmarks[14].y
                mouth_distance = abs(mouth_top - mouth_bottom)

                left_top_x, left_top_y = face_landmarks[386].x * w, face_landmarks[386].y * h
                left_bottom_x, left_bottom_y = face_landmarks[374].x * w, face_landmarks[374].y * h
                left_eye_h_distance = np.sqrt((left_bottom_x - left_top_x) ** 2 + (left_bottom_y - left_top_y) ** 2)

                left_l_x, left_l_y = face_landmarks[359].x * w, face_landmarks[359].y * h
                left_r_x, left_r_y = face_landmarks[362].x * w, face_landmarks[362].y * h
                left_eye_w_distance = np.sqrt((left_l_x - left_r_x) ** 2 + (left_r_y - left_l_y) ** 2)

                left_eye_distance = left_eye_h_distance / left_eye_w_distance

                right_top_x, right_top_y = face_landmarks[159].x * w, face_landmarks[159].y * h
                right_bottom_x, right_bottom_y = face_landmarks[145].x * w, face_landmarks[145].y * h
                right_eye_h_distance = np.sqrt((right_bottom_x - right_top_x) ** 2 + (right_bottom_y - right_top_y) ** 2)

                right_l_x, right_l_y = face_landmarks[133].x * w, face_landmarks[133].y * h
                right_r_x, right_r_y = face_landmarks[130].x * w, face_landmarks[130].y * h
                right_eye_w_distance = np.sqrt((right_l_x - right_r_x) ** 2 + (right_l_y - right_r_y) ** 2)

                right_eye_distance = right_eye_h_distance / right_eye_w_distance

                if mouth_distance > min_mouth_distance:    # mouth is open
                    now = "dog"
                elif left_eye_distance < min_eye_distance and right_eye_distance < min_eye_distance:    # eyes are narrowed
                    now = "detective"
                elif left_eye_distance > max_eye_distance and right_eye_distance > max_eye_distance:    # eyes open wide
                    now = "laser"
                else:
                    now = "christmas"
            else:
                cv2.putText(flipped, "Keep your head straight!", (120, 430), font, 1, (255, 255, 255), 5)    # warning!
                cv2.putText(flipped, "Keep your head straight!", (120, 430), font, 1, (55, 22, 255), 2)

        if now == prev:
            frame_counter += 1
        else:
            frame_counter = 0
            prev = now

        if frame_counter >= 5 and results.multi_face_landmarks is not None and straight_head:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_eye_x, left_eye_y = face_landmarks[33].x * w, face_landmarks[33].y * h
            right_eye_x, right_eye_y = face_landmarks[263].x * w, face_landmarks[263].y * h
            eye_distance = np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)

            mouth_top = face_landmarks[13].y
            mouth_bottom = face_landmarks[14].y
            mouth_distance = abs(mouth_top - mouth_bottom)

            left_top_x, left_top_y = face_landmarks[386].x * w, face_landmarks[386].y * h
            left_bottom_x, left_bottom_y = face_landmarks[374].x * w, face_landmarks[374].y * h
            left_eye_h_distance = np.sqrt((left_bottom_x - left_top_x) ** 2 + (left_bottom_y - left_top_y) ** 2)

            left_l_x, left_l_y = face_landmarks[359].x * w, face_landmarks[359].y * h
            left_r_x, left_r_y = face_landmarks[362].x * w, face_landmarks[362].y * h
            left_eye_w_distance = np.sqrt((left_l_x - left_r_x) ** 2 + (left_r_y - left_l_y) ** 2)

            left_eye_distance = left_eye_h_distance / left_eye_w_distance

            right_top_x, right_top_y = face_landmarks[159].x * w, face_landmarks[159].y * h
            right_bottom_x, right_bottom_y = face_landmarks[145].x * w, face_landmarks[145].y * h
            right_eye_h_distance = np.sqrt((right_bottom_x - right_top_x) ** 2 + (right_bottom_y - right_top_y) ** 2)

            right_l_x, right_l_y = face_landmarks[133].x * w, face_landmarks[133].y * h
            right_r_x, right_r_y = face_landmarks[130].x * w, face_landmarks[130].y * h
            right_eye_w_distance = np.sqrt((right_l_x - right_r_x) ** 2 + (right_l_y - right_r_y) ** 2)

            right_eye_distance = right_eye_h_distance / right_eye_w_distance

            if mouth_distance > min_mouth_distance:    # mouth is open
                factor = np.clip(0.4 * (eye_distance / basic_eye_distance), 0.1, 0.9)
                shift = int(factor / 0.02)

                mask_h, mask_w = dog_mask.shape[:2]
                new_w = int(mask_w * factor)
                new_h = int(mask_h * factor)
                resized_mask = cv2.resize(dog_mask, (new_w, new_h))

                nose_x = int(face_landmarks[4].x * w)
                nose_y = int(face_landmarks[4].y * h)
                x = nose_x - new_w // 2 - shift
                y = nose_y - new_h // 2

                drawing_mask(resized_mask, x, y)

            elif left_eye_distance < min_eye_distance and right_eye_distance < min_eye_distance:    # eyes are narrowed
                factor = np.clip(0.2 * (eye_distance / basic_eye_distance), 0.05, 0.95)
                shift = int(factor / 0.02)

                mask_h, mask_w = detective_mask.shape[:2]
                new_w = int(mask_w * factor)
                new_h = int(mask_h * factor)
                resized_mask = cv2.resize(detective_mask, (new_w, new_h))

                forehead_x = int(face_landmarks[151].x * w)
                forehead_y = int(face_landmarks[151].y * h)
                x = forehead_x - new_w // 2 - int(shift * 0.2)
                y = forehead_y - new_h // 2 - shift * 8

                drawing_mask(resized_mask, x, y)

            elif left_eye_distance > max_eye_distance and right_eye_distance > max_eye_distance:    # eyes open wide
                factor = np.clip(1.7 * (eye_distance / basic_eye_distance), 0.5, 7)
                shift = int(factor / 0.02)

                mask_h, mask_w = laser_mask.shape[:2]
                new_w = int(mask_w * factor)
                new_h = int(mask_h * factor)
                resized_mask = cv2.resize(laser_mask, (new_w, new_h))

                nose_x = int(face_landmarks[4].x * w)
                nose_y = int(face_landmarks[4].y * h)
                x = nose_x - new_w // 2
                y = nose_y - new_h // 2 + int(shift * 1.2)

                drawing_mask(resized_mask, x, y)

            else:
                factor = np.clip(0.3 * (eye_distance / basic_eye_distance), 0.05, 0.95)
                shift = int(factor / 0.02)

                mask_h, mask_w = christmas_mask.shape[:2]
                new_w = int(mask_w * factor)
                new_h = int(mask_h * factor)
                resized_mask = cv2.resize(christmas_mask, (new_w, new_h))

                forehead_x = int(face_landmarks[151].x * w)
                forehead_y = int(face_landmarks[151].y * h)
                x = forehead_x - new_w // 2 + shift * 3
                y = forehead_y - new_h // 2 - shift

                drawing_mask(resized_mask, x, y)

        if key == ord("s"):
            if frame_counter >= 5 and results.multi_face_landmarks is not None and straight_head:
                filename = f"screenshot_{screenshot_counter}.png"
                cv2.imwrite(filename, flipped)
                screenshot_counter += 1
                print("📸 The screenshot was saved successfully")
            else:
                print("⚠️ Cannot save: the head is not straight or the mask is not stable")

        drawing_button(flipped, 10, 10, 150, 40)    # back to main menu
        cv2.putText(flipped, "Main menu", (16, 40), font, 0.8, (255, 255, 255), 2)
        cv2.imshow("AnnChat", flipped)

cv2.destroyAllWindows()
