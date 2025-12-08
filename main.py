import cv2
import mediapipe as mp
import numpy as np


def drawing_mask(resized_mask, x, y):
    x_start = max(0, x)    # where?
    y_start = max(0, y)
    x_end = min(w, x + new_w)
    y_end = min(h, y + new_h)

    mask_x_start = max(0, -x)    # what?
    mask_y_start = max(0, -y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    mask = resized_mask[:, :, :3]
    alpha_mask = resized_mask[:, :, 3] / 255.0
    alpha = alpha_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    for color in range(3):
        flipped[y_start:y_end, x_start:x_end, color] = (1 - alpha) * flipped[y_start:y_end, x_start:x_end, color] + alpha * mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end, color]


dog_mask = cv2.imread('images/tongue.png', cv2.IMREAD_UNCHANGED)    # loading with transparency
detective_mask = cv2.imread('images/detective.png', cv2.IMREAD_UNCHANGED)
christmas_mask = cv2.imread('images/hat.png', cv2.IMREAD_UNCHANGED)
laser_mask = cv2.imread('images/laser.png', cv2.IMREAD_UNCHANGED)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
min_mouth_distance = 0.01
basic_eye_distance = 100
min_eye_distance = 0.3
max_eye_distance = 0.33

while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    h, w, _ = flipped.shape
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(flippedRGB)

    if results.multi_face_landmarks is not None:
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

    cv2.imshow("AnnChat", flipped)

cv2.destroyAllWindows()
