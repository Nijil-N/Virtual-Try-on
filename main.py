import cv2
import mediapipe as mp
import numpy as np
import os

def load_glasses_images(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_UNCHANGED)
            images.append(img)
    return images

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay transparent PNG onto background image"""
    bg = background.copy()

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_rgb.shape
    if y + h > bg.shape[0] or x + w > bg.shape[1] or x < 0 or y < 0:
        return bg  # Skip if overlay goes out of bounds

    roi = bg[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_rgb, overlay_rgb, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    bg[y:y+h, x:x+w] = dst

    return bg

def main():
    glasses_images = load_glasses_images('glasses')
    current_glasses_idx = 0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    print("\nPress 'n' to switch glasses | Press 'q' to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Get head landmarks (temples)
                left_temple = face_landmarks.landmark[234]
                right_temple = face_landmarks.landmark[454]

                h, w, _ = frame.shape

                x1 = int(left_temple.x * w)
                y1 = int(left_temple.y * h)
                x2 = int(right_temple.x * w)
                y2 = int(right_temple.y * h)

                head_width = int(1.2 * np.linalg.norm([x2 - x1, y2 - y1]))

                # Position: slightly above nose bridge (between eyes)
                nose = face_landmarks.landmark[168]  # bridge of nose
                center_x = int(nose.x * w)
                center_y = int(nose.y * h)

                top_left_x = center_x - head_width // 2
                top_left_y = center_y - head_width // 4   # reduced upward shift

                top_left_x = max(0, top_left_x)
                top_left_y = max(0, top_left_y)

                glasses_img = glasses_images[current_glasses_idx]

                # Maintain aspect ratio
                aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
                glasses_height = int(head_width * aspect_ratio)

                try:
                    frame = overlay_transparent(frame, glasses_img, top_left_x, top_left_y, overlay_size=(head_width, glasses_height))
                except:
                    pass  # skip overlay if size invalid

        cv2.putText(frame, f'Glasses #{current_glasses_idx + 1}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Virtual Glasses Try-On', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_glasses_idx = (current_glasses_idx + 1) % len(glasses_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()