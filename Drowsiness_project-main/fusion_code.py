import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
import time

# ================= FPS CONTROL =================
TARGET_FPS = 10
FRAME_DURATION = 1.0 / TARGET_FPS
# ===============================================

# ---------- MEDIAPIPE ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

# ---------- THRESHOLDS ----------
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
SIDE_FACE_THRESHOLD = 0.035   # 👈 sideways sensitivity

CLOSED_FRAMES = 20
YAWN_FRAMES = 15

ear_counter = 0
mar_counter = 0
alarm_on = False

# ---------- AUDIO ----------
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alert.wav")

def start_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        alarm_sound.play(-1)

def stop_alarm():
    global alarm_on
    if alarm_on:
        alarm_sound.stop()
        alarm_on = False

# ---------- FUNCTIONS ----------
def eye_aspect_ratio(eye, lm, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth, lm, w, h):
    left = (int(lm[mouth[0]].x * w), int(lm[mouth[0]].y * h))
    top = (int(lm[mouth[1]].x * w), int(lm[mouth[1]].y * h))
    bottom = (int(lm[mouth[2]].x * w), int(lm[mouth[2]].y * h))
    right = (int(lm[mouth[3]].x * w), int(lm[mouth[3]].y * h))
    return distance.euclidean(top, bottom) / distance.euclidean(left, right)

def is_face_sideways(lm):
    nose_x = lm[1].x
    left_cheek = lm[234].x
    right_cheek = lm[454].x

    face_center = (left_cheek + right_cheek) / 2
    deviation = abs(nose_x - face_center)

    return deviation > SIDE_FACE_THRESHOLD

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
frame_count = 0
results = None

# ---------- MAIN LOOP ----------
while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    status_text = "No Face"

    if frame_count % 3 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

    if results and results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        ear = (eye_aspect_ratio(LEFT_EYE, lm, w, h) +
               eye_aspect_ratio(RIGHT_EYE, lm, w, h)) / 2.0
        mar = mouth_aspect_ratio(MOUTH, lm, w, h)

        ear_counter = ear_counter + 1 if ear < EAR_THRESHOLD else 0
        mar_counter = mar_counter + 1 if mar > MAR_THRESHOLD else 0

        # ========== DROWSY ==========
        if ear_counter >= CLOSED_FRAMES or mar_counter >= YAWN_FRAMES:
            status_text = "Drowsy"
            start_alarm()

        else:
            stop_alarm()

            # ========== DISTRACTED ==========
            if is_face_sideways(lm):
                status_text = "Distracted"
            else:
                status_text = "Active"

        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, "FPS: 10 (Locked)", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Driver Monitoring System", frame)

    elapsed = time.time() - loop_start
    if FRAME_DURATION - elapsed > 0:
        time.sleep(FRAME_DURATION - elapsed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
