import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
# ========== MEDIAPIPE ==========
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# ========== LANDMARKS ==========
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

# ========== THRESHOLDS ==========
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CLOSED_FRAMES = 20
YAWN_FRAMES = 15

ear_counter = 0
mar_counter = 0
alarm_on = False

# ========== INITIALIZE PYGAME ==========
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alert.wav")

# ========== FUNCTIONS ==========
def eye_aspect_ratio(eye, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth, landmarks, w, h):
    left = (int(landmarks[mouth[0]].x * w), int(landmarks[mouth[0]].y * h))
    top = (int(landmarks[mouth[1]].x * w), int(landmarks[mouth[1]].y * h))
    bottom = (int(landmarks[mouth[2]].x * w), int(landmarks[mouth[2]].y * h))
    right = (int(landmarks[mouth[3]].x * w), int(landmarks[mouth[3]].y * h))

    vertical = distance.euclidean(top, bottom)
    horizontal = distance.euclidean(left, right)
    return vertical / horizontal

def start_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        alarm_sound.play(-1)  # -1 means loop continuously

def stop_alarm():
    global alarm_on
    if alarm_on:
        alarm_sound.stop()
        alarm_on = False

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        ear = (
            eye_aspect_ratio(LEFT_EYE, landmarks, w, h) +
            eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
        ) / 2.0

        mar = mouth_aspect_ratio(MOUTH, landmarks, w, h)

        # ===== EAR LOGIC =====
        if ear < EAR_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0

        # ===== MAR LOGIC =====
        if mar > MAR_THRESHOLD:
            mar_counter += 1
        else:
            mar_counter = 0

        # ===== FINAL DECISION =====
        if ear_counter >= CLOSED_FRAMES or mar_counter >= YAWN_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            start_alarm()
        else:
            stop_alarm()  # stop immediately if person is alert

        # ===== DISPLAY =====
        cv2.putText(frame, f"EAR: {ear:.2f}", (300,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
