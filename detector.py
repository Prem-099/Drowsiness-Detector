import cv2
import dlib
from scipy.spatial import distance
import datetime
import logging
import pygame

# ---------- SETUP ----------

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Left and Right eye landmark index ranges
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Initialize counters and logging
blink_counter = 0
drowsy_counter = 0
logging.basicConfig(filename='drowsiness_log.txt', level=logging.INFO)

# ---------- START CAMERA ----------
cap = cv2.VideoCapture(0)
pygame.mixer.init()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Get eye regions
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw facial landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # EAR below threshold -> possible blink or drowsy
        if ear < 0.23:
            drowsy_counter += 1
            if drowsy_counter >= 15:
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                logging.info(f"Drowsiness detected at {datetime.datetime.now()}")
                try:
                    pygame.mixer.music.load("warning-alarm-loop-1-279206.mp3")
                    pygame.mixer.music.play()
                except:
                    pass
        else:
            if 2 <= drowsy_counter < 15:
                blink_counter += 1  # counted as a blink
            drowsy_counter = 0

        # Show blink count
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()