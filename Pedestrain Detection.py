import cv2
import numpy as np
import os

body_classifier_path = r'C:/Users/Neha/OneDrive/Desktop/VSCode/OpenCV/Haarcascade/haarcascade_fullbody.xml'

if not os.path.exists(body_classifier_path):
    print(f"Error: The classifier file does not exist at {body_classifier_path}")
    exit()

body_classifier = cv2.CascadeClassifier(body_classifier_path)

if body_classifier.empty():
    print("Error: Could not load the body classifier. Make sure the XML file is valid and accessible.")
    exit()

video_path = r'C:\Users\Neha\OneDrive\Desktop\VSCode\OpenCV\Crowd.mp4'

if not os.path.exists(video_path):
    print(f"Error: The video file does not exist at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

print("Video opened successfully. Starting pedestrian detection...")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame from video. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: 
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
