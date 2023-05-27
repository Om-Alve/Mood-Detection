import cv2 as cv
import numpy as np
import dlib
import os

SHAPE_PREDICTOR_PATH = 'Data/shape_predictor_68_face_landmarks.dat'

cam = cv.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Cannot open webcam.")

predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
mode = input("How are you feeling?: ")
results = []
frames = []

while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_parts = landmarks.parts()
        for landmark in landmarks_parts:
            cv.circle(frame, (landmark.x, landmark.y), 2, (255, 0, 0), 3)

    cv.imshow('Frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        landmarks_arr = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks_parts])
        frames.append(landmarks_arr.flatten())
        results.append([mode])

    if key == ord('q'):
        break

data = np.hstack((results, frames))

if os.path.exists('Data/moods.npy'):
    old = np.load('Data/moods.npy')
    data = np.vstack((old, data))

np.save('Data/moods.npy', data)

cv.destroyAllWindows()
