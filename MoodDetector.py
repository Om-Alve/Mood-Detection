import numpy as np
import cv2 as cv
import dlib
from sklearn.neighbors import KNeighborsClassifier

# Load shape predictor and data
predictor = dlib.shape_predictor('Data/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
data = np.load('Data/moods.npy')

# Prepare the data
X = data[:, 1:].astype(int)
y = data[:, 0]

# Create and train the model
model = KNeighborsClassifier()
model.fit(X, y)

# Open video capture
cam = cv.VideoCapture(0)

# Process frames
while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmark = predictor(gray, face)
        landmarks = np.array([[point.x - face.left(), point.y - face.top()] for point in landmark.parts()]).flatten()
        pred = model.predict([landmarks])
        cv.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
        cv.putText(frame, pred[0], (face.left(), face.bottom() + 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release video capture and close windows
cam.release()
cv.destroyAllWindows()
