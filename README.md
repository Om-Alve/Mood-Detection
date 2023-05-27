# Mood Detection Project

This project consists of two Python scripts: `MoodTrainer.py` and `MoodDetector.py`. The `MoodTrainer.py` script is used to collect facial landmark data along with the corresponding mood labels, while the `MoodDetector.py` script uses the collected data to detect and classify moods in real-time.

## Requirements

To run the scripts, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- dlib
- scikit-learn

## Usage

### MoodTrainer.py

The `MoodTrainer.py` script is used to collect facial landmark data and labels for training the mood detection model. To run the script, follow these steps:

1. Make sure you have the necessary dependencies installed.

2. Place the `shape_predictor_68_face_landmarks.dat` file in the `Data` directory.

3. Open a terminal or command prompt and navigate to the directory containing the `MoodTrainer.py` file.

4. Run the script using the command: `python MoodTrainer.py`

5. When prompted, enter your current mood.

6. The script will open a webcam feed, detect facial landmarks, and display them on the screen. Press the 'c' key to capture a frame and record the corresponding facial landmarks and mood label.

7. Repeat the capture process for as many frames as desired. Press the 'q' key to stop capturing frames and exit the script.

8. The captured data will be saved in the `Data/moods.npy` file. If the file already exists, the new data will be appended to the existing data.

### MoodDetector.py

The `MoodDetector.py` script is used to detect and classify moods in real-time using the trained model. To run the script, follow these steps:

1. Make sure you have the necessary dependencies installed.

2. Place the `shape_predictor_68_face_landmarks.dat` file in the `Data` directory.

3. Open a terminal or command prompt and navigate to the directory containing the `MoodDetector.py` file.

4. Run the script using the command: `python MoodDetector.py`

5. The script will open a webcam feed and start detecting and classifying moods in real-time. The detected mood will be displayed on the screen for each detected face.

6. Press the 'q' key to stop the script and exit.

