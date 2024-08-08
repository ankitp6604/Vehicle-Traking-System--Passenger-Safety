# Vehicle-Traking-System-Driver-and-Passenger-Safety




This project implements a comprehensive real-time safety system for drivers and passengers. It includes features like driver distraction detection, drowsiness detection, yawn detection, face authentication, and scream detection, using computer vision and machine learning techniques.

## Features

- **Distraction Detection**: Monitors the driver's actions to identify any form of distraction, such as using a phone, eating, or interacting with the car's controls.
- **Drowsiness Detection**: Tracks the driver's facial landmarks to detect signs of drowsiness, focusing on eye closure and yawning.
- **Yawn Detection**: Counts the number of yawns, which can be a sign of fatigue or drowsiness.
- **Face Authentication**: Verifies the identity of the driver using facial recognition to ensure that the person behind the wheel is authorized to drive.
- **Scream Detection**: Detects screams within the car, potentially indicating an emergency situation.

## Datasets

- **Distraction Detection**: [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
- **Drowsiness Detection**: [Shape Predictor 68 Face Landmarks](https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat)

## Prerequisites

- Python 3.7+
- OpenCV
- Dlib
- TensorFlow Lite
- Numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ankitp6604/passenger-driver-safety-system.git
   cd passenger-driver-safety-system
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets from Kaggle and place them in the respective folders:
   - **Distraction Detection**: Place the State Farm dataset in `datasets/distraction/`.
   - **Drowsiness Detection**: Place the Shape Predictor file (`shape_predictor_68_face_landmarks.dat`) in `datasets/drowsiness/`.

## Python Files Overview

### 1. `distraction.py`

This script detects driver distraction using a pre-trained TensorFlow Lite model. It processes the video feed from a webcam and identifies actions that may indicate the driver is distracted, such as using a phone, eating, or looking away from the road.

- **Usage**: 
  ```bash
  python distraction.py
  ```

### 2. `drowsiness.py`

This script detects driver drowsiness by analyzing facial landmarks, particularly focusing on eye closure and yawning. It uses Dlib for facial landmark detection and calculates distances between landmarks to determine if the driver is drowsy.

- **Usage**: 
  ```bash
  python drowsiness.py
  ```

### 3. `yawn.py`

This script specifically detects and counts yawns from the driver's video feed. It uses facial landmarks to measure the distance between the upper and lower lips, identifying when the mouth is open wide enough to suggest yawning.

- **Usage**: 
  ```bash
  python yawn.py
  ```

### 4. `face_auth.py`

This script performs face authentication to ensure that only authorized drivers can operate the vehicle. It uses facial recognition techniques to compare the driver's face with a stored database of authorized users.

- **Usage**: 
  ```bash
  python face_auth.py
  ```

### 5. `scream.py`

This script detects screams within the vehicle, potentially indicating an emergency. It captures audio through the microphone and analyzes it to identify high-pitched sounds consistent with screaming.

- **Usage**: 
  ```bash
  python scream.py
  ```

## Usage

To use any of the safety features, simply run the corresponding script as described above. Each script will open a live video feed (or audio in the case of scream detection) and display real-time detections.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request with any enhancements or bug fixes.

