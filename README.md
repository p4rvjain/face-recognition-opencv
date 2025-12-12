# Face Recognition - OpenCV

This project is a real-time **Face Detection and Face Recognition system** built using **OpenCV**, **LBPH Face Recognizer**, and **Haar Cascade classifiers**.

The system includes:
- Creating a face dataset using a webcam  
- Training a recognition model  
- Running real-time face recognition  

---

## Features
- Live face detection with Haar Cascades  
- Automatic dataset generation  
- LBPH-based face recognition  
- Confidence score display  
- Simple, fast, and runs offline  

---

## Project Structure

```
face-recognition-opencv/
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── dataset/
│   └── (will be filled automatically when running face_dataset.py)
│
├── trainer/
│   └── trainer.yml (auto-generated after training)
│
├── face_dataset.py
├── face_training.py
├── face_recognition.py
├── .gitignore
└── README.md
```

---

## Installation & Requirements


Ensure your webcam is connected.

Install the required Python packages:

pip install opencv-contrib-python numpy pillow

---

## Running the Project

### 1. Capture Face Dataset
Run: `python face_dataset.py`

This will:
- Open your webcam  
- Detect your face  
- Capture and save 30 face samples to `dataset/`  
- Each image is saved as: `dataset/User.<ID>.<sampleNum>.jpg`

### 2. Train the LBPH Model
Run: `python face_training.py`

This will:
- Read all images from `dataset/`  
- Extract faces and labels  
- Train the LBPH face recognizer  
- Save the trained model as: `trainer/trainer.yml`

### 3. Run Real-Time Face Recognition
Run: `python face_recognition.py`

The program will:
- Detect faces on your webcam feed  
- Predict the user’s name  
- Show the confidence percentage  
- Display face bounding boxes in real time
