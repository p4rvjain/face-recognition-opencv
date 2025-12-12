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
face-recognition-opencv/
│
├── haarcascade/
│ └── haarcascade_frontalface_default.xml
│
├── dataset/
│ └── (will be filled automatically when running face_dataset.py)
│
├── trainer/
│ └── trainer.yml (auto-generated after training)
│
├── face_dataset.py
├── face_training.py
├── face_recognition.py
├── .gitignore
└── README.md

---

## Installation & Requirements


Ensure your webcam is connected.

Install the required Python packages:

pip install opencv-contrib-python numpy pillow

---

