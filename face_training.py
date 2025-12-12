import cv2
import numpy as np  # pip install numpy
from PIL import Image  # pip install pillow
import os

# Path for face image database
path = 'dataset'

# Load LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade from haarcascade/ folder
detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Function to get images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("\n[INFO] Training faces. This may take a few seconds...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.write('trainer/trainer.yml')

print("\n[INFO] {0} faces trained. Exiting program.".format(len(np.unique(ids))))
