# pip install opencv-contrib-python
import cv2
import os

# Start webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Load Haar Cascade (inside haarcascade/ folder)
face_detector = cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')

# Numeric face ID
face_id = input('\n Enter user ID and press <return>: ')

print("\n Initializing face capture...")
count = 0  # sampling face count

# Start face detection and capture 30 images
while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save captured image into the dataset folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg",
                    gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  
    if k == 27:  # ESC to exit
        break
    elif count >= 30:  # stop after 30 samples
        break

print("\n Exiting program and cleaning up…")
cam.release()
cv2.destroyAllWindows()
