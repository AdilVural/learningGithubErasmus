import cv2

def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def detect_faces(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)):
    face_cascade = get_face_detector()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces
