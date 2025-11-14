import argparse, json, cv2
from utils import detect_faces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='model_lbph.xml')
    ap.add_argument('--labels', default='labels.json')
    ap.add_argument('--threshold', type=float, default=70.0, help='Max afstand (lager = strenger)')
    args = ap.parse_args()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    with open(args.labels) as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Kon de webcam niet openen.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label_id, distance = recognizer.predict(face)

            if distance <= args.threshold:
                name = id_to_name.get(label_id, f'ID {label_id}')
                text = f'{name} ({distance:.1f})'
                color = (0, 255, 0)
            else:
                text = f'Onbekend ({distance:.1f})'
                color = (0, 0, 255)

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('recognize_realtime', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
