import argparse, os, cv2
from utils import detect_faces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--person', required=True, help='Naam/label van de persoon')
    ap.add_argument('--out', required=True, help='Output map voor face crops')
    ap.add_argument('--max', type=int, default=60, help='Max aantal beelden')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Kon de webcam niet openen.')

    print('[i] Start opname. SPACE = opslaan, q = stoppen.')
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)

        cv2.putText(frame, f'{args.person} | opgeslagen: {saved}/{args.max}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('capture_dataset', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(faces) == 0:
                print('[!] Geen gezicht gedetecteerd.')
                continue
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            fname = os.path.join(args.out, f'{args.person}_{saved:04d}.png')
            cv2.imwrite(fname, face)
            saved += 1
            print(f'[+] Saved {fname}')
            if saved >= args.max:
                print('[i] Max bereikt.')
                break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
