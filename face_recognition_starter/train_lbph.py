import argparse, os, json, cv2
import numpy as np

def load_images_and_labels(data_dir):
    images, labels, label_map = [], [], {}
    label_id = 0
    for person in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, person)
        if not os.path.isdir(d):
            continue
        if person not in label_map:
            label_map[person] = label_id
            label_id += 1
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(d, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(label_map[person])
    return images, np.array(labels, dtype=np.int32), label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--model', default='model_lbph.xml')
    ap.add_argument('--labels', default='labels.json')
    args = ap.parse_args()

    images, labels, label_map = load_images_and_labels(args.data_dir)
    if len(images) == 0:
        raise RuntimeError('Geen trainingsbeelden gevonden.')
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, labels)
    recognizer.save(args.model)
    with open(args.labels, 'w') as f:
        json.dump({v:k for k,v in label_map.items()}, f, indent=2)
    print(f'[i] Model -> {args.model}')
    print(f'[i] Labels -> {args.labels}')
    print(f'[i] Personen -> {list(label_map.keys())}')

if __name__ == '__main__':
    main()
