# Gezichtsherkenning (LBPH + OpenCV)

Dit is een eenvoudig, lokaal werkend gezichtsherkenningsproject in drie stappen:

1) **Dataset opnemen** (`capture_dataset.py`)
2) **Model trainen** (`train_lbph.py`)
3) **Realtime herkennen** (`recognize_realtime.py`)

> **Let op (AVG/GDPR):** Gebruik dit alleen met expliciete toestemming van betrokkenen, bewaar data versleuteld, documenteer bewaartermijnen en doelbinding.

## Installatie
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Gebruik

### 1) Dataset opnemen
```bash
python capture_dataset.py --person "Adil" --out data/Adil --max 60
```

### 2) Model trainen
```bash
python train_lbph.py --data_dir data --model model_lbph.xml --labels labels.json
```

### 3) Realtime herkennen
```bash
python recognize_realtime.py --model model_lbph.xml --labels labels.json --threshold 70
```

## Tips
- Neem 50–150 beelden per persoon op, met variatie in licht/pose.
- `--threshold` lager = strenger (minder false positives).

## Beperkingen
- LBPH is eenvoudig/snel maar minder nauwkeurig dan deep learning embeddings.
- Voor hogere nauwkeurigheid: FaceNet/ArcFace/`face_recognition`.
