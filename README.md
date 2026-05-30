# Face Recognition — Employee Check-in System

Real-time face recognition that automatically clocks employees in and out using an office camera. It detects faces, figures out who's who, logs their attendance, and greets them by name.

https://user-images.githubusercontent.com/83819024/217428805-baf58624-2f68-4c48-a434-0d7f6f9afd88.mp4

---

## What it does

An RTSP camera watches the office entrance. When someone walks by:

- Their face is detected and tracked across frames
- The system matches them against a database of registered employees
- If it's their first sighting today → **check-in**. Second time → **checkout**
- A Vietnamese voice greets them: *"Xin chào Anh Dũng"* in the morning, *"Tạm biệt Anh Dũng"* in the afternoon
- Each event is logged to SQLite with a timestamp, accuracy score, and snapshot

The whole pipeline runs in real time on a single machine.

---

## How it's built

**Face detection** — SSD-based detector (OpenCV DNN) scans each frame for faces. A separate model checks whether the person is wearing a mask (masked faces get a stricter matching threshold).

**Tracking** — A custom centroid tracker keeps IDs consistent across frames. Without tracking, the same person would be re-identified on every frame and logged multiple times. Faces that disappear for 10+ frames are deregistered.

**Recognition** — Detected faces are resized to 224×224, passed through a VGGFace2/ResNet50 model, and converted to 2048-dimensional embeddings. The embedding is compared against all stored employee embeddings using cosine distance — the closest match below threshold wins.

**Attendance logic** — Simple: no checkin today? → insert checkin. Checkin exists but no checkout? → insert checkout. Already both? → update checkout timestamp. This handles people coming and going throughout the day.

**Voice** — Windows SAPI with a Vietnamese voice. Checks the time: before noon = "Xin chào", after noon = "Tạm biệt".

---

## Project layout

```
.
├── main_track_identify.py      # Entry point — camera → pipeline
├── packages/
│   ├── detectFaceCNN4.py       # Face detection + mask check
│   ├── tracking_objects.py     # Tracker, matching, DB logging, voice alert
│   ├── findFace2.py            # Cosine similarity search
│   ├── getEmbeddings2.py       # Pre-compute embeddings from dataset/
│   ├── identifyFace.py         # VGGFace2 embedding model
│   ├── alertCheck.py           # Vietnamese TTS greeting
│   ├── insert_information2.py  # SQLite check-in/out logic
│   ├── add_staff_information.py
│   └── postAlert.py            # Optional: push to external API
├── database/data_base.sql      # SQLite (checkin, checkout, staff info)
├── dataset/                    # Employee photos, one folder per person
├── embeddingNPY/               # Pre-computed embeddings + staff codes
├── Models/                     # SSD, mask detector, Haar cascade
├── model_embedding/            # VGGFace2/ResNet50 saved model
└── requirements.txt
```

---

## Getting started

```bash
pip install -r requirements.txt
```

**1. Register employees**

Drop 1–3 face photos of each person into `dataset/{staff_code}/`. The folder name is their employee ID.

Then generate embeddings:

```bash
python -c "from packages.getEmbeddings2 import get_embedding; get_embedding()"
```

**2. Add staff info**

Insert names and positions into the `information_staff` table in the SQLite database.

**3. Point it at a camera**

Edit the `src` variable in `main_track_identify.py` — RTSP URL for an IP camera, or `0` for a webcam.

**4. Run**

```bash
python main_track_identify.py
```

---

## Tech stack

| Layer | What |
|---|---|
| Detection | OpenCV DNN + SSD Caffe model |
| Mask check | Custom MobileNetV2-based classifier |
| Embeddings | VGGFace2 / ResNet50 (TensorFlow) |
| Matching | Cosine distance over 2048-dim vectors |
| Tracking | Custom centroid tracker (distance + IoU) |
| Database | SQLite |
| Voice | pyttsx3 + Windows Vietnamese SAPI |
| Runtime | Python 3.8+, TensorFlow < 2.11 |

---

## Notes

- The mask detection score adjusts the recognition threshold — if someone's wearing a mask, the system requires a closer embedding match before logging them.
- Voice greetings need a Vietnamese TTS voice installed on Windows. On Linux/Mac you'd swap the SAPI engine for something else.
- There's an optional hook to post attendance events to 1Office (`postAlert.py`) — disabled by default.
