"""Download the two MediaPipe task bundles into models/.

Run: python fetch_models.py
"""
import os
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "models/gesture_recognizer.task": f"{BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "models/face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def main():
    os.makedirs("models", exist_ok=True)
    for path, url in MODELS.items():
        if os.path.exists(path):
            print(f"skip {path} (exists)")
            continue
        print(f"downloading {path} ...")
        urllib.request.urlretrieve(url, path)
    print("done")


if __name__ == "__main__":
    main()
