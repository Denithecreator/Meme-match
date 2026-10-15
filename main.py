"""Webcam -> hand gesture / face expression -> meme popup.

Run: python main.py
Quit: press 'q' in the camera window.

Tracks up to 2 hands at once. Add your own memes by editing memes_config.json:
  - gesture_rules: point "image" at a PNG, "rule" must be one of:
    thumbs_up, thumbs_down, peace, open_palm, fist, pointing, love_you (MediaPipe's
    built-in gesture recognizer, see gesture_map.py), plus call_me, middle_finger,
    finger_heart (BTS-style mini heart, one hand), full_heart, welcome_gesture,
    hands_on_head (two-hand combos), mewing, finger_in_mouth, hand_on_ear
    (one-hand + face combos) from custom_gesture.py.
  - face_rules: "rule" must be one of face_rules.py's outputs (mouth_open, smile, teeth,
    kiss, eyebrows_raised, wink, eyes_closed). "teeth" is a wide+open mouth geometry
    heuristic (grin), not literal tooth pixel detection. "kiss" uses landmark z-depth
    to catch lips pushed toward the camera (pout/duck-face).
  - custom_memes: give one or more reference photos of you making an expression;
    at runtime your live face is matched against those via landmark geometry
    (not identity), so any face shape works.
"""
import json
import os
import time
import ctypes
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options
import numpy as np

from face_rules import classify_expression
from gesture_map import to_rule
import custom_gesture
import custom_match

CONFIG_PATH = "memes_config.json"
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
FACE_MODEL_PATH = "models/face_landmarker.task"
CAM_WINDOW = "Camera"
MEME_WINDOW = "Meme Match"
MEME_SIZE = (480, 480)  # w, h
HISTORY_LEN = 7    # recent frames considered when deciding what's currently "shown"
CONFIRM_RATIO = 0.55  # fraction of HISTORY_LEN that must agree to confirm a match

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_or_placeholder(path, label):
    w, h = MEME_SIZE
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img, (w, h))
    img = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(img, label, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def blank_meme():
    w, h = MEME_SIZE
    img = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(img, "...", (w // 2 - 20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2, cv2.LINE_AA)
    return img


def screen_width():
    try:
        return ctypes.windll.user32.GetSystemMetrics(0)
    except Exception:
        return 1920


def build_lookup(config):
    """rule/name -> loaded meme image, plus custom face-match db."""
    lookup = {}
    for entry in config.get("gesture_rules", []):
        lookup[("gesture", entry["rule"])] = load_or_placeholder(entry.get("image"), entry["name"])
    for entry in config.get("face_rules", []):
        lookup[("face", entry["rule"])] = load_or_placeholder(entry.get("image"), entry["name"])

    custom_memes = config.get("custom_memes", [])
    custom_images = {
        m["name"]: load_or_placeholder(m.get("image"), m["name"]) for m in custom_memes
    }
    custom_db = custom_match.build_reference_db(custom_memes)
    return lookup, custom_images, custom_db


def image_for_match(match, lookup, custom_images):
    if match is None:
        return None
    kind, name = match
    if kind == "custom":
        return custom_images.get(name)
    return lookup.get((kind, name))


def draw_hand(frame, landmarks_norm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def draw_face(frame, landmarks_norm, w, h, stride=4):
    for i, p in enumerate(landmarks_norm):
        if i % stride:
            continue
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (80, 200, 80), -1)


def main():
    config = load_config()
    lookup, custom_images, custom_db = build_lookup(config)

    gesture_opts = vision.GestureRecognizerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cv2.namedWindow(CAM_WINDOW)
    cv2.namedWindow(MEME_WINDOW)
    cv2.moveWindow(CAM_WINDOW, 40, 60)
    cv2.moveWindow(MEME_WINDOW, min(screen_width() - MEME_SIZE[0] - 40, screen_width() // 2 + 40), 60)

    match_history = deque(maxlen=HISTORY_LEN)
    confirmed_match = None
    displayed = blank_meme()
    start_time = time.monotonic()

    with vision.GestureRecognizer.create_from_options(gesture_opts) as recognizer, \
         vision.FaceLandmarker.create_from_options(face_opts) as landmarker:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            gesture_res = recognizer.recognize_for_video(mp_img, timestamp_ms)
            face_res = landmarker.detect_for_video(mp_img, timestamp_ms)

            current_match = None
            current_image = None
            label = ""
            face_lm = face_res.face_landmarks[0] if face_res.face_landmarks else None
            if face_lm is not None:
                draw_face(frame, face_lm, w, h)

            hand_lms = gesture_res.hand_landmarks
            for hlm in hand_lms:
                draw_hand(frame, hlm, w, h)

            hand_labels = []
            per_hand_rule = [None] * len(hand_lms)
            per_hand_jaw = [None] * len(hand_lms)
            for i, hlm in enumerate(hand_lms):
                per_hand_jaw[i] = custom_gesture.classify_hand_with_face(hlm, face_lm)
                custom_name = custom_gesture.classify_hand(hlm, gesture_res.handedness[i][0].category_name)
                if custom_name:
                    per_hand_rule[i] = custom_name
                elif gesture_res.gestures[i]:
                    per_hand_rule[i] = to_rule(gesture_res.gestures[i][0].category_name)
                hand_labels.append(per_hand_jaw[i] or per_hand_rule[i] or "-")

            if hand_labels:
                cv2.putText(frame, "hands: " + " | ".join(hand_labels), (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

            # Priority: custom face match (most specific) > deliberate two-hand
            # combos > finger-on-jaw (mewing) > passive face expression > single-hand gesture.
            if face_lm is not None:
                custom_name = custom_match.match(face_lm, custom_db)
                if custom_name:
                    current_match = ("custom", custom_name)
                    current_image = custom_images.get(custom_name)
                    label = f"custom: {custom_name}"

            if current_match is None and len(hand_lms) == 2:
                combo = custom_gesture.classify_hands_on_head(hand_lms[0], hand_lms[1], face_lm)
                if not combo:
                    combo = custom_gesture.classify_two_hands(hand_lms[0], hand_lms[1])
                if combo:
                    current_match = ("gesture", combo)
                    current_image = lookup.get(("gesture", combo))
                    label = f"gesture: {combo}"

            if current_match is None:
                for jaw_rule in per_hand_jaw:
                    if jaw_rule:
                        current_match = ("gesture", jaw_rule)
                        current_image = lookup.get(("gesture", jaw_rule))
                        label = f"gesture: {jaw_rule}"
                        break

            if current_match is None and face_lm is not None:
                expr = classify_expression(face_lm)
                if expr:
                    current_match = ("face", expr)
                    current_image = lookup.get(("face", expr))
                    label = f"face: {expr}"

            if current_match is None:
                for rule in per_hand_rule:
                    if rule:
                        current_match = ("gesture", rule)
                        current_image = lookup.get(("gesture", rule))
                        label = f"gesture: {rule}"
                        break

            # Smoothing: a strict "N identical frames in a row" debounce breaks
            # under landmark jitter near a threshold boundary -- one stray None
            # (or a different rule) resets the streak and the gesture never
            # accumulates enough consecutive frames to display. A majority vote
            # over a rolling window tolerates that flicker while staying responsive.
            match_history.append(current_match)
            best_match, best_count = Counter(match_history).most_common(1)[0]
            if best_count / len(match_history) >= CONFIRM_RATIO and best_match != confirmed_match:
                confirmed_match = best_match
                img = image_for_match(confirmed_match, lookup, custom_images)
                displayed = img if img is not None else blank_meme()

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(CAM_WINDOW, frame)
            cv2.imshow(MEME_WINDOW, displayed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
