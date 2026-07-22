"""Custom meme matching: compare live face expression to reference photos.

Since it's the same person's face each time, we don't do face *identity*
matching -- we build a small geometric feature vector (normalized distances
between key landmarks) that captures *expression*, and nearest-neighbor it
against vectors extracted once from each reference photo.
"""
import math
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options

FACE_MODEL_PATH = "models/face_landmarker.task"

# (landmark_a, landmark_b) pairs describing mouth/eye/brow/jaw shape,
# normalized by inter-ocular distance so it's scale/translation invariant.
FEATURE_PAIRS = [
    (61, 291),   # mouth width
    (13, 14),    # mouth height (inner)
    (78, 308),   # mouth corners (outer)
    (159, 145),  # left eye height
    (386, 374),  # right eye height
    (105, 159),  # left brow to eye
    (334, 386),  # right brow to eye
    (10, 152),   # forehead to chin (face length)
    (234, 454),  # cheek to cheek (face width)
    (4, 152),    # nose tip to chin
]
L_EYE_OUT, R_EYE_OUT = 33, 263


def feature_vector(lm):
    scale = math.hypot(lm[L_EYE_OUT].x - lm[R_EYE_OUT].x, lm[L_EYE_OUT].y - lm[R_EYE_OUT].y) or 1e-6
    return [
        math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y) / scale
        for a, b in FEATURE_PAIRS
    ]


def _distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def build_reference_db(custom_memes):
    """custom_memes: list of {"name": str, "refs": [image paths]}.
    Returns list of (meme_name, feature_vector) for every ref photo that had a detectable face.
    """
    db = []
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        for meme in custom_memes:
            for path in meme.get("refs", []):
                img = cv2.imread(path)
                if img is None:
                    print(f"[warn] could not read ref photo: {path}")
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if not result.face_landmarks:
                    print(f"[warn] no face found in ref photo: {path}")
                    continue
                lm = result.face_landmarks[0]
                db.append((meme["name"], feature_vector(lm)))
    return db


def match(lm, db, threshold=0.35):
    """Return best-matching meme name from db, or None if nothing close enough."""
    if not db:
        return None
    live_vec = feature_vector(lm)
    best_name, best_dist = None, float("inf")
    for name, ref_vec in db:
        d = _distance(live_vec, ref_vec)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name if best_dist < threshold else None
