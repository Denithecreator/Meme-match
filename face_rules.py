"""Rule-based face expression classification from MediaPipe FaceMesh landmarks.

Thresholds are heuristic and may need tuning per-face/lighting (constants below).
"""
import math

# Tunable thresholds
MOUTH_OPEN_RATIO = 0.055     # inner-mouth height / face height
SMILE_WIDTH_RATIO = 0.42     # mouth width / face width
EAR_CLOSED = 0.18            # eye-aspect-ratio below this = closed
BROW_RAISE_RATIO = 0.16      # brow-to-eye distance / face height
TEETH_MIN_OPEN_RATIO = 0.02  # min inner-mouth height / face height to count as "open" for a grin
TEETH_ASPECT_MIN = 2.3       # mouth width/height ratio above this reads as a wide grin, not a round gasp
KISS_MAX_WIDTH_RATIO = 0.32  # mouth width / face width must be narrower than this to count as puckered
KISS_PROTRUSION_RATIO = 0.04 # how far the lips push toward the camera vs. the nose bridge, / face height
SAD_DROOP_RATIO = 0.012      # how far mouth corners sit below mouth center, / face height, to count as a frown

# Landmark indices (MediaPipe FaceMesh, 468-point model)
FOREHEAD, CHIN = 10, 152
LEFT_CHEEK, RIGHT_CHEEK = 234, 454
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
L_EYE_OUT, L_EYE_IN, L_EYE_TOP, L_EYE_BOT = 33, 133, 159, 145
R_EYE_OUT, R_EYE_IN, R_EYE_TOP, R_EYE_BOT = 263, 362, 386, 374
L_BROW, R_BROW = 105, 334
NASION = 168  # stable point between the eyes, used as a depth reference for pout/kiss


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_expression(lm):
    face_h = _dist(lm[FOREHEAD], lm[CHIN]) or 1e-6
    face_w = _dist(lm[LEFT_CHEEK], lm[RIGHT_CHEEK]) or 1e-6

    mouth_raw_h = _dist(lm[MOUTH_TOP], lm[MOUTH_BOTTOM])
    mouth_raw_w = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    mouth_h = mouth_raw_h / face_h
    mouth_w = mouth_raw_w / face_w
    mouth_aspect = mouth_raw_w / (mouth_raw_h or 1e-6)

    mouth_z = (lm[MOUTH_TOP].z + lm[MOUTH_BOTTOM].z) / 2
    pout_protrusion = (lm[NASION].z - mouth_z) / face_h

    mouth_center_y = (lm[MOUTH_TOP].y + lm[MOUTH_BOTTOM].y) / 2
    corner_y = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2
    mouth_droop = (corner_y - mouth_center_y) / face_h  # positive = corners sag below center: a frown

    l_ear = _dist(lm[L_EYE_TOP], lm[L_EYE_BOT]) / (_dist(lm[L_EYE_OUT], lm[L_EYE_IN]) or 1e-6)
    r_ear = _dist(lm[R_EYE_TOP], lm[R_EYE_BOT]) / (_dist(lm[R_EYE_OUT], lm[R_EYE_IN]) or 1e-6)

    l_brow = _dist(lm[L_BROW], lm[L_EYE_TOP]) / face_h
    r_brow = _dist(lm[R_BROW], lm[R_EYE_TOP]) / face_h
    brow_raise = (l_brow + r_brow) / 2

    both_closed = l_ear < EAR_CLOSED and r_ear < EAR_CLOSED
    one_closed = (l_ear < EAR_CLOSED) != (r_ear < EAR_CLOSED)

    if mouth_h > MOUTH_OPEN_RATIO and mouth_aspect < TEETH_ASPECT_MIN:
        return "mouth_open"  # round wide-open mouth: shock/gasp
    if mouth_w > SMILE_WIDTH_RATIO and mouth_h > TEETH_MIN_OPEN_RATIO:
        return "teeth"  # wide + open: grin showing teeth (geometry proxy, not pixel teeth detection)
    if mouth_w < KISS_MAX_WIDTH_RATIO and pout_protrusion > KISS_PROTRUSION_RATIO:
        return "kiss"  # narrow mouth pushed toward the camera: pucker/pout
    if mouth_droop > SAD_DROOP_RATIO:
        return "sad"  # mouth corners sagging below center: frown
    if both_closed:
        return "eyes_closed"
    if one_closed:
        return "wink"
    if brow_raise > BROW_RAISE_RATIO:
        return "eyebrows_raised"
    if mouth_w > SMILE_WIDTH_RATIO:
        return "smile"  # wide but closed-ish mouth

    return None
