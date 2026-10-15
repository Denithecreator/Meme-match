"""Geometric hand-gesture rules for poses MediaPipe's canned gesture recognizer
doesn't cover: call-me, BTS-style finger heart, middle finger, two-hand full heart.

Landmark indices (MediaPipe Hands, 21 points):
0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip
"""
import math

TIP_IDS = [4, 8, 12, 16, 20]

FINGER_HEART_TIP_DIST_RATIO = 0.45   # thumb tip / index tip closeness, relative to hand scale
FULL_HEART_TIP_DIST_RATIO = 0.7      # cross-hand thumb/index closeness, relative to hand scale
EXTENDED_DIST_RATIO = 0.6            # tip-to-mcp distance ratio counted as "extended" (direction-agnostic)
POINT_DOWN_DROP_RATIO = 0.25         # how far below its mcp the index tip must sit to count as "down"
HAND_ON_HEAD_RADIUS_RATIO = 1.3      # palm-to-forehead distance (x face height) counted as "on head"
JAW_TOUCH_RATIO = 0.18               # index fingertip-to-jawline distance (x face height) counted as "touching"
MOUTH_TOUCH_RATIO = 0.14             # index fingertip-to-mouth-center distance (x face height) counted as "inserted"
MOUTH_MIN_OPEN_RATIO = 0.015         # min inner-mouth height (x face height) required alongside the touch, so a
                                      # finger merely resting near closed lips doesn't count
EAR_TOUCH_RATIO = 0.3                # palm-to-ear-proxy distance (x face height) counted as "on the ear"
SIX_SEVEN_HEIGHT_DIFF_RATIO = 0.9    # min vertical gap between the two hands (x hand scale) for the 6-7 seesaw
OPEN_LOOSE_RATIO = 1.4               # tip-to-wrist distance ratio; a folded fist keeps tips near the palm (~1.0x),
                                      # an extended finger reaches well beyond that

# FaceMesh jaw contour landmarks, left cheek -> chin -> right cheek
JAW_LANDMARKS = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
# FaceMesh doesn't mesh the ears themselves; the face-oval edge points closest
# to each ear (cheek/temple height) are used as a proxy anchor.
EAR_LANDMARKS = [234, 454]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _midpoint(a, b):
    class _Pt:
        pass
    p = _Pt()
    p.x, p.y = (a.x + b.x) / 2, (a.y + b.y) / 2
    return p


def _hand_scale(lm):
    return _dist(lm[0], lm[9]) or 1e-6  # wrist to middle-finger MCP


def _fingers_up(lm, handedness_label):
    """[thumb, index, middle, ring, pinky] booleans (True = extended)."""
    if handedness_label == "Right":
        thumb = lm[4].x < lm[3].x
    else:
        thumb = lm[4].x > lm[3].x
    rest = [lm[TIP_IDS[i]].y < lm[TIP_IDS[i] - 2].y for i in range(1, 5)]
    return [thumb] + rest


def classify_hand(lm, handedness_label):
    """Return a custom gesture name for a single hand, or None."""
    thumb, index, middle, ring, pinky = _fingers_up(lm, handedness_label)
    scale = _hand_scale(lm)

    if middle and not index and not ring and not pinky:
        return "middle_finger"

    thumb_extended = _dist(lm[4], lm[0]) > OPEN_LOOSE_RATIO * scale
    pinky_extended = _dist(lm[20], lm[0]) > OPEN_LOOSE_RATIO * scale
    other_fingers_folded = all(
        _dist(lm[tip], lm[mcp]) < EXTENDED_DIST_RATIO * scale
        for tip, mcp in ((8, 5), (12, 9), (16, 13))
    )
    if thumb_extended and pinky_extended and other_fingers_folded:
        return "call_me"

    tip_dist = _dist(lm[4], lm[8]) / scale
    if tip_dist < FINGER_HEART_TIP_DIST_RATIO and not middle and not ring and not pinky:
        # Distinguish from a resting fist: a real finger-heart crosses thumb+index
        # ABOVE the knuckle line, away from the palm. A fist keeps them tucked at
        # palm level, so gate on the pinch point sitting higher than the index MCP.
        pinch_y = (lm[4].y + lm[8].y) / 2
        if pinch_y < lm[5].y:
            return "finger_heart"

    return None


def is_pointing_down(lm):
    """Index finger extended and aimed down, other fingers folded.

    _fingers_up()'s tip-vs-pip comparison assumes the hand points roughly
    upward, so it can't tell "extended downward" from "curled" -- this uses a
    direction-agnostic distance check instead, plus an explicit downward gate.
    """
    scale = _hand_scale(lm)
    index_extended = _dist(lm[8], lm[5]) > EXTENDED_DIST_RATIO * scale
    middle_folded = _dist(lm[12], lm[9]) < EXTENDED_DIST_RATIO * scale
    ring_folded = _dist(lm[16], lm[13]) < EXTENDED_DIST_RATIO * scale
    pinky_folded = _dist(lm[20], lm[17]) < EXTENDED_DIST_RATIO * scale
    pointing_down = (lm[8].y - lm[5].y) > POINT_DOWN_DROP_RATIO * scale
    return index_extended and middle_folded and ring_folded and pinky_folded and pointing_down


def _is_loosely_open(lm):
    """At least half the (non-thumb) fingertips sit well away from the wrist.

    Looser than _fingers_up()'s per-finger check -- just enough to tell a
    relaxed, half-open hand (like a seesaw/weighing motion) from a fist.
    """
    scale = _hand_scale(lm)
    open_count = sum(1 for tip in (8, 12, 16, 20) if _dist(lm[tip], lm[0]) > OPEN_LOOSE_RATIO * scale)
    return open_count >= 2


def classify_two_hands(lm_a, lm_b):
    """Return a combo gesture name when both hands together mean something, else None."""
    # Check pointing-down first: it has an explicit direction gate, so it's the
    # more specific condition. full_heart's tip-distance threshold is loose enough
    # that two hands pointing down near each other can satisfy it by coincidence.
    if is_pointing_down(lm_a) and is_pointing_down(lm_b):
        return "welcome_gesture"

    scale = (_hand_scale(lm_a) + _hand_scale(lm_b)) / 2
    thumb_dist = _dist(lm_a[4], lm_b[4]) / scale
    index_dist = _dist(lm_a[8], lm_b[8]) / scale
    if thumb_dist < FULL_HEART_TIP_DIST_RATIO and index_dist < FULL_HEART_TIP_DIST_RATIO:
        return "full_heart"

    # "6-7" trend: a palms-up seesaw motion, one hand rising as the other falls.
    # It's fundamentally a repeated MOTION, not a static pose, so a single frame
    # can only approximate it -- one hand caught noticeably higher than the
    # other, both loosely open (not fists). Checked last since it's the
    # loosest condition here and shouldn't preempt the more specific poses above.
    height_diff = abs(lm_a[9].y - lm_b[9].y) / scale
    if height_diff > SIX_SEVEN_HEIGHT_DIFF_RATIO and _is_loosely_open(lm_a) and _is_loosely_open(lm_b):
        return "six_seven"

    return None


def classify_hand_with_face(hand_lm, face_lm):
    """Return a single-hand + face combo gesture name, or None.

    Checks poses whose defining signal is contact between a hand and a face
    region MediaPipe doesn't classify on its own: finger on the jaw (mewing),
    finger inserted in the mouth or pressed to closed lips (finger_in_mouth /
    shush), and a hand resting on the ear.
    """
    if face_lm is None:
        return None
    face_h = _dist(face_lm[10], face_lm[152]) or 1e-6
    tip = hand_lm[8]

    # Mewing selfies are basically a neutral closed mouth -- there's no reliable
    # face-only geometry for it, so the actual signal is the finger tracing the jaw.
    # Doesn't require other fingers folded since the resting hand pose varies.
    min_jaw_d = min(_dist(tip, face_lm[i]) for i in JAW_LANDMARKS)
    if min_jaw_d < JAW_TOUCH_RATIO * face_h:
        return "mewing"

    # Finger at the mouth: tip close to the mouth center either way -- whether
    # it reads as "inserted" or "shushing" depends on whether the lips are
    # parted (finger_in_mouth) or pressed shut (shush), same touch zone either way.
    mouth_mid = _midpoint(face_lm[MOUTH_TOP], face_lm[MOUTH_BOTTOM])
    mouth_open_h = _dist(face_lm[MOUTH_TOP], face_lm[MOUTH_BOTTOM]) / face_h
    if _dist(tip, mouth_mid) / face_h < MOUTH_TOUCH_RATIO:
        return "finger_in_mouth" if mouth_open_h > MOUTH_MIN_OPEN_RATIO else "shush"

    # Hand on ear: palm near either ear-proxy point (FaceMesh doesn't cover the
    # ears themselves, so the nearest face-oval edge point stands in).
    palm = hand_lm[9]
    min_ear_d = min(_dist(palm, face_lm[i]) for i in EAR_LANDMARKS)
    if min_ear_d < EAR_TOUCH_RATIO * face_h:
        return "hand_on_ear"

    return None


def classify_hands_on_head(lm_a, lm_b, face_lm):
    """Return 'hands_on_head' if both palms rest near the forehead/scalp area.

    FaceMesh only covers the face oval, not the scalp, so this uses the
    forehead landmark (10) as a proxy anchor with a generous radius.
    """
    if face_lm is None:
        return None
    forehead = face_lm[10]
    face_h = _dist(face_lm[10], face_lm[152]) or 1e-6
    radius = HAND_ON_HEAD_RADIUS_RATIO * face_h
    palm_a, palm_b = lm_a[9], lm_b[9]
    if _dist(palm_a, forehead) < radius and _dist(palm_b, forehead) < radius:
        return "hands_on_head"
    return None
