"""Maps MediaPipe GestureRecognizer's built-in gesture categories to our rule names."""

CATEGORY_TO_RULE = {
    "Thumb_Up": "thumbs_up",
    "Thumb_Down": "thumbs_down",
    "Victory": "peace",
    "Open_Palm": "open_palm",
    "Closed_Fist": "fist",
    "Pointing_Up": "pointing",
    "ILoveYou": "love_you",
    # "None" = model saw a hand but no recognized gesture; not mapped on purpose.
}


def to_rule(category_name):
    return CATEGORY_TO_RULE.get(category_name)
