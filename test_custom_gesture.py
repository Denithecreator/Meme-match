import unittest
from types import SimpleNamespace

from custom_gesture import classify_hand_with_face, classify_hands_on_head


def _landmark(x, y):
    return SimpleNamespace(x=x, y=y)


def _face_and_hands(palm_y):
    face = [_landmark(0, 0) for _ in range(153)]
    face[10] = _landmark(0.5, 0.3)
    face[152] = _landmark(0.5, 0.7)

    hands = []
    for palm_x in (0.4, 0.6):
        hand = [_landmark(0, 0) for _ in range(10)]
        hand[9] = _landmark(palm_x, palm_y)
        hands.append(hand)
    return face, hands


def _face_and_finger(tip_x, tip_y, mouth_open=False):
    face = [_landmark(0, 0) for _ in range(468)]
    face[10] = _landmark(0.5, 0.3)
    face[152] = _landmark(0.5, 0.7)
    face[61] = _landmark(0.44, 0.52)
    face[291] = _landmark(0.56, 0.52)
    face[13] = _landmark(0.5, 0.515 if mouth_open else 0.52)
    face[14] = _landmark(0.5, 0.525 if mouth_open else 0.52)

    hand = [_landmark(0, 0) for _ in range(21)]
    hand[8] = _landmark(tip_x, tip_y)
    return face, hand


class ClassifyHandsOnHeadTests(unittest.TestCase):
    def test_hands_near_forehead_are_hands_on_head(self):
        face, hands = _face_and_hands(0.32)

        self.assertEqual(classify_hands_on_head(*hands, face), "hands_on_head")

    def test_raised_hands_are_absolute_cinema(self):
        face, hands = _face_and_hands(0.14)

        self.assertEqual(classify_hands_on_head(*hands, face), "absolute_cinema")

    def test_hands_between_thresholds_are_unclassified(self):
        face, hands = _face_and_hands(0.19)

        self.assertIsNone(classify_hands_on_head(*hands, face))


class ClassifyHandWithFaceTests(unittest.TestCase):
    def test_closed_mouth_corner_touches_are_finger_in_mouth(self):
        for tip_x in (0.44, 0.56):
            with self.subTest(tip_x=tip_x):
                face, hand = _face_and_finger(tip_x, 0.52)

                self.assertEqual(classify_hand_with_face(hand, face), "finger_in_mouth")

    def test_closed_mouth_center_touch_is_shush(self):
        face, hand = _face_and_finger(0.5, 0.52)

        self.assertEqual(classify_hand_with_face(hand, face), "shush")

    def test_open_mouth_center_touch_remains_finger_in_mouth(self):
        face, hand = _face_and_finger(0.5, 0.52, mouth_open=True)

        self.assertEqual(classify_hand_with_face(hand, face), "finger_in_mouth")


if __name__ == "__main__":
    unittest.main()
