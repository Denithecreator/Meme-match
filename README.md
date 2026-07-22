# meme-mirror

Your webcam watches your hands and your face. The moment you make a pose it recognizes — thumbs up, middle finger, mewing, a wink, a duck-face — the matching meme pops up in a second window.

No training, no dataset, no cloud calls. Two MediaPipe models produce landmark coordinates; every rule after that is plain geometry over those coordinates in a few hundred lines of Python, so every decision the program makes is readable and tunable in one file.

```
Camera  ──► MediaPipe GestureRecognizer (2 hands)  ─┐
        └─► MediaPipe FaceLandmarker (468 points)  ─┤
                                                    ├─► rule layer ─► priority resolver ─► rolling-vote smoother ─► meme window
   gesture_map.py / custom_gesture.py / face_rules.py / custom_match.py
```

---

## Contents

- [Quick start](#quick-start)
- [What it detects](#what-it-detects)
- [How it works](#how-it-works)
  - [Landmarks in, rules out](#landmarks-in-rules-out)
  - [Priority resolution](#priority-resolution)
  - [Smoothing: rolling vote, not a debounce](#smoothing-rolling-vote-not-a-debounce)
  - [Custom face matching](#custom-face-matching)
- [Configuring your own memes](#configuring-your-own-memes)
- [Tuning detection](#tuning-detection)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [License](#license)

---

## Quick start

Requires Python 3.9+ and a webcam.

```bash
git clone https://github.com/umaarmirzaa/meme-mirror.git
cd meme-mirror
pip install -r requirements.txt
python fetch_models.py
python main.py
```

`fetch_models.py` pulls the two model bundles (~12 MB total) from the official Google MediaPipe bucket into `models/`. They are not committed to the repo.

Two windows open: **Camera** (your mirrored feed, landmark overlay, the currently detected label at the top, and a per-hand readout at the bottom-left) and **Meme Match** (the matched image). Press `q` in the Camera window to quit.

The repo ships with no meme images — `memes_config.json` maps every rule to a filename, and any file that isn't there renders as a gray tile with the rule name printed on it. So a fresh clone runs immediately and doubles as a live map of what the detector is seeing. Drop your own images in and they replace the placeholders.

---

## What it detects

**Hand gestures from MediaPipe's built-in recognizer** (`gesture_map.py`) — `thumbs_up`, `thumbs_down`, `peace`, `open_palm`, `fist`, `pointing`, `love_you`.

**Hand gestures from custom geometry** (`custom_gesture.py`), for poses the canned model has no class for:

| Rule | What it looks for |
|---|---|
| `middle_finger` | middle finger extended, neighbours curled |
| `finger_heart` | BTS-style thumb-and-index mini heart, one hand |
| `full_heart` | both hands brought together into a heart |
| `welcome_gesture` | both hands pointing down together (mapped to "aap yahan aa jayein" in the sample config) |
| `hands_on_head` | both palms near the forehead — needs face + both hands |
| `mewing` | index fingertip tracing the jawline |
| `finger_in_mouth` | index fingertip inside a parted mouth |
| `shush` | index fingertip on closed lips |
| `hand_on_ear` | open palm resting near the ear |
| `six_seven` | the 2025 "6-7" seesaw — see [Limitations](#limitations) |

`mewing`, `finger_in_mouth`, `shush` and `hand_on_ear` are hand-plus-face combos and need both visible. `finger_in_mouth` and `shush` share a touch zone and are split purely by whether the mouth is parted or shut.

**Face expressions from landmark geometry** (`face_rules.py`) — `mouth_open`, `smile`, `teeth`, `kiss`, `sad`, `eyebrows_raised`, `wink`, `eyes_closed`.

**Your own reference photos** (`custom_match.py`) — see [Custom face matching](#custom-face-matching).

Both hands are tracked at once. The Camera window's bottom-left line (`hands: peace | middle_finger`) shows what each hand reads as, even before a meme locks in — useful when tuning.

---

## How it works

### Landmarks in, rules out

MediaPipe gives normalized coordinates: 21 points per hand, 468 points for the face. Nothing downstream is learned — each rule is a ratio between distances, and every ratio is normalized by a face or hand span so it holds at any distance from the camera.

Example, the whole `smile` rule: mouth corner-to-corner distance divided by cheek-to-cheek distance, above a threshold. Example, `kiss`: mouth narrower than a width threshold **and** the lips' z-depth pushed toward the camera relative to the nasion (the stable point between the eyes). Every threshold is a named constant at the top of its module.

### Priority resolution

Several rules can be true in the same frame — you can smile while flashing a peace sign. `main.py` resolves this most-specific-first:

```
custom face match  >  two-hand combo  >  one-hand + face combo  >  passive face expression  >  single-hand gesture
```

The reasoning: a deliberate two-handed pose is a stronger signal of intent than a passive expression your face happens to be holding, and an expression you posed for beats a hand shape the recognizer picked up incidentally.

### Smoothing: rolling vote, not a debounce

Landmark detection jitters near a threshold boundary. The obvious fix — "require N identical frames in a row" — actually fails here: a single stray frame flickering to `None` or to a neighbouring rule resets the streak, and a gesture held perfectly steady can end up never accumulating enough consecutive frames to ever display.

Instead the last `HISTORY_LEN = 7` frames are kept in a ring buffer and the most common entry wins if it holds at least `CONFIRM_RATIO = 0.55` of the window. A few flickered frames are absorbed as long as one match still dominates, and confirmation typically takes 2–3 frames rather than a fixed hold count — more reliable *and* faster than the naive version.

### Custom face matching

`custom_memes` in the config lets you supply reference photos of yourself pulling a specific face. At startup, `custom_match.py` runs the face landmarker over each reference and stores a normalized geometric signature; at runtime your live face is compared against those signatures.

This matches **expression geometry, not identity** — nothing about who you are is encoded, and any face shape works with your own references. Reference images stay on your machine; nothing is uploaded.

---

## Configuring your own memes

Everything lives in `memes_config.json`. Two ways to trigger an image:

**1. `gesture_rules` / `face_rules`** — point `image` at a file. `rule` must be one of the names listed above; `name` is just the label drawn on the placeholder tile.

```json
{ "name": "thumbs_up", "rule": "thumbs_up", "image": "memes/images/approve.jpg" }
```

**2. `custom_memes`** — no rule name, just reference photos of the expression you want to trigger it:

```json
{
  "name": "my_disappointed_face",
  "image": "memes/images/disappointed.png",
  "refs": ["memes/refs/disappointed/ref1.jpg", "memes/refs/disappointed/ref2.jpg"]
}
```

More reference photos across different lighting and angles gives a more robust match. Images are loaded once at startup, so restart after editing.

---

## Tuning detection

If a rule fires too eagerly or not at all, change the threshold — not the smoothing. The smoothing constants in `main.py` compensate for jitter; they can't fix a threshold that's simply in the wrong place.

- Face rules: `MOUTH_OPEN_RATIO`, `SMILE_WIDTH_RATIO`, `EAR_CLOSED`, `BROW_RAISE_RATIO`, `TEETH_ASPECT_MIN`, `TEETH_MIN_OPEN_RATIO`, `KISS_MAX_WIDTH_RATIO`, `KISS_PROTRUSION_RATIO`, `SAD_DROOP_RATIO` — all at the top of `face_rules.py`.
- Hand rules: the ratio constants at the top of `custom_gesture.py`.

Watch the bottom-left `hands:` readout and the top-left label while you adjust — they show raw per-frame classification, so you can see a threshold flip in real time.

---

## Project layout

| File | Role |
|---|---|
| `main.py` | capture loop, model setup, priority resolution, smoothing, both windows |
| `gesture_map.py` | MediaPipe's built-in gesture category names → rule names |
| `custom_gesture.py` | geometry rules for one-hand, two-hand, and hand+face poses |
| `face_rules.py` | expression classification from the 468-point face mesh |
| `custom_match.py` | reference-photo signature building and matching |
| `memes_config.json` | rule → image mapping |
| `fetch_models.py` | downloads the two `.task` model bundles |

---

## Limitations

Worth knowing before you file a bug:

- **`six_seven` is approximate.** The real trend is a repeated seesaw *motion*; this detector is single-frame, so it approximates it as one open hand caught noticeably higher than the other. Less reliable than the pose-based rules by construction.
- **`teeth` doesn't see teeth.** It's a wide-and-open mouth geometry proxy for a grin, so a wide open-mouthed laugh and genuine shock can read the same. Nudge `TEETH_ASPECT_MIN` / `TEETH_MIN_OPEN_RATIO` for your face.
- **`kiss` leans on monocular depth.** MediaPipe's z-estimates from a single camera are noisy; the pout rule is the most lighting-sensitive of the set.
- **Thresholds are per-face.** Defaults were tuned on one face under one lighting setup. Expect to adjust a couple of constants.
- **Windows-tuned window placement.** `main.py` queries screen width through `ctypes.windll` and falls back to 1920 elsewhere — the app still runs fine on macOS/Linux, the second window may just land in a less convenient spot. The `cv2.CAP_DSHOW` capture backend is likewise a Windows hint.
- **One face at a time.** `num_faces=1`, `num_hands=2`.

---

## License

MIT — see [LICENSE](LICENSE).

Meme images are not included in this repository; whatever you drop into `memes/images/` stays yours and stays local.
