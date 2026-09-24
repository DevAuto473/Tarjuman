# Recording on the Raspberry Pi

Collect the dataset on the Pi with its own camera, train on the laptop, send the
model back. This document is the whole procedure, plus the two traps that cost
the most time.

Verified against: **Pi 5, Raspberry Pi OS Bookworm (Python 3.11), Camera Module 3.**

---

## Why this split is the right one

The final system runs on the Pi. So the Pi's camera is the camera that matters,
and recording with it is not a compromise — it is the correct choice. Training
happens on the laptop only because training is a one-off batch job that wants
CPU, and the Pi would take hours to do what the laptop does in a minute.

Nothing about the dataset is laptop-specific. `dynamic_gestures_v5.csv` is
numbers; it moves by USB stick.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 libportaudio2 libgl1 libglib2.0-0 git
```

* `python3-picamera2` — the camera stack. **Not available through pip.**
* `libportaudio2` — `sounddevice` is a hard dependency of mediapipe and fails to
  import without it, with an error that names PortAudio and not mediapipe.
* `libgl1`, `libglib2.0-0` — the OpenCV wheels link against these but do not
  ship them. Without them `import cv2` raises
  `libGL.so.1: cannot open shared object file`, which reads like a broken
  install and is really a missing system library.

Node.js is **not** needed on the Pi. The `npm run …` shortcuts exist for the
laptop; here the collector is started directly with the venv's Python.

### Bookworm runs Wayland

`cv2.imshow` draws through GTK/X11, so on a Wayland desktop it goes via
XWayland. On a normal Raspberry Pi OS Desktop install that is already running
and the window simply appears. If it does not — no window, or a
`cannot connect to display` error — check that a display is actually reachable:

```bash
echo "$WAYLAND_DISPLAY / $DISPLAY"     # at least one must be set
```

Run the collector from a terminal **on the Pi's own desktop**, not from an SSH
session that has inherited no display. Over SSH use `ssh -X` (which needs
`xauth` on the Pi) or a VNC session.

---

## 2. The virtualenv — the trap that wastes an afternoon

`picamera2` is installed by apt into `/usr/lib/python3/dist-packages`. A normal
virtualenv cannot see that directory. Create the venv **with system packages
visible**, or the camera backend is invisible no matter how correctly apt
installed it:

```bash
cd ~/TarjumanV1
python3 -m venv --system-site-packages venv
```

Get this wrong and the symptom is `ModuleNotFoundError: No module named
'picamera2'` while `python3 -c "import picamera2"` works fine outside the venv.

Verify before going further:

```bash
venv/bin/python -c "import picamera2; print('picamera2 OK')"
```

---

## 3. Python packages

```bash
venv/bin/pip install -r requirements-pi.txt
```

Use `requirements-pi.txt`, **not** `requirements.txt`. The main file is a
`pip freeze` from Windows carrying training, the server, speech and plotting —
none of which the collector imports.

Every pin in `requirements-pi.txt` was checked to publish a `cp311` aarch64
wheel, and the full set was resolved for `aarch64/cp311` before this document
was written: **26 wheels, ~240 MB, nothing compiles.** Budget the download time
and the SD card space; jaxlib alone is 77 MB and OpenCV 56 MB.

### Two things that look wrong here and are not

**`numpy` is 2.x, not `<2`.** The usual advice for a 2024-era package is to pin
`numpy<2` against the ABI break. It does not apply here, and pinning it makes
the install impossible:

* `mediapipe==0.10.14` requires `jax` and `jaxlib`; `jax==0.10.1` and
  `jaxlib==0.10.1` each require **`numpy>=2.0`**. Resolving this file with
  `numpy==1.26.4` fails outright with `ResolutionImpossible`.
* The break itself does not occur. `mediapipe 0.10.14` + `numpy 2.4.6` was
  installed and `Hands().process()` was run against a numpy-2 array — it
  returned normally. MediaPipe passes images across pybind11 as buffers and
  does not use the numpy C API that changed in 2.0.
* And the laptop has been running exactly this pair all along:
  `requirements.txt` is a freeze of a working install.

Keeping the same numpy on both machines is also the point: features recorded on
the Pi and features used for training should come out of identical arithmetic.

**OpenCV is the `contrib` build, not the slim one.** Not a choice —
`mediapipe==0.10.14` declares `opencv-contrib-python` as a hard dependency. In a
clean venv where only `mediapipe` was requested, pip installed
`opencv-contrib-python 5.0.0.93` on its own. Asking for `opencv-python` instead
does not save the 56 MB; it ends up with both.

---

## 4. `.env` for the Pi

```ini
CAMERA_SOURCE=picamera2

# Match the laptop exactly. MediaPipe normalises x by width and y by height
# SEPARATELY, so a different aspect ratio distorts the landmark coordinates —
# and with them every feature derived from the body frame.
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

CAMERA_FPS=60

# Only if the room is dim and you cannot add light — see below.
# CAMERA_EV=1.5
```

### Why 60 fps, and what it costs

60 is not about collecting more samples: every take is resampled to
`SEQUENCE_LENGTH` regardless. It is about **exposure time**. At 60 fps each
frame is exposed for at most 1/60 s, and a hand crossing a longer exposure
smears. A blurred hand is one MediaPipe places badly, and every feature
downstream inherits that error. The grabber thread keeps only the newest frame,
so a faster sensor also halves how stale the frame handed to the tracker is —
which holds even though inference on the Pi runs slower than the sensor.

The price is light. A short exposure needs a lit room, and that is a lamp
problem, not a settings problem.

**The Pi path now measures the result.** It used to pass a frame rate and trust
whatever came back: the brightness guard lived entirely in the OpenCV branch,
so on the Pi a dark picture was possible with nothing looking at it.
`_tune_picamera_ae()` closes that. On startup it:

* **meters the centre of the frame**, not the whole of it. This is the direct
  fix for a lit wall or doorway behind you: whole-frame metering averages that
  brightness in, the AE shortens the exposure to protect it, and you become a
  silhouette. Centre metering measures the signer.
* applies `CAMERA_EV` / `--ev` if you set one — a deliberate stop or two of
  over-exposure, costing noise rather than frame rate.
* **measures the picture and prints the number.** Below the usable threshold it
  says so, and names the three real remedies rather than leaving you to guess.

It does not chase gain by hand the way the USB path does. libcamera's AE
already raises analogue gain once it cannot lengthen the exposure further; the
missing piece was never gain control, it was that nobody looked at the result.

If it reports a dark frame: add light in front of you first, then `--fps 30` to
double the exposure time, then `--ev 1.5`.

---

## 5. Light before anything else

MediaPipe cannot find a hand in a silhouette, and the recorder will reject take
after take with *"A hand was seen in only 37% of frames"*.

**Light in front of you, not behind you.** A lamp aimed at your face and hands;
never a window or a lit doorway at your back. This is not polish — half of the
`sorry` recordings in the previous dataset failed on broken hand tracking, and
lighting is the most likely cause.

**Both shoulders inside the frame, every take.** This one is structural, not
cosmetic — see section 7.

---

## 6. Record

```bash
cd ~/TarjumanV1
PYTHONPATH=src venv/bin/python data_collector.py
```

It asks which machine this is before it opens anything, and applies that
machine's profile:

| | camera | size | fps |
|---|---|---|---|
| `laptop` | auto | 640x480 | 60 |
| `pi` | picamera2 | 640x480 | 60 |

Then it prints every value **and where it came from** before touching the
sensor, and after opening prints what the sensor actually delivered — a request
the camera quietly ignored shows up now rather than after a session of
recording at the wrong size.

Skip the question and override anything:

```bash
# the profile, no questions asked
PYTHONPATH=src venv/bin/python data_collector.py --device pi --yes

# a dim room: fewer frames, longer exposure each
PYTHONPATH=src venv/bin/python data_collector.py --device pi --fps 30

# backlit, and moving the lamp is not an option
PYTHONPATH=src venv/bin/python data_collector.py --device pi --ev 1.5

# a different sensor mode
PYTHONPATH=src venv/bin/python data_collector.py --device pi --size 1280x720

# fewer samples per word while testing the setup
PYTHONPATH=src venv/bin/python data_collector.py --device pi --samples 5
```

`--width/--height` work instead of `--size`; `--camera` overrides the source.
Precedence is **command line > `.env` > profile**, so the `.env` above still
decides whenever no flag is given. `--help` lists all of it.

Keep the size identical across every session. Changing it mid-dataset is the
one override that quietly corrupts the data — see section 7.

No `npm`, and no Node.js on the Pi at all. `PYTHONPATH=src` is what the npm
wrapper does on the laptop, and it is the whole of what it does — the collector
needs nothing else from Node.

A **monitor attached to the Pi is required.** The recorder opens an OpenCV
window and reads the `r` key from it.

Per word: calibrate three times, then press `r` once per sample. Rest between
batches is automatic.

---

## 7. The one thing that would silently ruin the dataset

Of the 140 values recorded per frame, 139 are expressed in **shoulder-width
units** —

```python
to_body = lambda p: (p - origin) / width   # origin = shoulder midpoint
```

— so lens angle, resolution and how far you sit all cancel out. That is what
makes the data portable at all.

But it only holds while the body is detected:

```python
if anchors is not None and anchors.valid:
    ...                      # body coordinates
else:
    ...                      # RAW FRAME COORDINATES
```

**With no shoulders in frame the whole scheme collapses to raw pixel
coordinates**, which are tied to that specific camera and framing. Takes
recorded that way are not comparable to anything — including each other.

So: sit far enough back that both shoulders are visible, and check that the
status panel says the body is detected before you start. It is worth one minute
of checking at the top of every session.

---

## 8. Back to the laptop

```
data/dynamic_gestures_v5.csv      Pi  ->  laptop
```

Then on the laptop:

```bash
npm run audit     # separability and completeness, before training
npm run train
```

and copy `sign_model.onnx` back to the Pi.

For the Pi to *run* the translator (rather than record for it) it needs the
server's dependencies too — `onnxruntime`, `websockets` and the speech stack.
That is a separate install, and a separate conversation.
