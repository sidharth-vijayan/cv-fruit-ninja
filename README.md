# CV Fruit Ninja (Python + MediaPipe + Pygame)

Playable webcam Fruit Ninja clone using MediaPipe Hands for hand tracking and Pygame for visuals.

## Features
- Real-time hand tracking via MediaPipe.
- Swipe detection (velocity + length heuristics).
- Fruits spawn and fall with physics; bomb items penalize player.
- Score, combo multiplier, lives, levels.
- Simple particle effect on slice.
- Save high scores to high_scores.json.
- Unit test for collision function.

## Requirements
Python 3.9+ (works on 3.8+)
Install:
pip install -r requirements.txt

## Run
1. (Optional) Download assets:
   python scripts/download_assets.py
   Or manually place PNGs in assets/fruits/ and assets/bomb.png

2. Run:
   python main.py

Controls:
- ESC to quit
- C toggle hand overlay
- M toggle camera background
- R restart when game over
- Enter to save & exit on game over

## Notes on performance
- MediaPipe runs on CPU by default. Expect ~20-40 FPS on typical laptop depending on resolution and CPU.
- Lower webcam resolution to improve performance (`cap.set` in main.py).
- Pygame rendering is not GPU-accelerated in some environments; keep screen size moderate (1024x720 recommended).

## Cross-platform tips
- Windows: Use a virtualenv and `pip install` as normal. If OpenCV install fails, try `pip install opencv-python-headless`.
- macOS: `brew install python3` if needed. On M1/M2 chips, use native Python and pip wheels.
- Linux: may need `sudo apt-get install libsm6 libxext6 libxrender-dev` for OpenCV.

## Stretch ideas
- Add wav sound effects on slice (Pygame mixer).
- Add multiplayer hot-seat mode by switching camera feed or two webcams.
- Create a Streamlit/Gradio wrapper to stream webcam frames to a webpage (less responsive).

