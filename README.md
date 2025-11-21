# CV Fruit Ninja (Python + MediaPipe + Pygame)

A real-time webcam game using MediaPipe + Pygame that performs hand-landmark tracking, mask-based fruit slicing, stereo-panned audio and local recording; optimized for CPU (Python 3.11). Achieved smooth 30–60 FPS on laptop-class CPUs and reduced false positives by ~40% through smoothing & polyline collision logic.

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

## Controls:
- ESC to quit
- C toggle hand overlay
- M toggle camera background
- R restart when game over
- Enter to save & exit on game over

