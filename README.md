# Subway Motion YOLO Controller

YOLO tracks your body and hand gestures, then sends arrow keys to the PC game window on Ubuntu/X11. Android emulator input remains available with `SUBWAY_INPUT_MODE=android`.

## Run it on the PC game (Ubuntu/X11)

1. Install the key sender: `sudo apt install xdotool`.
2. Start the PC game first and leave its window title exactly “Subway Surf” (the window title reported by your game). The controller requires one exact game-window match so it cannot send keys to the terminal or preview.
3. Run the controller:

```bash
python3 -m pip install opencv-python ultralytics huggingface-hub numpy
python3 yolo_phone_controller.py
```

The first run downloads the person and hand models. The PC game must be open before the controller starts. The input worker focuses the game for keypresses and leaves focus there; click the game and dismiss any pause overlay after startup, then avoid clicking the preview while playing. If its title differs, set `SUBWAY_WINDOW_NAME` to an exact-match regular expression. To use Android instead, run `SUBWAY_INPUT_MODE=android python3 yolo_phone_controller.py`.

## Controls

- **Step into the left camera third:** switch one lane left.
- **Step into the right camera third:** switch one lane right.
- **The center third maps to the game’s middle lane.** Step into any of the three zones and the runner moves to that matching lane; return to the center zone to move back to the middle. The preview marks both boundaries and your tracked center. A zone must be stable for three frames to trigger. Start the game run in its middle lane when you start the controller.
- **Jump your body:** jump in the game.
- **Crouch:** slide under obstacles. Stand upright again to re-arm the next action. On startup, stand upright in view while the controller calibrates for about one second.
- **Clap (PC):** each clap toggles gesture input. After dying, clap once to pause input, select Play/Replay in the game, then clap again to resume and reset the controller’s lane estimate to center. In Android mode, a clap taps pause; resume manually and press **R** in the preview.
- **Middle finger:** switch control to the next visible person. This requires two or more people in camera view.
- **Ctrl+C in the launching terminal:** stop the controller. **Q** also quits if the preview has focus (which can pause the game).

The preview shows the selected person, current camera zone, detected hand boxes, clap palm count, inference FPS, and latest key or ADB input result.

## Performance notes

Person tracking and body jump/crouch detection use `yolo26n.pt` at 320px. Body height and vertical position are calibrated from the first second of standing upright. Hand gestures use a YOLO26n HaGRID model on the selected person every other frame for clap and person switching. A background worker sends keyboard events or ADB swipes so input delivery does not block camera inference. If NVIDIA CUDA is available, the script selects it; otherwise it uses CPU.

The hand dataset mentioned in the pasted chat did not include a download URL. The separate dataset found under `Desktop/robot` is labeled for bags and people, so it is not used here. Distance estimation, segmentation, Albumentations, and Ray Tune are for spatial analysis or model training; they do not reduce live gesture-control latency in this script.

`subway_yolo.py` is a separate original runner demo. Install its dependencies with `python3 -m pip install opencv-python ultralytics numpy`, then run `python3 subway_yolo.py`. Use `yolo_phone_controller.py` for the PC game or Android emulator.
