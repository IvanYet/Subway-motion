"""YOLO body/hand controls for Subway Surfers on PC (X11) or Android."""
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "0"
    else:
        DEVICE = "cpu"
    torch.set_num_threads(4)
except Exception:
    DEVICE = "cpu"

# Tuning: lower imgsz is faster; raise it if the webcam view is very wide/dim.
CAMERA = os.environ.get("SUBWAY_CAMERA")  # Optional camera index; otherwise scan indices 0–5.
INPUT_MODE = os.environ.get("SUBWAY_INPUT_MODE", "pc").lower()  # pc or android
GAME_WINDOW_NAME = os.environ.get("SUBWAY_WINDOW_NAME", r"^Subway Surf$")
PERSON_MODEL = str(Path.home() / "yolo26n.pt") if (Path.home() / "yolo26n.pt").exists() else "yolo26n.pt"
HAND_MODEL_REPO = "SuperSura/hand-gesture-yolo26n"
HAND_MODEL_FILE = "yolo26n_hand-gesture_12c_v1.pt"
PERSON_SIZE = 320
HAND_SIZE = 416
HAND_EVERY = 2
PERSON_CONF = 0.40
CLAP_CONF = 0.58
GESTURE_CONFIRM_CHECKS = 2
LANE_ZONE_CONFIRM = 3
CENTER_ZONE_FRACTION = 0.20  # Narrow center zone makes smaller side-to-side moves switch lanes.
BODY_CALIBRATION_FRAMES = 20
BODY_ACTION_CONFIRM = 2
BODY_NEUTRAL_CHECKS = 3
JUMP_LIFT_FRACTION = .10
CROUCH_HEIGHT_FRACTION = .80

cv2.setNumThreads(1)


def find_adb():
    sdk_adb = Path.home() / "Android/Sdk/platform-tools/adb"
    return str(sdk_adb) if sdk_adb.exists() else (shutil.which("adb") or "adb")


ADB = find_adb()
GAME_WINDOW_ID = None


def adb(*args, capture=False, timeout=5):
    return subprocess.run([ADB, *args], check=True, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                          stderr=subprocess.PIPE if capture else subprocess.DEVNULL)


def get_screen_size():
    out = adb("shell", "wm", "size", capture=True).stdout
    sizes = re.findall(r"(\d+)x(\d+)", out)
    if not sizes:
        raise RuntimeError("Could not read Android screen size from `adb shell wm size`.")
    # `wm size` prints physical then override size, so the last value is active.
    return tuple(map(int, sizes[-1]))


def swipe_args(direction, width, height):
    if direction == "pause":
        # Subway Surfers' pause button sits at the upper-left of the portrait HUD.
        return [ADB, "shell", "input", "tap", str(int(width * .08)), str(int(height * .10))]
    cx, cy = width // 2, height // 2
    distance = int(min(width, height) * .30)
    coords = {
        "left": (cx + distance, cy, cx - distance, cy),
        "right": (cx - distance, cy, cx + distance, cy),
        "up": (cx, cy + distance, cx, cy - distance),
        "down": (cx, cy - distance, cx, cy + distance),
    }[direction]
    return [ADB, "shell", "input", "swipe", *(str(v) for v in coords), "100"]


def input_worker(work, stop, status, lock, mode, window_id, window_title):
    """Deliver game controls off the inference thread."""
    while not stop.is_set():
        try:
            item = work.get(timeout=.1)
        except queue.Empty:
            continue
        if item is None:
            break
        direction, width, height = item
        try:
            if mode == "pc" and direction == "pause":
                with lock:
                    status["sent"] = "CLAP: GESTURE CONTROL PAUSED"
                    status["sent_at"] = time.monotonic()
                continue
            if mode == "pc":
                key = {"left": "Left", "right": "Right", "up": "Up", "down": "Down"}[direction]
                active = subprocess.run(["xdotool", "getactivewindow"], check=False,
                                        capture_output=True, text=True, timeout=2).stdout.strip()
                active_title = subprocess.run(["xdotool", "getwindowname", active], check=False,
                                              capture_output=True, text=True, timeout=2).stdout.strip() if active else ""
                command = ["xdotool", "key", "--clearmodifiers", "--delay", "45", key]
                if active != window_id and active_title != window_title:
                    # Only focus the game when another window owns focus. Keep it active
                    # afterward; restoring the preview makes Unity/Wine pause the run.
                    command = ["xdotool", "windowactivate", "--sync", window_id, *command[1:]]
            else:
                command = swipe_args(direction, width, height)
            proc = subprocess.run(command, check=False, timeout=2, capture_output=True, text=True)
            if proc.returncode:
                detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[-1]
                raise RuntimeError(detail[:60])
            with lock:
                status["sent"] = (f"KEY SENT: {direction.upper()}" if mode == "pc" else
                                  "CLAP PAUSE TAP SENT" if direction == "pause" else f"ADB SENT: {direction.upper()}")
                status["sent_at"] = time.monotonic()
        except Exception as exc:
            with lock:
                detail = str(exc).strip()
                status["sent"] = f"INPUT ERROR: {detail[:55] or type(exc).__name__}"
                status["sent_at"] = time.monotonic()
        finally:
            work.task_done()


def label_text(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id)).lower().replace(" ", "_")
    return str(class_id)


def main():
    global GAME_WINDOW_ID
    try:
        if INPUT_MODE == "pc":
            xdotool = shutil.which("xdotool")
            if not xdotool:
                raise SystemExit("PC key input needs xdotool. Install it with: sudo apt install xdotool")
            matches = subprocess.run([xdotool, "search", "--onlyvisible", "--name", GAME_WINDOW_NAME],
                                     check=False, capture_output=True, text=True, timeout=3)
            windows = [window_id.strip() for window_id in matches.stdout.splitlines() if window_id.strip()]
            if matches.returncode or not windows:
                visible = subprocess.run([xdotool, "search", "--onlyvisible", "--name", ".*"],
                                         check=False, capture_output=True, text=True, timeout=3)
                titles = []
                for wid in visible.stdout.splitlines():
                    title = subprocess.run([xdotool, "getwindowname", wid], check=False,
                                           capture_output=True, text=True, timeout=2).stdout.strip()
                    if re.search(r"subway|surf|wine|unity", title, re.IGNORECASE):
                        titles.append(f"{wid}: {title}")
                detail = f" Matching visible titles: {'; '.join(titles)}" if titles else " No visible game-like window was found."
                raise SystemExit("Can't find the exact Subway Surfers window. Make sure the game is open and not minimized." + detail)
            if len(windows) != 1:
                titles = [subprocess.run([xdotool, "getwindowname", wid], check=False,
                                         capture_output=True, text=True, timeout=2).stdout.strip()
                          for wid in windows]
                raise SystemExit(f"Found multiple matching windows ({'; '.join(titles)}). Set SUBWAY_WINDOW_NAME to a regex matching only the game.")
            GAME_WINDOW_ID = windows[0]
            game_title = subprocess.run([xdotool, "getwindowname", GAME_WINDOW_ID], check=False,
                                        capture_output=True, text=True, timeout=2).stdout.strip()
            screen_w = screen_h = 0
        elif INPUT_MODE == "android":
            lines = adb("devices", capture=True).stdout.splitlines()
            devices = [line.split()[0] for line in lines[1:] if "\tdevice" in line]
            if not devices:
                raise RuntimeError("No ADB device. Start Pixel 8a and wait for Android to finish booting.")
            if len(devices) > 1:
                raise RuntimeError("More than one ADB device is connected. Leave just the emulator you want to control.")
            screen_w, screen_h = get_screen_size()
        else:
            raise SystemExit("Set SUBWAY_INPUT_MODE to 'pc' or 'android'.")
    except FileNotFoundError as exc:
        missing_tool = "xdotool" if INPUT_MODE == "pc" else "adb"
        raise SystemExit(f"{missing_tool} is missing. On Ubuntu install it with: sudo apt install {missing_tool}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ADB failed: {exc}")
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    if INPUT_MODE == "pc":
        print(f"Using PC keyboard input for game window '{game_title}' (window {GAME_WINDOW_ID})")
    else:
        print(f"Using ADB: {ADB}")
    print(f"Loading person tracker ({PERSON_MODEL}) on {DEVICE}…")
    person_model = YOLO(PERSON_MODEL)
    print("Loading YOLO hand-gesture model…")
    hand_path = hf_hub_download(HAND_MODEL_REPO, HAND_MODEL_FILE)
    hand_model = YOLO(hand_path)
    hand_names = hand_model.names
    class_ids = {label_text(hand_names, cid): int(cid) for cid in hand_names}
    required = {"palm", "middle_finger", "fist"}
    missing = required - set(class_ids)
    if missing:
        raise SystemExit(f"Hand model is missing expected classes: {', '.join(sorted(missing))}")

    camera_indices = [int(CAMERA)] if CAMERA is not None else range(6)
    camera = None
    for camera_index in camera_indices:
        candidate = cv2.VideoCapture(camera_index)
        if candidate.isOpened():
            camera = candidate
            print(f"Using webcam camera index {camera_index}")
            break
        candidate.release()
    if camera is None:
        if CAMERA is None:
            raise SystemExit("Could not open a webcam at indices 0–5. Set SUBWAY_CAMERA to your webcam's index and retry, e.g. SUBWAY_CAMERA=1 python3 yolo_phone_controller.py")
        raise SystemExit(f"Could not open webcam index {CAMERA}. Try another index, e.g. SUBWAY_CAMERA=1 python3 yolo_phone_controller.py")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    selected_id = None
    armed = False
    lock_candidate_id = None
    lock_candidate_checks = 0
    countdown_until = None
    last_selected_box = None
    tracker_config = str(Path(__file__).with_name("bytetrack_lock.yaml"))
    current_lane_zone = 1
    game_lane = 0  # Assume a new run starts in the middle lane.
    lane_candidate = None
    lane_candidate_checks = 0
    lane_zone_name = "CENTER"
    zone_debug = "ZONE: waiting for person"
    body_calibration = []
    body_baseline_height = None
    body_baseline_center_y = None
    body_candidate = None
    body_candidate_checks = 0
    body_neutral_checks = 0
    body_latched = False
    body_debug = "BODY: waiting for person"
    frame_number = 0
    controls_paused = False
    previous_clap_gap = None
    previous_clap_at = 0.0
    clap_latched = False
    gesture_name = "WAITING FOR PERSON"
    last_hand_box = None
    last_hand_seen = 0.0
    hand_candidate_label = None
    hand_candidate_checks = 0
    clap_debug = "CLAP PAIR: need 2 visible palms"
    sent_status = {"sent": "No swipe sent yet", "sent_at": 0.0}
    status_lock = threading.Lock()
    swipe_queue = queue.Queue(maxsize=4)
    stop_worker = threading.Event()
    worker = threading.Thread(target=input_worker,
                              args=(swipe_queue, stop_worker, sent_status, status_lock,
                                    INPUT_MODE, GAME_WINDOW_ID,
                                    game_title if INPUT_MODE == "pc" else None),
                              daemon=True)
    worker.start()
    fps = 0.0
    prev_frame_at = time.monotonic()

    print(f"{INPUT_MODE.upper()} input ready. Keep the Subway Surf window active while playing.")
    print("Show a fist to the camera to select yourself. After selection, controls start in 3 seconds.")
    print("Controls: camera zones = matching game lanes; jump your body = jump; crouch your body = slide; clap = pause/resume controls.")
    print("In PC mode, press Ctrl+C in this terminal to stop. A clap pauses/resumes gesture input without changing game focus.")
    print("Stand upright for one second while body calibration finishes. Returning to the center camera zone moves the runner to the middle lane.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            frame_number += 1
            frame_dt = max(.001, now - prev_frame_at)
            prev_frame_at = now
            fps = fps * .85 + (1.0 / frame_dt) * .15

            tracked = person_model.track(frame, persist=True, tracker=tracker_config,
                                         imgsz=PERSON_SIZE, conf=PERSON_CONF,
                                         classes=[0], max_det=4, verbose=False, device=DEVICE)
            result = tracked[0] if tracked else None
            people = {}
            if result is not None and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids_tensor = result.boxes.id
                if ids_tensor is not None:
                    ids = ids_tensor.int().cpu().tolist()
                    for i, pid in enumerate(ids):
                        if i < len(boxes):
                            people[int(pid)] = boxes[i]
            # Before arming, scan every visible person for a fist. Require two
            # consecutive hand checks so a noisy one-frame detection cannot lock.
            if selected_id is None and frame_number % HAND_EVERY == 0:
                fist_person = None
                for pid, person_box in people.items():
                    bx1, by1, bx2, by2 = map(int, person_box)
                    bw, bh = max(1, bx2-bx1), max(1, by2-by1)
                    rx1, rx2 = max(0, bx1-int(bw*.28)), min(frame.shape[1], bx2+int(bw*.28))
                    ry1, ry2 = max(0, by1-int(bh*.18)), min(frame.shape[0], by1+int(bh*.62))
                    roi = frame[ry1:ry2, rx1:rx2]
                    if not roi.size:
                        continue
                    result_hand = hand_model.predict(roi, imgsz=HAND_SIZE,
                        classes=[class_ids["fist"]], conf=CLAP_CONF,
                        verbose=False, device=DEVICE)[0]
                    if result_hand.boxes is not None and len(result_hand.boxes):
                        fist_person = pid
                        break
                if fist_person is not None:
                    if lock_candidate_id == fist_person:
                        lock_candidate_checks += 1
                    else:
                        lock_candidate_id = fist_person
                        lock_candidate_checks = 1
                else:
                    lock_candidate_id = None
                    lock_candidate_checks = 0
                if lock_candidate_checks >= GESTURE_CONFIRM_CHECKS:
                    selected_id = lock_candidate_id
                    countdown_until = now + 3.0
                    lock_candidate_id = None
                    print(f"Locked onto person {selected_id}; starting 3-second countdown")
            if selected_id is not None and not armed and countdown_until is not None and now >= countdown_until:
                armed = True
                print("Countdown complete; controls enabled")

            if selected_id is None:
                for pid, person_box in people.items():
                    x1, y1, x2, y2 = map(int, person_box)
                    color = (80, 180, 255) if pid == lock_candidate_id else (180, 180, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"PERSON {pid}", (x1, max(18, y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            if selected_id in people:
                last_selected_box = people[selected_id].copy()
                x1, y1, x2, y2 = map(int, people[selected_id])
                col = (60, 245, 190)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"PLAYER {selected_id}  LOCKED",
                            (x1, max(18, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1, cv2.LINE_AA)

            selected_box = people.get(selected_id) if armed else None
            if selected_id is not None and not armed and selected_id in people:
                countdown_left = max(1, int(np.ceil(countdown_until - now)))
                gesture_name = f"LOCKED TO PLAYER {selected_id} — STARTING IN {countdown_left}"
                preview_box = people[selected_id]
                px1, py1, px2, py2 = map(int, preview_box)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (70, 240, 195), 2)
            elif selected_id is None:
                gesture_name = (f"SHOW FIST TO SELECT — PERSON {lock_candidate_id} CONFIRMING"
                                if lock_candidate_id is not None else "SHOW FIST TO SELECT YOURSELF")
            if armed and selected_box is None and last_selected_box is not None:
                # Keep the lock visible during brief occlusions; never use this stale box for controls.
                ghost_x = int((last_selected_box[0] + last_selected_box[2]) / 2)
                ghost_y = int((last_selected_box[1] + last_selected_box[3]) / 2)
                cv2.circle(frame, (ghost_x, ghost_y), 9, (0, 220, 255), 2)
                cv2.putText(frame, f"PLAYER {selected_id} OCCLUDED — LOCK HELD",
                            (max(8, ghost_x - 90), max(20, ghost_y - 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 220, 255), 1, cv2.LINE_AA)
                lane_candidate = None
                lane_candidate_checks = 0
            # Keep the center lane zone narrow so smaller body shifts reach side lanes.
            frame_w = frame.shape[1]
            center_half_width = int(frame_w * CENTER_ZONE_FRACTION / 2)
            left_boundary = frame_w // 2 - center_half_width
            right_boundary = frame_w // 2 + center_half_width
            cv2.line(frame, (left_boundary, 0), (left_boundary, frame.shape[0]), (220, 220, 220), 1)
            cv2.line(frame, (right_boundary, 0), (right_boundary, frame.shape[0]), (220, 220, 220), 1)
            if selected_box is not None:
                person_center_x = int((selected_box[0] + selected_box[2]) / 2)
                person_center_y = float((selected_box[1] + selected_box[3]) / 2)
                person_height = max(1.0, float(selected_box[3] - selected_box[1]))
                if person_center_x < left_boundary:
                    detected_zone, lane_zone_name = 0, "LEFT"
                elif person_center_x >= right_boundary:
                    detected_zone, lane_zone_name = 2, "RIGHT"
                else:
                    detected_zone, lane_zone_name = 1, "CENTER"
                cv2.circle(frame, (person_center_x, int((selected_box[1]+selected_box[3])/2)),
                           7, (0, 220, 255), -1)
                if detected_zone == lane_candidate:
                    lane_candidate_checks += 1
                else:
                    lane_candidate = detected_zone
                    lane_candidate_checks = 1
                game_lane_name = "LEFT" if game_lane < 0 else "RIGHT" if game_lane > 0 else "CENTER"
                zone_debug = (f"CAM {lane_zone_name} x={person_center_x}/{frame_w} "
                              f"→ GAME {game_lane_name} confirm={lane_candidate_checks}/{LANE_ZONE_CONFIRM}")
                if lane_candidate_checks >= LANE_ZONE_CONFIRM and detected_zone != current_lane_zone:
                    target_lane = detected_zone - 1
                    lane_delta = target_lane - game_lane
                    if lane_delta and not controls_paused:
                        direction = "left" if lane_delta < 0 else "right"
                        try:
                            if swipe_queue.maxsize - swipe_queue.qsize() < abs(lane_delta):
                                raise queue.Full
                            for _ in range(abs(lane_delta)):
                                swipe_queue.put_nowait((direction, screen_w, screen_h))
                            game_lane = target_lane
                            current_lane_zone = detected_zone
                            gesture_name = f"{lane_zone_name} ZONE → LANE {direction.upper()}"
                            print(gesture_name)
                        except queue.Full:
                            gesture_name = "LANE INPUT QUEUE BUSY"
                    elif not lane_delta:
                        current_lane_zone = detected_zone
                if body_baseline_height is None:
                    body_calibration.append((person_height, person_center_y))
                    body_debug = f"BODY CALIBRATING {len(body_calibration)}/{BODY_CALIBRATION_FRAMES} — stand upright"
                    if len(body_calibration) >= BODY_CALIBRATION_FRAMES:
                        body_baseline_height = float(np.median([sample[0] for sample in body_calibration]))
                        body_baseline_center_y = float(np.median([sample[1] for sample in body_calibration]))
                        body_debug = "BODY READY — jump or crouch"
                else:
                    center_lift = (body_baseline_center_y - person_center_y) / body_baseline_height
                    height_ratio = person_height / body_baseline_height
                    body_action = "up" if center_lift > JUMP_LIFT_FRACTION else (
                        "down" if height_ratio < CROUCH_HEIGHT_FRACTION and center_lift < .05 else None)
                    body_debug = f"BODY {('JUMP' if body_action == 'up' else 'CROUCH' if body_action == 'down' else 'NEUTRAL')} " \
                                 f"lift={center_lift:+.2f} height={height_ratio:.2f}"
                    if body_action:
                        body_neutral_checks = 0
                        if body_action == body_candidate:
                            body_candidate_checks += 1
                        else:
                            body_candidate = body_action
                            body_candidate_checks = 1
                        if (body_candidate_checks >= BODY_ACTION_CONFIRM and not body_latched
                                and not controls_paused):
                            try:
                                swipe_queue.put_nowait((body_action, screen_w, screen_h))
                                body_latched = True
                                gesture_name = "BODY JUMP → JUMP" if body_action == "up" else "BODY CROUCH → SLIDE"
                            except queue.Full:
                                gesture_name = "BODY INPUT QUEUE BUSY"
                    else:
                        body_candidate = None
                        body_candidate_checks = 0
                        body_neutral_checks += 1
                        if body_neutral_checks >= BODY_NEUTRAL_CHECKS:
                            body_latched = False
                            # Slowly follow camera auto-exposure/position drift while upright.
                            body_baseline_height = .98 * body_baseline_height + .02 * person_height
                            body_baseline_center_y = .98 * body_baseline_center_y + .02 * person_center_y
            if armed and selected_box is None:
                gesture_name = "LOCKED PERSON NOT VISIBLE"
            hand_checked = frame_number % HAND_EVERY == 0 and selected_box is not None
            observed = []

            if hand_checked:
                # Crop to the locked player's upper body so the hand is larger for YOLO.
                bx1, by1, bx2, by2 = map(int, selected_box)
                bw, bh = max(1, bx2-bx1), max(1, by2-by1)
                rx1 = max(0, bx1 - int(bw*.28))
                rx2 = min(frame.shape[1], bx2 + int(bw*.28))
                ry1 = max(0, by1 - int(bh*.18))
                ry2 = min(frame.shape[0], by1 + int(bh*.62))
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size:
                    # A lower detector threshold helps see both palms during a quick clap.
                    hand_result = hand_model.predict(roi, imgsz=HAND_SIZE,
                                                      classes=[class_ids[x] for x in ("palm", "middle_finger")],
                                                      conf=CLAP_CONF, verbose=False, device=DEVICE)[0]
                    if hand_result.boxes is not None:
                        for hand_box in hand_result.boxes:
                            cls_id = int(hand_box.cls[0])
                            conf = float(hand_box.conf[0])
                            label = label_text(hand_names, cls_id)
                            hx1, hy1, hx2, hy2 = map(float, hand_box.xyxy[0].cpu().tolist())
                            hx1 += rx1; hx2 += rx1; hy1 += ry1; hy2 += ry1
                            hand_w, hand_h = hx2-hx1, hy2-hy1
                            roi_w, roi_h = rx2-rx1, ry2-ry1
                            # Discard huge shirt/background detections and tiny noise boxes.
                            if hand_w < 10 or hand_h < 10 or hand_w > roi_w*.70 or hand_h > roi_h*.65:
                                continue
                            observed.append((label, conf, (hx1, hy1, hx2, hy2)))

            # Persist a hand box only after its class is consistent across checks.
            if hand_checked:
                if observed:
                    frame_best = max(observed, key=lambda item: item[1])
                    if frame_best[0] == hand_candidate_label:
                        hand_candidate_checks += 1
                    else:
                        hand_candidate_label = frame_best[0]
                        hand_candidate_checks = 1
                    if hand_candidate_checks >= GESTURE_CONFIRM_CHECKS:
                        last_hand_box = frame_best
                        last_hand_seen = now
                else:
                    hand_candidate_label = None
                    hand_candidate_checks = 0

            if hand_checked:
                # A clap is an open-palm pair moving quickly from spread apart to together.
                # Trigger once, then require the hands to separate before another clap can count.
                palms = [(box, conf) for label, conf, box in observed
                         if label == "palm" and conf >= CLAP_CONF]
                clap_gap = None
                if len(palms) >= 2 and selected_box is not None:
                    # Use the two strongest detections, then measure their horizontal gap.
                    strongest = sorted(palms, key=lambda item: item[1], reverse=True)[:2]
                    centers = sorted(((b[0]+b[2])/2 for b, _ in strongest))
                    clap_gap = (centers[1] - centers[0]) / max(1.0, float(selected_box[2]-selected_box[0]))
                    clap_debug = f"CLAP PALMS: 2+  GAP: {clap_gap:.2f} (spread > .24, together < .13)"
                else:
                    clap_debug = f"CLAP PALMS: {len(palms)} — show both hands apart"

                if clap_gap is not None:
                    if clap_gap > .24:
                        clap_latched = False
                        previous_clap_gap = clap_gap
                        previous_clap_at = now
                    elif (not clap_latched and previous_clap_gap is not None
                          and previous_clap_gap > .24 and clap_gap < .13
                          and now - previous_clap_at < .70):
                        clap_latched = True
                        if INPUT_MODE == "pc":
                            was_paused = controls_paused
                            controls_paused = not controls_paused
                            if was_paused:
                                # A new Subway Surf run starts in the center lane. Resync
                                # our estimate after the player restarts following a death.
                                game_lane = 0
                                current_lane_zone = 1
                                lane_candidate = None
                                lane_candidate_checks = 0
                                gesture_name = "CLAP — RESUMED; LANE STATE RESET TO CENTER"
                            else:
                                gesture_name = "CLAP — GESTURE INPUT PAUSED"
                            with status_lock:
                                sent_status["sent"] = ("GESTURES PAUSED" if controls_paused
                                                        else "GESTURES RESUMED; LANE RESET")
                        else:
                            controls_paused = True
                            gesture_name = "CLAP DETECTED — PAUSE GAME; PRESS R AFTER RESUMING"
                            try:
                                swipe_queue.put_nowait(("pause", screen_w, screen_h))
                            except queue.Full:
                                gesture_name = "CLAP DETECTED — INPUT QUEUE BUSY; GAME CONTROL PAUSED"
                        print(gesture_name)

            # Draw all current detections so it is clear whether YOLO sees one or both hands.
            if hand_checked:
                for label, conf, box in observed:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 210, 255), 1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(18, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .38, (90, 210, 255), 1, cv2.LINE_AA)
            # Keep the stable best hand label visible briefly, avoiding label flicker.
            if last_hand_box is not None and now-last_hand_seen < .40:
                label, conf, box = last_hand_box
                x1, y1, x2, y2 = map(int, box)
                col = (0, 80, 255) if label == "middle_finger" else (255, 190, 40)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(18, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .44, col, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0,0), (frame.shape[1],158), (10,22,27), -1)
            cv2.putText(frame, "YOLO SUBWAY CONTROLS | person tracking", (12,23), cv2.FONT_HERSHEY_SIMPLEX,
                        .53, (245,250,250), 1, cv2.LINE_AA)
            cv2.putText(frame, f"TARGET {selected_id if selected_id is not None else 'searching'} | {gesture_name}",
                        (12,49), cv2.FONT_HERSHEY_SIMPLEX, .43, (70,240,195), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{DEVICE.upper()} {fps:.1f} FPS | {sent_status['sent']}",
                        (12,74), cv2.FONT_HERSHEY_SIMPLEX, .36, (240,240,230), 1, cv2.LINE_AA)
            cv2.putText(frame, zone_debug, (12,91), cv2.FONT_HERSHEY_SIMPLEX, .36,
                        (100,210,255), 1, cv2.LINE_AA)
            cv2.putText(frame, body_debug, (12,108), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (100,240,190), 1, cv2.LINE_AA)
            cv2.putText(frame, clap_debug, (12,125), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (100,210,255), 1, cv2.LINE_AA)
            if controls_paused:
                paused_message = ("GESTURES PAUSED — clap again to resume" if INPUT_MODE == "pc"
                                  else "GAME PAUSED — resume, then press R here")
                cv2.putText(frame, paused_message, (12,146),
                            cv2.FONT_HERSHEY_SIMPLEX, .32, (100,190,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Zone = lane | jump = jump | crouch = slide | clap = toggle gesture control",
                        (12,146), cv2.FONT_HERSHEY_SIMPLEX, .32, (210,220,225), 1, cv2.LINE_AA)
            cv2.imshow("YOLO Subway Surfers — hand gesture debug", frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("r") and controls_paused:
                controls_paused = False
                previous_clap_gap = None
                clap_latched = False
                gesture_name = "CONTROL RESUMED"
    finally:
        camera.release()
        try:
            swipe_queue.put_nowait(None)
        except queue.Full:
            pass
        stop_worker.set()
        worker.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
