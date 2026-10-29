"""A small Subway Surfers-style runner controlled by YOLO pose gestures."""
import math
import random
import time

import cv2
import numpy as np
from ultralytics import YOLO

WIDTH, HEIGHT = 1100, 680
CAMERA = 0
MODEL = "yolo11n-pose.pt"  # Ultralytics downloads this model the first time it runs.
LANES = [WIDTH // 2 - 130, WIDTH // 2, WIDTH // 2 + 130]


def text(img, s, xy, scale=.7, color=(240, 248, 250), thick=2):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    print("Loading YOLO pose model. First run may download yolo11n-pose.pt.")
    model = YOLO(MODEL)
    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Could not open webcam. Check CAMERA=0 in subway_yolo.py.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    lane, score, coins = 1, 0, 0
    obstacles = []
    speed, spawn = 260.0, 0.5
    jump_until = slide_until = 0.0
    last_gesture = 0.0
    gesture_lock = False
    neutral_frames = 0
    baseline_x = None
    running, ended = False, False
    best = 0
    last = time.monotonic()
    rng = random.Random()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Webcam frame read failed.")
            break
        frame = cv2.flip(frame, 1)
        now = time.monotonic()
        dt = min(now - last, .06)
        last = now

        # YOLO pose inference: person is tracked from COCO keypoints, no keyboard injection.
        results = model.predict(frame, imgsz=416, conf=.35, verbose=False, device="cpu")
        pose = None
        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps = results[0].keypoints.data.cpu().numpy()
            if len(kps):
                pose = max(kps, key=lambda k: float(np.mean(k[:, 2])))
                for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
                    if pose[a, 2] > .25 and pose[b, 2] > .25:
                        cv2.line(frame, tuple(pose[a, :2].astype(int)), tuple(pose[b, :2].astype(int)), (65, 235, 195), 3)
                for x, y, conf in pose:
                    if conf > .25:
                        cv2.circle(frame, (int(x), int(y)), 4, (30, 190, 255), -1)

        gesture = "WAITING FOR PERSON"
        if pose is not None:
            # Keypoints: nose 0, shoulders 5/6, wrists 9/10, hips 11/12.
            def kp(i): return pose[i] if pose[i, 2] > .25 else None
            ls, rs, nose, lw, rw, lh, rh = (kp(i) for i in [5, 6, 0, 9, 10, 11, 12])
            if ls is not None and rs is not None:
                shoulder_x = (ls[0] + rs[0]) / 2 / frame.shape[1]
                if baseline_x is None:
                    baseline_x = shoulder_x
                # Smooth the neutral center slowly, except while a deliberate lean is happening.
                lean = shoulder_x - baseline_x
                hands_up = (nose is not None and lw is not None and rw is not None and
                            lw[1] < nose[1] - 12 and rw[1] < nose[1] - 12)
                crouch = False
                if lh is not None and rh is not None:
                    torso = ((lh[1] + rh[1]) - (ls[1] + rs[1])) / 2
                    crouch = torso < frame.shape[0] * .20
                if hands_up: gesture = "JUMP: BOTH HANDS UP"
                elif crouch: gesture = "SLIDE: CROUCH"
                elif lean < -.09: gesture = "LEFT: LEAN LEFT"
                elif lean > .09: gesture = "RIGHT: LEAN RIGHT"
                else: gesture = "READY"

                active = gesture not in ("READY", "WAITING FOR PERSON")
                if not active:
                    neutral_frames += 1
                    if neutral_frames >= 8:
                        gesture_lock = False
                        baseline_x = baseline_x * .96 + shoulder_x * .04
                else:
                    neutral_frames = 0
                if active and not gesture_lock and now - last_gesture > .35:
                    if gesture.startswith("LEFT"): lane = max(0, lane - 1)
                    elif gesture.startswith("RIGHT"): lane = min(2, lane + 1)
                    elif gesture.startswith("JUMP"): jump_until = now + .72; slide_until = 0
                    elif gesture.startswith("SLIDE"): slide_until = now + .65; jump_until = 0
                    last_gesture = now
                    gesture_lock = True

        if running:
            score += dt * speed * .10
            speed = min(490, 260 + score * .035)
            spawn -= dt
            if spawn <= 0:
                typ = rng.choices(["coin", "barrier", "train", "bar"], [32, 28, 20, 20])[0]
                obstacles.append({"lane": rng.randrange(3), "type": typ, "y": 190, "hit": False})
                spawn = max(.48, 1.05 - score / 1500)
            for obj in obstacles:
                obj["y"] += speed * dt
                if not obj["hit"] and 416 < obj["y"] < 475 and obj["lane"] == lane:
                    if obj["type"] == "coin": coins += 1; obj["hit"] = True
                    elif obj["type"] == "barrier" and now > jump_until: ended = True
                    elif obj["type"] == "bar" and now > slide_until: ended = True
                    elif obj["type"] == "train": ended = True
            obstacles = [o for o in obstacles if o["y"] < HEIGHT + 40 and not o["hit"]]
            if ended:
                best = max(best, int(score)); running = False

        # Game view
        game = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        for y in range(HEIGHT):
            q = y / HEIGHT
            game[y, :] = (int(56 - 25*q), int(48 + 24*q), int(24 + 19*q))
        # skyline silhouettes
        for i in range(14):
            x = i * 84 - int(score * .16) % 84
            h = 65 + (i * 47 % 105)
            cv2.rectangle(game, (x, 205-h), (x+58, 205), (40, 54, 58), -1)
            for wy in range(215-h, 200, 25): cv2.rectangle(game, (x+12, wy), (x+18, wy+9), (25, 170, 218), -1)
        cv2.fillPoly(game, [np.array([[390,205],[710,205],[1010,HEIGHT],[90,HEIGHT]])], (63, 59, 51))
        for lx in [350, 450, 550, 650, 750]:
            bottom = WIDTH//2 + (lx-WIDTH//2)*3
            cv2.line(game, (lx,205), (bottom,HEIGHT), (151,156,147), 5)
        for sy in range(245, HEIGHT, 45):
            cv2.line(game, (125, sy), (975, sy), (96, 94, 82), 2)
        for obj in sorted(obstacles, key=lambda o:o["y"]):
            z = max(.05, (obj["y"]-190)/(HEIGHT-190))
            x = int(WIDTH/2 + (LANES[obj["lane"]]-WIDTH/2)*z)
            size = int(14 + z*58); y = int(obj["y"])
            if obj["type"] == "coin":
                cv2.circle(game, (x,y), max(5,size//3), (40,207,255), -1)
                cv2.circle(game, (x,y), max(3,size//3-3), (100,238,255), 2)
            else:
                col = (70,90,230) if obj["type"] == "train" else (70,120,240) if obj["type"] == "barrier" else (40,180,230)
                cv2.rectangle(game, (x-size//2,y-size), (x+size//2,y), col, -1, cv2.LINE_AA)
                if obj["type"] == "bar": cv2.rectangle(game,(x-size//2,y-size),(x+size//2,y-size//2),(190,225,245),-1)
        px = LANES[lane]; jumping = now < jump_until; sliding = now < slide_until
        py = 430 - (int(86*math.sin(math.pi*(jump_until-now)/.72)) if jumping else 0)
        cv2.ellipse(game,(px,470),(38,10),0,0,360,(15,20,23),-1)
        if sliding:
            cv2.rectangle(game,(px-32,py-3),(px+32,py+29),(50,215,185),-1)
            cv2.circle(game,(px+23,py-3),13,(175,198,255),-1)
        else:
            cv2.rectangle(game,(px-20,py-5),(px+20,py+42),(50,215,185),-1)
            cv2.circle(game,(px,py-22),15,(175,198,255),-1)
            cv2.line(game,(px-13,py+40),(px-19,py+58),(40,155,235),8)
            cv2.line(game,(px+13,py+40),(px+19,py+58),(40,155,235),8)
        text(game, f"SCORE {int(score)}     COINS {coins}     BEST {best}",(25,40),.75)
        text(game, "YOLO MOTION: " + gesture, (25,75),.65,(75,245,205))
        text(game, "Q quit | arrows move | space jump | down slide | R restart",(24,HEIGHT-20),.48,(210,215,205),1)
        if not running:
            shade=game.copy();cv2.rectangle(shade,(0,0),(WIDTH,HEIGHT),(10,20,24),-1);game=cv2.addWeighted(shade,.64,game,.36,0)
            title="RUN OVER" if ended else "SUBWAY MOTION RUNNER"
            text(game,title,(WIDTH//2-200,HEIGHT//2-32),1.15,(245,250,248),3)
            text(game,"Move naturally in front of the webcam. Press ENTER to start.",(WIDTH//2-295,HEIGHT//2+20),.62)
        # webcam inset
        fh, fw = frame.shape[:2]; thumb=cv2.resize(frame,(240,180))
        game[HEIGHT-195:HEIGHT-15,WIDTH-255:WIDTH-15]=thumb
        cv2.rectangle(game,(WIDTH-256,HEIGHT-196),(WIDTH-14,HEIGHT-14),(220,230,230),2)
        text(game,"YOLO POSE CAMERA",(WIDTH-250,HEIGHT-202),.42,(220,230,230),1)
        cv2.imshow("Subway Motion — YOLO Pose Controller", game)
        key=cv2.waitKey(1)&0xff
        if key in (ord('q'),27): break
        if key in (10,13) and not running:
            running=True;ended=False;score=0;coins=0;obstacles=[];spawn=.4;speed=260;lane=1
        if key==ord('r'):
            running=True;ended=False;score=0;coins=0;obstacles=[];spawn=.4;speed=260;lane=1
        if key==81 or key==ord('a'): lane=max(0,lane-1)
        if key==83 or key==ord('d'): lane=min(2,lane+1)
        if key==82 or key==ord(' '): jump_until=time.monotonic()+.72;slide_until=0
        if key==84: slide_until=time.monotonic()+.65;jump_until=0
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
