#!/usr/bin/env python3
# YOLO object detection (zone-gated) + Ultrasonic distance + pico2wave TTS
# Speaks: "<names> detected at <N> centimeters far."

from ultralytics import YOLO
import cv2
import time
import subprocess
import threading
import queue
import os
import RPi.GPIO as GPIO

# -------------------- YOLO CONFIG --------------------
MODEL_WEIGHTS = "yolo11n.pt"
CAM_INDEX = 0

# Zone (normalized 0..1)
zone_x_min = 0.2
zone_x_max = 0.8
zone_y_min = 0.2
zone_y_max = 0.8

IMG_SIZE = 320
CONF_TH = 0.35

# Pico2Wave TTS
PICO_LANG = "en-GB"   # en-GB or en-US
COOLDOWN_S = 3.0      # seconds between announcements
# -----------------------------------------------------

# ---------- TTS worker (non-blocking) -----------
speak_q = queue.Queue()

def tts_worker():
    while True:
        text = speak_q.get()
        if text is None:
            break
        wav_path = "tts_out.wav"
        try:
            subprocess.run(["pico2wave", "-l", PICO_LANG, "-w", wav_path, text], check=True)
            subprocess.run(["aplay", "-q", wav_path], check=False)
        except Exception as e:
            print("[TTS ERROR]", e)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            speak_q.task_done()

def speak(text):
    text = text.replace("_", " ")
    print("ðŸ”Š", text)
    speak_q.put(text)

threading.Thread(target=tts_worker, daemon=True).start()
# ------------------------------------------------

def phrase_with_and(names):
    """Return a clear phrase with commas and 'and'."""
    names = list(dict.fromkeys(names))  # remove duplicates, preserve order
    n = len(names)
    if n == 1:
        return names[0]
    if n == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]

# ---------------- Ultrasonic setup ----------------
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(2)

last_spoken = None      # last distance spoken
last_time = 0           # time of last distance speech
# --------------------------------------------------

# Load YOLO and camera
model = YOLO(MODEL_WEIGHTS)
cap = cv2.VideoCapture(CAM_INDEX)

last_spoken_time = 0  # YOLO cooldown tracker

print("Running: YOLO zone detection + Ultrasonic distance + TTS (press 'q' to quit)")

try:
    while True:
        # ----- Capture frame -----
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_TH, device="cpu", verbose=False)
        annotated = frame.copy()

        # draw zone rectangle
        zx1, zy1 = int(zone_x_min * w), int(zone_y_min * h)
        zx2, zy2 = int(zone_x_max * w), int(zone_y_max * h)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

        now = time.time()
        names_in_zone = []

        # ----- YOLO: collect objects whose center lies inside the zone -----
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            name = model.names[cls]

            # bbox center
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            cx_norm = cx / w
            cy_norm = cy / h

            if (zone_x_min <= cx_norm <= zone_x_max) and (zone_y_min <= cy_norm <= zone_y_max):
                names_in_zone.append(name)
                # draw box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, name, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ----- Ultrasonic distance measurement -----
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = round((pulse_duration * 34300) / 2, 2)
        print(f"Distance: {distance} cm")

        # ----- Speaking logic: requires BOTH conditions -----
        yolo_ready = bool(names_in_zone) and (now - last_spoken_time > COOLDOWN_S)

        ultrasonic_ready = False
        if distance <= 85:
            if (last_spoken is None) or ((abs(distance - last_spoken) > 2 and time.time() - last_time > 2) \
                                         or (time.time() - last_time > 10)):
                ultrasonic_ready = True

        if yolo_ready and ultrasonic_ready:
            # ðŸ‘‡ CHANGED: include object names
            names_phrase = phrase_with_and(names_in_zone)
            speak(f"{names_phrase} detected at {int(distance)} centimeters far.")
            last_spoken_time = now
            last_spoken = distance
            last_time = time.time()

        # ----- Show window & key handling -----
        cv2.imshow("YOLO Zone + Ultrasonic Distance (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    speak_q.put(None)  # stop TTS worker
    GPIO.cleanup()