#!/usr/bin/env python3
import cv2
import pytesseract
import subprocess
import os
import time

def speak(text):
    """Convert text to speech"""
    if text.strip():
        try:
            subprocess.run(['pico2wave', '-w', '/tmp/ocr_speech.wav', text], 
                          check=True, capture_output=True)
            subprocess.run(['aplay', '-q', '/tmp/ocr_speech.wav'], 
                          check=True)
            os.remove('/tmp/ocr_speech.wav')
        except Exception as e:
            print(f"Error: {e}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_spoken = ""
last_speak_time = 0
speak_delay = 5  # Seconds between auto-speak

print("Auto-speaking OCR active")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    x1, y1 = width//4, height//4
    x2, y2 = 3*width//4, 3*height//4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Extract and process ROI
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
    
    # Auto-speak if text changed and enough time passed
    current_time = time.time()
    if text and text != last_spoken:
        if current_time - last_speak_time > speak_delay:
            print("Speaking:", text)
            cv2.putText(frame, "Speaking...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Auto-Speaking OCR', frame)
            cv2.waitKey(1)
            
            speak(text)
            last_spoken = text
            last_speak_time = current_time
    
    # Display text
    if text:
        cv2.putText(frame, text[:40], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Auto-Speaking OCR', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()