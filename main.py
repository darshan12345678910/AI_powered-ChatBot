import cv2
import torch
import time
import pyttsx3
import speech_recognition as sr
from ultalytics import YOLO
from proactive import activate_proactive_system
from ocr import ICr

# Load YOLOv10 model
model = YOLO("weights/yolov10s.pt")

# Constants
KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 500
CHECK_INTERVAL = 10
DETECTION_THRESHOLD = 0.5

# Text-to-speech initialization
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech Error:", e)

def recognize_speech():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.6
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            return recognizer.recognize_google(audio).lower()
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            print("Speech not recognized.")
        except sr.RequestError:
            print("Network error.")
    return ""

def estimate_distance(pixel_width):
    return float("inf") if pixel_width == 0 else (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def classify_objects(results, frame_width):
    positions = {"left": [], "center": [], "right": []}
    distances = {}
    left, right = frame_width * 0.33, frame_width * 0.66

    for result in results:
        for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            if conf < DETECTION_THRESHOLD:
                continue
            obj = result.names[int(cls)]
            x_min, _, x_max, _ = xyxy.tolist()
            pixel_width = x_max - x_min
            distance = estimate_distance(pixel_width)
            distances[obj] = distance

            if x_max < left:
                positions["left"].append(obj)
            elif x_min > right:
                positions["right"].append(obj)
            else:
                positions["center"].append(obj)
    return positions, distances, bool(distances)

def generate_feedback(positions, distances):
    description = []

    for side, objs in positions.items():
        if objs:
            count_desc = ', '.join(f"{objs.count(o)} {o}{'s' if objs.count(o) > 1 else ''}" for o in set(objs))
            description.append(f"{count_desc} on the {side}.")

    for obj, dist in distances.items():
        if dist < 0.5:
            description.append(f"Warning! {obj} is too close.")
        elif dist < 1:
            description.append(f"{obj} is close.")

    final_msg = " ".join(description) or "No significant objects detected."
    speak(final_msg)

def detect_objects_realtime():
    cap = cv2.VideoCapture(0)
    is_active = False
    last_check = time.time()

    while True:
        print("\nSay 'start detection', 'stop detection', 'exit', or 'activate proactive'.")
        command = recognize_speech()

        if "start detection" in command:
            is_active = True
            speak("Object detection activated.")

        elif "activate proactive" in command or "emergency" in command:
            speak("Activating proactive vision system.")
            activate_proactive_system()

        elif "OCR" in command or "readit" in command:
            speak("Activating proactive vision system.")
            ICr()
        elif "stop detection" in command:
            is_active = False
            speak("Object detection stopped.")
        elif "exit" in command:
            break

        while is_active:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            frame_width = frame.shape[1]
            results = model.predict(frame)
            positions, distances, found = classify_objects(results, frame_width)

            if found:
                generate_feedback(positions, distances)

            if time.time() - last_check > CHECK_INTERVAL:
                command = recognize_speech()
                if "stop detection" in command or "exit" in command:
                    is_active = False
                    speak("Object detection stopped.")
                last_check = time.time()

            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_objects_realtime()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        cv2.destroyAllWindows()
