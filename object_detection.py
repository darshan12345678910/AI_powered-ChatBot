# object_detection.py
import cv2
import torch
import speech_recognition as sr
from ultralytics import YOLO
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import time
import pyttsx3
from proactive import activate_proactive_system

# Load YOLOv10 Model
model = YOLO("weights/yolov10s.pt")

# Constants for Depth Estimation
KNOWN_WIDTH = 0.5  
FOCAL_LENGTH = 500 

# Global variable to cache the last spoken message
last_description = ""

# Voice recognition function
def recognize_speech():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.6

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            return command
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            return ""
        except sr.RequestError:
            print("Network error. Check your internet connection.")
            return ""
        except sr.WaitTimeoutError:
            print("No speech detected, please try again.")
            return ""

# Distance estimation function
def estimate_distance(pixel_width):
    if pixel_width == 0:
        return float("inf")
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

# Object position classification with distance estimation
def classify_object_positions(results, frame_width):
    object_positions = {"left": [], "center": [], "right": []}
    detected_anything = False
    distances = {}

    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.66

    for result in results:
        detected_boxes = result.boxes
        detected_classes = detected_boxes.cls
        detected_confidences = detected_boxes.conf
        detected_xyxy = detected_boxes.xyxy

        object_names = [result.names[int(cls.item())] for cls in detected_classes]
        threshold = 0.5
        filtered_objects = [(obj, conf.item(), xyxy) for obj, conf, xyxy in zip(object_names, detected_confidences, detected_xyxy) if conf.item() >= threshold]

        if filtered_objects:
            detected_anything = True
            for obj, _, xyxy in filtered_objects:
                x_min, _, x_max, _ = xyxy.tolist()
                pixel_width = x_max - x_min
                distance = estimate_distance(pixel_width)

                distances[obj] = distance

                if x_max < left_boundary:
                    object_positions["left"].append(obj)
                elif x_min > right_boundary:
                    object_positions["right"].append(obj)
                else:
                    object_positions["center"].append(obj)

    return object_positions, distances, detected_anything

# Generate and play audio feedback
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_file = "proactive_response.wav"
        tts.save(audio_file)
        data, samplerate = sf.read(audio_file)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print("Speech generation failed:", e)
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)  # Increase speech speed
    engine.say(text)
    engine.runAndWait()

def generate_audio_feedback(object_positions, distances):
    """Generate and play spoken feedback with simple proximity alerts."""
    environment_description = []

    for position, objs in object_positions.items():
        if objs:
            obj_counts = {obj: objs.count(obj) for obj in set(objs)}
            obj_list = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in obj_counts.items()]
            environment_description.append(f"{', '.join(obj_list)} on the {position}.")

    if not environment_description:
        environment_description.append("No significant objects detected.")

    # Add proximity alerts (no numeric distances)
    for obj, distance in distances.items():
        if distance < 0.5:
            environment_description.append(f"Warning! {obj} is too close.")
        elif distance < 1:
            environment_description.append(f"{obj} is close.")

    full_description = " ".join(environment_description)
    
    # Convert text to speech
    tts = gTTS(text=full_description, lang='en')
    audio_file = "environment_description.wav"
    tts.save(audio_file)

    # Load and play audio
    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()

# Main function for real-time detection
def detect_objects_realtime():
    cap = cv2.VideoCapture(0)
    is_active = False
    last_check_time = time.time()

    while True:
        print("\nSay 'start detection' to begin, 'stop detection' to exit, or 'exit' to quit.")
        command = recognize_speech()

        if "start detection" in command:
            is_active = True
            print("Object detection activated. Say 'stop detection' to stop.")
            speak("Object detection activated. Say 'stop detection' to stop.")
            
        elif "activate proactive" in command or "emergency" in command:
            print("Activating proactive vision system...")
            speak("Activating proactive vision system...")
            activate_proactive_system()
        elif "stop detection" in command:
            is_active = False
            print("Object detection stopped.")
        elif "exit" in command:
            print("Exiting program.")
            break

        while is_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            frame_height, frame_width, _ = frame.shape
            results = model.predict(frame)
            object_positions, distances, detected_anything = classify_object_positions(results, frame_width)

            if detected_anything:
                generate_audio_feedback(object_positions, distances)
            else:
                print("No significant objects detected.")

            # Check voice command every 10 seconds
            current_time = time.time()
            if current_time - last_check_time > 10:
                command = recognize_speech()
                if "stop detection" in command or "exit" in command:
                    is_active = False
                    print("Object detection stopped.")
                last_check_time = current_time

            time.sleep(0.1)  # Reduce CPU usage

    cap.release()
    cv2.destroyAllWindows()

# Run main
if __name__ == "__main__":
    try:
        detect_objects_realtime()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        cv2.destroyAllWindows()
