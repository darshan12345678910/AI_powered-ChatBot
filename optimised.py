import cv2
import torch
import time
import pyttsx3
import speech_recognition as sr
import threading
from proactive import activate_proactive_system
from ocr import ICr
import torchvision.ops as ops
from collections import Counter

# Load TorchScript YOLOv10 model
model = torch.jit.load("weights/yolov10s.torchscript")
model.eval()

# Constants
KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 500
CHECK_INTERVAL = 10
DETECTION_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
FRAME_SKIP = 3  # Skip 3 frames between processing

CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech Error:", e)

def postprocess(outputs, conf_threshold=0.25, iou_threshold=0.45):
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(0)

    mask = outputs[:, 4] > conf_threshold
    detections = outputs[mask]
    if detections.shape[0] == 0:
        return []
    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5].int()
    keep = ops.nms(boxes, scores, iou_threshold)
    detections = detections[keep]
    results = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        results.append({
            'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
            'conf': conf.item(),
            'class_id': int(cls_id.item())
        })
    return results

def format_yolo_output(detections, frame_height, frame_width, total_time_ms, preprocess_time_ms, inference_time_ms, postprocess_time_ms):
    """Format output in YOLOv10 style"""
    if not detections:
        print(f"0: {frame_height}x{frame_width} (no detections), {total_time_ms:.1f}ms")
    else:
        # Count objects by class
        class_counts = Counter()
        for det in detections:
            class_name = CLASS_NAMES.get(det["class_id"], f"object_{det['class_id']}")
            class_counts[class_name] += 1
        
        # Format detection summary
        detection_summary = ", ".join([f"{count} {cls}" + ("s" if count > 1 else "") 
                                     for cls, count in class_counts.items()])
        
        print(f"0: {frame_height}x{frame_width} {detection_summary}, {total_time_ms:.1f}ms")
    
    # Speed summary
    print(f"Speed: {preprocess_time_ms:.1f}ms preprocess, {inference_time_ms:.1f}ms inference, {postprocess_time_ms:.1f}ms postprocess per image at shape (1, 3, {frame_height}, {frame_width})")

def estimate_distance(pixel_width):
    return float("inf") if pixel_width == 0 else (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def classify_objects(detections, frame_width):
    positions = {"left": [], "center": [], "right": []}
    distances = {}
    left_bound = frame_width * 0.33
    right_bound = frame_width * 0.66

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        pixel_width = x2 - x1
        distance = estimate_distance(pixel_width)
        label = CLASS_NAMES.get(det["class_id"], f"object_{det['class_id']}")

        distances[label] = distance
        center_x = (x1 + x2) / 2
        if center_x < left_bound:
            positions["left"].append(label)
        elif center_x > right_bound:
            positions["right"].append(label)
        else:
            positions["center"].append(label)

    return positions, distances, bool(distances)

def generate_feedback(positions, distances):
    description = []
    for side, objs in positions.items():
        if objs:
            counts = [f"{objs.count(o)} {o}{'s' if objs.count(o) > 1 else ''}" for o in set(objs)]
            count_desc = ", ".join(counts)
            description.append(f"{count_desc} on the {side}.")

    for obj, dist in distances.items():
        if dist < 0.5:
            description.append(f"Warning! {obj} is too close.")
        elif dist < 1:
            description.append(f"{obj} is close.")

    final_msg = " ".join(description) or "No significant objects detected."
    speak(final_msg)

# Global command variable and lock for thread-safe access
command = ""
command_lock = threading.Lock()

def speech_listener():
    global command
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.6

    while True:
        with sr.Microphone() as source:
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {text}")
                with command_lock:
                    command = text
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                pass
            except sr.RequestError:
                print("Network error.")
            time.sleep(0.1)

total_frames = 0
detected_frames = 0

def detect_objects_realtime():
    global command, total_frames, detected_frames
    cap = cv2.VideoCapture(0)
    is_active = False
    last_check = time.time()
    frame_count = 0

    listener_thread = threading.Thread(target=speech_listener, daemon=True)
    listener_thread.start()

    while True:
        with command_lock:
            cmd = command
            command = ""

        if "start detection" in cmd:
            is_active = True
            speak("Object detection activated.")
        elif "activate proactive" in cmd or "emergency" in cmd:
            speak("Activating proactive vision system.")
            activate_proactive_system()
        elif "ocr" in cmd or "readit" in cmd:
            speak("Activating OCR.")
            ICr()
        elif "stop detection" in cmd:
            is_active = False
            speak("Object detection stopped.")
        elif "exit" in cmd:
            speak("Exiting program.")
            break

        while is_active:
            ret, frame = cap.read()
            if not ret:
                continue

            # Check commands inside the detection loop
            with command_lock:
                inner_cmd = command
                command = ""

            if "exit" in inner_cmd:
                speak("Exiting program.")
                is_active = False
                break
            elif "stop detection" in inner_cmd:
                speak("Object detection stopped.")
                is_active = False
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue  # Skip this frame

            total_frames += 1

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            start_preprocess = time.time()
            img = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            preprocess_time = time.time() - start_preprocess

            with torch.no_grad():
                start_inference = time.time()
                output = model(img_tensor)[0]
                inference_time = time.time() - start_inference

                start_postprocess = time.time()
                detections = postprocess(output, conf_threshold=DETECTION_THRESHOLD, iou_threshold=IOU_THRESHOLD)
                positions, distances, found = classify_objects(detections, frame_width)
                postprocess_time = time.time() - start_postprocess

            # Convert times to milliseconds
            preprocess_time_ms = preprocess_time * 1000
            inference_time_ms = inference_time * 1000
            postprocess_time_ms = postprocess_time * 1000
            total_time_ms = preprocess_time_ms + inference_time_ms + postprocess_time_ms

            # Print YOLOv10 format output
            format_yolo_output(detections, frame_height, frame_width, total_time_ms, 
                             preprocess_time_ms, inference_time_ms, postprocess_time_ms)

            if found:
                detected_frames += 1
                generate_feedback(positions, distances)
                last_check = time.time()

            if time.time() - last_check > CHECK_INTERVAL:
                with command_lock:
                    check_cmd = command
                    command = ""
                if "stop detection" in check_cmd or "exit" in check_cmd:
                    is_active = False
                    speak("Object detection stopped.")
                    if "exit" in check_cmd:
                        break
                last_check = time.time()

            time.sleep(0.1)

        if not is_active and "exit" in cmd:
            break

    cap.release()

    if total_frames > 0:
        detection_accuracy = detected_frames / total_frames * 100
        print("\n--- Final Results ---")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with detections: {detected_frames} ({detection_accuracy:.2f}%)")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_objects_realtime()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        cv2.destroyAllWindows()
