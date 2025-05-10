import cv2
import time
import pyttsx3
from vision_agent import analyze_image_for_navigation

# Setup pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def activate_proactive_system(prompt=None):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera not accessible.")
        return

    print("Capturing image for proactive system...")
    
    cam.read()  # Warm-up
    time.sleep(2)
    ret, frame = cam.read()

    if ret:
        path = "captured_frame.jpg"
        cv2.imwrite(path, frame)
        print("Image captured.")

        response = analyze_image_for_navigation(path, prompt=prompt)
        print("\nProactive Response:\n", response)

        speak(response)
    else:
        print("Capture failed.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    activate_proactive_system() 