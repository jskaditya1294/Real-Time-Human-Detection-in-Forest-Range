import os
import time
import cv2
import pytesseract
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from twilio.rest import Client

# # Set Tesseract path (Windows only, update this path if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pytesseract

# Set the correct Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Twilio Configuration
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox/Verified Number
YOUR_WHATSAPP_NUMBER = "whatsapp:+91"  # Replace with your WhatsApp number

# Load YOLO model
model = YOLO("yolo11m.pt")

def extract_datetime_from_image(image):
    """Extracts date and time text from the bottom of the image using OCR."""
    height, width, _ = image.shape
    bottom_crop = image[int(height * 0.9):, :]  # Crop bottom 10% where timestamp is located

    # Convert to grayscale and apply OCR
    gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")

    # Find date and time in text
    date, time_text = "Unknown Date", "Unknown Time"
    words = text.split()
    
    for i in range(len(words)):
        if "/" in words[i]:  # Detect date format like 3/8/2024
            date = words[i]
        if ":" in words[i]:  # Detect time format like 11:47:53 AM
            time_text = words[i] + " " + words[i + 1] if i + 1 < len(words) else words[i]

    return date, time_text

def send_whatsapp_message(image_path, date, time_text):
    """Sends a WhatsApp message with the detected image and timestamp."""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=YOUR_WHATSAPP_NUMBER,
        body=f"🚨 Alert! A person was detected in the image.\n📅 Date: {date}\n⏰ Time: {time_text}\n🖼 Image: {image_path}"
    )
    
    print(f"WhatsApp Message Sent! SID: {message.sid}")

def detect_person(image_path, model, conf_threshold=0.25):
    """Runs YOLO model to detect a person and extracts timestamp."""
    results = model(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print("Error reading image:", image_path)
        return False, None, None, None

    person_detected = False

    if len(results) == 0 or len(results[0].boxes) == 0:
        return False, image, None, None

    for box in results[0].boxes:
        cls = box.cls.item()
        conf = box.conf.item()
        coords = box.xyxy[0].tolist()

        if int(cls) == 0 and conf >= conf_threshold:
            person_detected = True
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    date, time_text = extract_datetime_from_image(image)
    return person_detected, image, date, time_text

class NewImageHandler(FileSystemEventHandler):
    """Handles new image detection and processing."""
    def __init__(self, output_folder, model, conf_threshold=0.25):
        super().__init__()
        self.output_folder = output_folder
        self.model = model
        self.conf_threshold = conf_threshold

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                time.sleep(1)  # Delay to ensure full file write
                detected, annotated_image, date, time_text = detect_person(file_path, self.model, self.conf_threshold)

                if detected and annotated_image is not None:
                    output_path = os.path.join(self.output_folder, os.path.basename(file_path))
                    cv2.imwrite(output_path, annotated_image)
                    print(f"Person detected: {file_path}. Saved to {output_path}")

                    # Send WhatsApp Alert with extracted date/time
                    send_whatsapp_message(file_path, date, time_text)
                else:
                    print(f"No person detected in {file_path}.")

if __name__ == '__main__':
    folder_path = r"C:\Users\user\Desktop\JSK_project"
    
    output_folder = "person_detected_images"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    event_handler = NewImageHandler(output_folder, model, conf_threshold=0.25)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=True)
    observer.start()

    print("Monitoring started. Waiting for new images...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
