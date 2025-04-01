Real-Time Human Detection in Forest Range
📌 Overview

This project is designed to detect human presence in forest areas to aid in wildlife conservation efforts. Using YOLO (You Only Look Once) object detection, the system processes images and identifies people within the forest range. Upon detection, an automated WhatsApp alert is sent via Twilio API to notify authorities or conservation teams.
🚀 Features

✅ Real-time human detection using YOLO deep learning model
✅ Automated alert system via Twilio WhatsApp API
✅ OCR-based timestamp extraction from images
✅ Automatic image monitoring using Watchdog
✅ Processed image storage for further analysis
📂 Project Structure

📦 JSK_project
 ┣  person_detected_images   # Folder to store images with detected persons
 ┣  detect_person_pipeline.py  # Main script for detection & alerting
 ┣  requirements.txt  # Python dependencies
 ┣  README.md  # Project documentation
 ┗ .gitignore  # Files to be ignored in Git

 📊 How It Works

1️⃣ Monitors a folder for new images
2️⃣ Uses YOLO to detect humans
3️⃣ Extracts date & time using OCR
4️⃣ Sends a WhatsApp alert with the detected image
5️⃣ Stores annotated images in the person_detected_images folder
