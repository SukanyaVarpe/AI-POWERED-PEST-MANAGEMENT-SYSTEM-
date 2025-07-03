# 🌿 AI Powered Pest Management System

An **AI-powered pest management system** using **YOLO object detection with Streamlit frontend** and **laser-based targeted pest elimination** for **precision farming**.

---

## 🎯 **Project Overview**

This system leverages **Ultralytics YOLOv8** to detect insects in crop images and **live camera feeds**, enabling **targeted pest elimination using a laser mounted on a camera-guided XY platform**.

It helps:
- Identify pests early to prevent damage.
- Eliminate pests using precise, automated laser targeting.
- Reduce pesticide use through targeted interventions.

---

## ✨ **Features**

✅ **Live Camera Feed Detection:** Real-time pest detection using YOLO via a connected camera.  
✅ **Laser-based Pest Elimination:** Automatically targets and eliminates detected pests using a laser with G-code and serial communication for precise XY movement.  
✅ **Image Upload Detection:** Upload images for pest detection if live camera is not used.  
✅ **About Page:** View training results, F1 curves, and validation images.  
✅ **Contact Page:** Collect user queries and feedback for system improvement.  
✅ **User-Friendly Interface:** Built with Streamlit for ease of use in labs and field conditions.

---

## 🖼️ **Demo Screenshots**

### Home Page
![Home](./static/home_screenshot.jpg)

### Detection Page
![Detection](./static/detection_screenshot.jpg)

### Live Camera Feed
![Live](./static/live_camera_screenshot.jpg)

---

## 🛠️ **Tech Stack**

- **Framework:** Streamlit
- **Detection:** Ultralytics YOLOv8
- **Hardware Integration:** OpenCV for camera, PySerial for laser XY platform control
- **Image Processing:** OpenCV, PIL
- **Language:** Python 3.x

---

## 🚀 **Getting Started**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-pest-management.git
cd ai-pest-management
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available:

bash
Copy
Edit
pip install streamlit ultralytics opencv-python pillow numpy pyserial
3️⃣ Run the Application
bash
Copy
Edit
streamlit run app.py
Open in your browser:

arduino
Copy
Edit
http://localhost:8501
📂 Project Structure
php
Copy
Edit
├── app.py                  # Main Streamlit app
├── yolov8n.pt              # YOLO weights (replace with your trained model if needed)
├── treat.json              # Optional pest metadata
├── static/                 # Images for UI and results
│   ├── l.jpg
│   ├── h.jpeg
│   ├── results.png
│   └── ...
└── requirements.txt        # Project dependencies
⚙️ How It Works
🚦 Detection Flow:
User uploads an image or starts live camera feed.

YOLO model detects pest locations.

If live detection, coordinates are converted to G-code.

Laser module moves to pest location and fires automatically using:

G-code (G00, G01, M3, M5)

Serial communication with Arduino/GRBL-based XY platform.

📡 Implemented Hardware Integration
✅ Live camera YOLO detection with OpenCV
✅ Laser kill operation with automated pest targeting
✅ Camera-guided XY platform movement using G-code and PySerial

This system has been tested on live insects for precise elimination while ensuring safety and effectiveness in controlled lab conditions.
