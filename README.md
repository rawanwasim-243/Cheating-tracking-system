# 🎓 Student Cheating Tracking and Alert System

A real-time Desktop Computer Vision application built with **Python**, **YOLOv8**, and **CustomTkinter**. The system tracks students during exams, identifies cheating behaviors (such as using mobile phones or looking at a classmate's paper), dynamically changes bounding box colors, captures evidence screenshots, and logs incidents in a local database.

---

## 📌 Features

- **Real-Time Detection & Tracking**:
  - Detects **cell phone usage** using YOLOv8 object detection.
  - Detects **looking/turning head toward neighbors** using YOLOv8-Pose (facial keypoints analysis).
- **Dynamic Visual Feedback**:
  - 🟩 **Green Bounding Box**: Normal student behavior.
  - 🟥 **Red Bounding Box**: Suspected cheating incident detected.
- **Automated Evidence Logging**:
  - Automatically captures high-resolution screenshots upon detecting suspicious behavior.
  - Saves evidence images locally in `alerts/screenshots/`.
- **Database Integration**:
  - Logs incident timestamp, behavior type, and confidence score using SQLite database (`cheating_alerts.db`).
- **Interactive GUI (Desktop App)**:
  - Built using **CustomTkinter** for a modern user experience.
  - Includes Video Uploading, Real-time Visualizer, Recent Alerts History, and Direct Folder Access.

---

## 🛠️ Tech Stack & Requirements

* **Programming Language**: Python 3.8+
* **Deep Learning Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt` & `yolov8n-pose.pt`)
* **Computer Vision**: OpenCV (`opencv-python`)
* **GUI Framework**: CustomTkinter & Pillow (PIL)
* **Database**: SQLite3

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone or download this repository, then install the required Python packages via terminal:

```bash
pip install ultralytics opencv-python pillow customtkinter
