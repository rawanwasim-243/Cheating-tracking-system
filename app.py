import os
import cv2
import sqlite3
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Database Setup (SQLite)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("cheating_alerts.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            behavior TEXT,
            confidence REAL,
            screenshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_alert_to_db(behavior, confidence, screenshot_path):
    conn = sqlite3.connect("cheating_alerts.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO alerts (timestamp, behavior, confidence, screenshot_path)
        VALUES (?, ?, ?, ?)
    ''', (now, behavior, confidence, screenshot_path))
    conn.commit()
    conn.close()
    return now

# ---------------------------------------------------------
# 2. GUI Application using CustomTkinter
# ---------------------------------------------------------
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class CheatingDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cheating Tracking & Alert System")
        self.geometry("1100x700")

        # Load models: Standard YOLOv8 for Objects (Phone) & YOLOv8-Pose for Head Movement
        self.detector_model = YOLO("yolov8n.pt")       # Cell Phone detection
        self.pose_model = YOLO("yolov8n-pose.pt")      # Head & Pose detection

        self.is_processing = False
        self.cap = None

        # Create screenshot directory
        self.output_dir = os.path.join("alerts", "screenshots")
        os.makedirs(self.output_dir, exist_ok=True)

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Frame: Video Display
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Upload a video to start processing", font=("Arial", 16))
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Right Frame: Controls & Alerts
        self.sidebar_frame = ctk.CTkFrame(self, width=300)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="Upload Video", command=self.upload_video)
        self.btn_upload.pack(padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="Stop Processing", fg_color="red", hover_color="darkred", command=self.stop_video)
        self.btn_stop.pack(padx=20, pady=5)

        self.btn_gallery = ctk.CTkButton(self.sidebar_frame, text="Open Screenshots Folder", command=self.open_screenshots)
        self.btn_gallery.pack(padx=20, pady=10)

        self.alerts_label = ctk.CTkLabel(self.sidebar_frame, text="Recent Alerts History:", font=ctk.CTkFont(size=14, weight="bold"))
        self.alerts_label.pack(padx=20, pady=(20, 5), anchor="w")

        self.history_box = ctk.CTkScrollableFrame(self.sidebar_frame, height=300)
        self.history_box.pack(padx=10, pady=5, fill="both", expand=True)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            self.is_processing = True
            self.process_video()

    def stop_video(self):
        self.is_processing = False
        if self.cap:
            self.cap.release()

    def open_screenshots(self):
        os.system(f'explorer "{os.path.abspath(self.output_dir)}"')

    def add_notification_card(self, behavior, conf, timestamp):
        card = ctk.CTkFrame(self.history_box, fg_color="#3a3a3a")
        card.pack(fill="x", pady=5, padx=5)

        title = ctk.CTkLabel(card, text=f"⚠️ {behavior}", text_color="#ff4d4d", font=ctk.CTkFont(weight="bold"))
        title.pack(anchor="w", padx=5, pady=2)

        info = ctk.CTkLabel(card, text=f"Time: {timestamp}\nConf: {conf:.2f}", justify="left", font=ctk.CTkFont(size=11))
        info.pack(anchor="w", padx=5, pady=2)

    def detect_looking_side(self, keypoints):
        """ Sensitive detection for looking left or right """
        if keypoints is None or len(keypoints) < 5:
            return False
        
        nose = keypoints[0]       # Point 0: Nose
        left_eye = keypoints[1]   # Point 1: Left Eye
        right_eye = keypoints[2]  # Point 2: Right Eye
        left_ear = keypoints[3]   # Point 3: Left Ear
        right_ear = keypoints[4]  # Point 4: Right Ear

        # 1. Check if one of the eyes/ears is completely hidden (Head turned away)
        if (left_eye[2] < 0.35 and right_eye[2] > 0.5) or (right_eye[2] < 0.35 and left_eye[2] > 0.5):
            return True
            
        if (left_ear[2] < 0.25 and right_ear[2] > 0.5) or (right_ear[2] < 0.25 and left_ear[2] > 0.5):
            return True

        # 2. Check asymmetry between nose and eyes
        if nose[2] > 0.4 and left_eye[2] > 0.4 and right_eye[2] > 0.4:
            dist_l = abs(nose[0] - left_eye[0])
            dist_r = abs(nose[0] - right_eye[0])

            if dist_r > 0:
                ratio = dist_l / dist_r
                # If head turned slightly side (ratio skewed)
                if ratio > 2.0 or ratio < 0.5:
                    return True

        return False

    def process_video(self):
        if not self.is_processing or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_processing = False
            messagebox.showinfo("Finished", "Video processing finished.")
            return

        frame_cheating_detected = False
        latest_behavior = ""
        latest_conf = 0.0

        # 1. Detect Cell Phones
        obj_results = self.detector_model(frame, verbose=False)
        phone_boxes = []
        if obj_results and len(obj_results) > 0:
            for box in obj_results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == 67: # Cell Phone
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    phone_boxes.append((px1, py1, px2, py2))
                    frame_cheating_detected = True
                    latest_behavior = "Cheating: Phone Detected"
                    latest_conf = float(box.conf[0])

        # 2. Detect Head Pose & Turning Side
        pose_results = self.pose_model(frame, verbose=False)

        if pose_results and len(pose_results) > 0:
            for pose in pose_results[0]:
                boxes = pose.boxes.xyxy.cpu().numpy()
                kpts = pose.keypoints.data.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    person_kpts = kpts[i]

                    # Detect if student is looking away to neighbor
                    is_looking_away = self.detect_looking_side(person_kpts)

                    # Check if student is near any phone box
                    has_phone = False
                    for (px1, py1, px2, py2) in phone_boxes:
                        if px1 >= x1 and px2 <= x2 and py1 >= y1 and py2 <= y2:
                            has_phone = True
                            break

                    # Color Logic: Red for Cheater, Green for Normal
                    if is_looking_away or has_phone:
                        color = (0, 0, 255) # RED
                        behavior = "Cheating: Phone" if has_phone else "Cheating: Looking at Neighbor"
                        label = f"ALERT: {behavior}"
                        
                        frame_cheating_detected = True
                        latest_behavior = behavior
                        latest_conf = 0.88
                    else:
                        color = (0, 255, 0) # GREEN
                        label = "Student Normal"

                    # Draw Student Bounding Box & Label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Phone Boxes in Red
        for (px1, py1, px2, py2) in phone_boxes:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Trigger Save & Alert when Cheating Occurs
        if frame_cheating_detected:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            screenshot_path = os.path.join(self.output_dir, f"cheating_{timestamp_str}.jpg")
            cv2.imwrite(screenshot_path, frame)

            time_recorded = save_alert_to_db(latest_behavior, latest_conf, screenshot_path)
            self.add_notification_card(latest_behavior, latest_conf, time_recorded)

        # Update Image on GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((700, 480))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")

        self.after(10, self.process_video)

if __name__ == "__main__":
    init_db()
    app = CheatingDetectionApp()
    app.mainloop()