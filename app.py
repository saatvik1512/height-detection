import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import Image
import tempfile
import os
import time

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Constants
KNOWN_DISTANCE = 60.96  # cm
KNOWN_WIDTH = 14.3  # cm
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STABILITY_THRESHOLD = 10  # pixels
STABILITY_TIME = 5  # seconds

class HeightDetectionApp:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.pose = mp_pose.Pose()
        self.stable_start_time = None
        self.last_height = None
        self.stable_height = None
        self.height_buffer = []
        
    def speak(self, text):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('rate', 150)
        engine.setProperty('voice', voices[0].id)
        engine.say(text)
        engine.runAndWait()

    def find_focal_length(self, measured_distance, real_width, width_in_rf_image):
        return (width_in_rf_image * measured_distance) / real_width

    def find_distance(self, focal_length, real_face_width, face_width_in_frame):
        return (real_face_width * focal_length) / face_width_in_frame

    def detect_face(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
        face_width = 0
        
        for (x, y, h, w) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
            face_width = w
            
        return face_width

    def check_stability(self, current_height):
        """Check if the height measurement is stable."""
        if current_height is None:
            return False, 0
        
        # Add current height to buffer
        self.height_buffer.append(current_height)
        if len(self.height_buffer) > 10:  # Keep last 10 measurements
            self.height_buffer.pop(0)
        
        # Check if measurements are stable
        if len(self.height_buffer) >= 5:
            height_variance = np.std(self.height_buffer)
            if height_variance < STABILITY_THRESHOLD:
                if self.stable_start_time is None:
                    self.stable_start_time = time.time()
                time_stable = time.time() - self.stable_start_time
                return True, time_stable
            else:
                self.stable_start_time = None
                return False, 0
        return False, 0

    def measure_height(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        height = None
        stability_message = ""
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            # Get head and foot points
            if landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.5 and \
               landmarks[mp_pose.PoseLandmark.NOSE.value].visibility > 0.5:
                
                foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
                head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * h
                pixel_height = abs(foot_y - head_y)
                
                # Convert pixel height to real height (you might need to adjust this formula)
                height = round((pixel_height * 0.5), 1)  # Simplified conversion
                
                # Check stability
                is_stable, time_stable = self.check_stability(height)
                
                if is_stable:
                    remaining_time = max(0, STABILITY_TIME - time_stable)
                    if remaining_time > 0:
                        stability_message = f"Stay still for {remaining_time:.1f} more seconds..."
                    else:
                        if self.stable_height is None:
                            self.stable_height = np.mean(self.height_buffer)
                            stability_message = f"Final Height: {self.stable_height:.1f} cm"
                else:
                    stability_message = "Please stand still..."
                    self.stable_height = None
                
        return height, image, stability_message

def main():
    st.title("Height Detection System")
    
    app = HeightDetectionApp()
    
    st.sidebar.title("Settings")
    detection_mode = st.sidebar.radio(
        "Select Detection Mode",
        ("Distance Detection", "Height Measurement")
    )
    
    # Initialize webcam
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    status_text = st.empty()  # Placeholder for status messages
    final_height_text = st.empty()  # Placeholder for final height
    camera = cv2.VideoCapture(0)
    
    if run:
        if detection_mode == "Distance Detection":
            st.write("Stand in front of the camera and maintain proper distance")
            
            while run:
                _, frame = camera.read()
                face_width = app.detect_face(frame)
                
                if face_width != 0:
                    distance = app.find_distance(
                        focal_length=1000,
                        real_face_width=KNOWN_WIDTH,
                        face_width_in_frame=face_width
                    )
                    
                    cv2.putText(
                        frame,
                        f"Distance: {round(distance,2)} cm",
                        (30, 35),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        GREEN,
                        2
                    )
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                
        else:  # Height Measurement
            st.write("Stand straight and ensure your full body is visible")
            
            while run:
                _, frame = camera.read()
                height, processed_frame, stability_message = app.measure_height(frame)
                
                if height:
                    cv2.putText(
                        processed_frame,
                        f"Current Height: {height} cm",
                        (30, 35),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        GREEN,
                        2
                    )
                    
                    # Display stability message on frame
                    cv2.putText(
                        processed_frame,
                        stability_message,
                        (30, 70),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        RED if "still" in stability_message else GREEN,
                        2
                    )
                    
                    # Update status in Streamlit
                    status_text.text(stability_message)
                    
                    # Display final height if measurement is complete
                    if app.stable_height is not None:
                        final_height_text.markdown(f"""
                        ### Final Measurement
                        **Height:** {app.stable_height:.1f} cm
                        
                        *Stand still for new measurement*
                        """)
                
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(processed_frame)
    
    else:
        st.write("Click 'Start Camera' to begin")
        camera.release()

if __name__ == '__main__':
    main()