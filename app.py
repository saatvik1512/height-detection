import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
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
        if current_height is None:
            return False, 0
        
        self.height_buffer.append(current_height)
        if len(self.height_buffer) > 10:
            self.height_buffer.pop(0)
        
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
            
            if landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.5 and \
               landmarks[mp_pose.PoseLandmark.NOSE.value].visibility > 0.5:
                
                foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
                head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * h
                pixel_height = abs(foot_y - head_y)
                
                height = round((pixel_height * 0.5), 1)
                
                is_stable, time_stable = self.check_stability(height)
                
                if is_stable:
                    remaining_time = max(0, STABILITY_TIME - time_stable)
                    if remaining_time > 0:
                        stability_message = f"Stay still for {remaining_time:.1f}s..."
                    else:
                        if self.stable_height is None:
                            self.stable_height = np.mean(self.height_buffer)
                            stability_message = f"Final Height: {self.stable_height:.1f} cm"
                else:
                    stability_message = "Please stand still..."
                    self.stable_height = None
                
        return height, image, stability_message

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.app = HeightDetectionApp()
        self.detection_mode = "Height Measurement"  # Default mode
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detection_mode == "Distance Detection":
            face_width = self.app.detect_face(img)
            if face_width != 0:
                distance = self.app.find_distance(
                    focal_length=1000,
                    real_face_width=KNOWN_WIDTH,
                    face_width_in_frame=face_width
                )
                cv2.putText(img, f"Distance: {round(distance,2)} cm",
                           (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, GREEN, 2)
        else:
            height, processed_img, stability_message = self.app.measure_height(img)
            if height:
                cv2.putText(processed_img, f"Current Height: {height} cm",
                           (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, GREEN, 2)
                cv2.putText(processed_img, stability_message,
                           (30, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                           RED if "still" in stability_message else GREEN, 2)
                img = processed_img
                
                if self.app.stable_height is not None:
                    st.session_state.final_height = self.app.stable_height
                    
        return img

def main():
    st.title("Height Detection System")
    
    detection_mode = st.sidebar.radio(
        "Select Detection Mode",
        ("Distance Detection", "Height Measurement")
    )
    
    if 'final_height' not in st.session_state:
        st.session_state.final_height = None

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if ctx.video_processor:
        ctx.video_processor.detection_mode = detection_mode

    if detection_mode == "Distance Detection":
        st.write("Stand in front of the camera and maintain proper distance")
    else:
        st.write("Stand straight and ensure your full body is visible")
        if st.session_state.final_height:
            st.markdown(f"""
            ### Final Measurement
            **Height:** {st.session_state.final_height:.1f} cm
            *Stand still for new measurement*
            """)

if __name__ == '__main__':
    main()
