import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import threading

# --- Page Configuration ---
st.set_page_config(layout="centered")
st.title("Virtual Drag and Drop (ACV)")

st.markdown("""
### How to Use:
1.  **Click "Start"** below to activate your webcam.
2.  Allow browser permissions for the webcam.
3.  Ensure your hand is visible to the camera. The application will detect your hand and track your index finger.
4.  Move your index finger over the colored square to "grab" it and drag it around the screen.
5.  **Click "Stop"** to deactivate the webcam.
""")

# --- DraggableObject Class ---
class DraggableObject:
    def __init__(self, posCenter, size=(100, 100), color=(0, 215, 255)):
        self.posCenter = posCenter
        self.size = size
        self.color = color

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        # Check if cursor is within the object's bounds
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor  # Update object's center to cursor position

    def draw(self, img):
        cx, cy = self.posCenter
        w, h = self.size
        overlay = img.copy()
        # Draw a filled rectangle on the overlay
        cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, -1)
        # Blend the overlay with the original image for transparency
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

# --- WebRTC Video Transformer ---
# This class will handle the video processing for each frame.
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Using a lock to prevent race conditions on the draggable object
        self.lock = threading.Lock()
        self.draggable_object = DraggableObject([320, 240], size=(100, 100))
        
        # Initialize MediaPipe Hands
        self.mp_hands_module = mp.solutions.hands
        self.hands_solution = self.mp_hands_module.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands_solution.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands_module.HAND_CONNECTIONS)
                
                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[self.mp_hands_module.HandLandmark.INDEX_FINGER_TIP]
                h_img, w_img, _ = img.shape
                cursor_pos = (int(index_finger_tip.x * w_img), int(index_finger_tip.y * h_img))

                # Update object position (thread-safe)
                with self.lock:
                    self.draggable_object.update(cursor_pos)

        # Draw the object (thread-safe)
        with self.lock:
            self.draggable_object.draw(img)

        return img

# --- Streamlit UI ---
# RTCConfiguration is needed for deployment on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drag-and-drop",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("Application is running. Use the controls above to start/stop the webcam.")
