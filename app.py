import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
SHAPE_COLORS = {
    'rectangle': (255, 0, 0),    # Blue
    'circle': (0, 255, 0),       # Green
    'triangle': (0, 0, 255),     # Red
    'star': (255, 255, 0),       # Yellow
    'crown': (255, 0, 255)       # Magenta
}

class ShapeManager:
    def __init__(self):
        self.current_shape = None
        self.position = [320, 240]  # Default center position
        self.size = (100, 100)      # Default size
    
    def draw_shape(self, frame, shape_type, position=None):
        if position is not None:
            self.position = position

        x, y = self.position
        w, h = self.size
        color = SHAPE_COLORS.get(shape_type, (255, 0, 0))

        if shape_type == 'rectangle':
            cv2.rectangle(frame, 
                         (int(x - w/2), int(y - h/2)), 
                         (int(x + w/2), int(y + h/2)), 
                         color, -1)
        
        elif shape_type == 'circle':
            cv2.circle(frame, 
                      (int(x), int(y)), 
                      int(w/2), 
                      color, -1)
        
        elif shape_type == 'triangle':
            points = np.array([
                [x, y - h/2],
                [x - w/2, y + h/2],
                [x + w/2, y + h/2]
            ], np.int32)
            cv2.fillPoly(frame, [points], color)
        
        elif shape_type == 'star':
            points = np.array([
                [x, y - h/2],              # top
                [x + w/4, y - h/4],        # right upper
                [x + w/2, y],              # right
                [x + w/4, y + h/4],        # right lower
                [x, y + h/2],              # bottom
                [x - w/4, y + h/4],        # left lower
                [x - w/2, y],              # left
                [x - w/4, y - h/4]         # left upper
            ], np.int32)
            cv2.fillPoly(frame, [points], color)
        
        elif shape_type == 'crown':
            points = np.array([
                [x - w/2, y + h/2],        # bottom left
                [x + w/2, y + h/2],        # bottom right
                [x + w/2, y],              # middle right
                [x + w/4, y - h/2],        # top right
                [x, y],                    # middle top
                [x - w/4, y - h/2],        # top left
                [x - w/2, y]               # middle left
            ], np.int32)
            cv2.fillPoly(frame, [points], color)

class HandTrackerApp:
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.shape_manager = ShapeManager()
        
    def process_frame(self, frame, shape_type):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip position
                index_finger = hand_landmarks.landmark[8]
                h, w, c = frame.shape
                finger_x = int(index_finger.x * w)
                finger_y = int(index_finger.y * h)
                
                # Update shape position
                self.shape_manager.draw_shape(frame, shape_type, [finger_x, finger_y])
        else:
            # Draw shape at current position if no hand detected
            self.shape_manager.draw_shape(frame, shape_type)
        
        return frame

def main():
    st.title("Hand Tracking Shape Drawer")
    st.write("Move your hand to control shapes!")

    # Sidebar for controls
    st.sidebar.title("Controls")
    shape_type = st.sidebar.selectbox(
        "Select Shape",
        ['rectangle', 'circle', 'triangle', 'star', 'crown']
    )

    # Initialize HandTrackerApp
    tracker = HandTrackerApp()

    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Callback for video frame processing
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame with hand tracking and shape drawing
        processed_frame = tracker.process_frame(img, shape_type)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="hand-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Instructions
    st.markdown("""
    ### Instructions:
    1. Allow camera access when prompted
    2. Select a shape from the sidebar
    3. Move your index finger to control the shape
    4. The shape will follow your index finger tip
    
    ### Tips:
    - Keep your hand within camera view
    - Use good lighting for better tracking
    - Move slowly for better control
    """)

if __name__ == "__main__":
    main()