import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Initialize session state
if "run_app" not in st.session_state:
    st.session_state.run_app = False
if "initializing" not in st.session_state: # New state for loading
    st.session_state.initializing = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "hands_solution" not in st.session_state:
    st.session_state.hands_solution = None

# --- UI Elements ---
st.set_page_config(layout="centered") # Center the main content area
st.title("Virtual Drag and Drop (ACV)")

st.markdown("""
### How to Use:
1.  **Click "Start Application"** below to activate your webcam.
2.  Ensure your hand is visible to the camera. The application will detect your hand and track your index finger.
3.  Move your index finger over the colored square to "grab" it and drag it around the screen.
4.  **Click "Stop Application"** to deactivate the webcam and end the session.
""")

# --- DraggableObject Class (unchanged from original logic) ---
class DraggableObject:
    def __init__(self, posCenter, size=(100, 100), color=(0, 215, 255), label=""):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.label = label

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        # Check if cursor is within the object's bounds
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor # Update object's center to cursor position

    def draw(self, img):
        cx, cy = self.posCenter
        w, h = self.size
        overlay = img.copy()
        # Draw a filled rectangle on the overlay
        cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, -1)
        # Blend the overlay with the original image for transparency
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

# Initialize draggable object (once)
draggable_object = DraggableObject([320, 240], size=(100, 100)) # Adjusted initial position

# --- MediaPipe Setup (drawing utility can be global) ---
mp_draw = mp.solutions.drawing_utils
mp_hands_module = mp.solutions.hands # Alias for clarity

# --- Start/Stop Button Logic ---
# Use columns to center the button
col1, col2, col3 = st.columns([1, 1.2, 1]) # Adjust ratios for better centering

with col2:
    if not st.session_state.run_app and not st.session_state.initializing:
        if st.button("Start Application", key="start_app_button", use_container_width=True):
            st.session_state.initializing = True
            st.rerun() # Rerun to show spinner

    elif st.session_state.initializing:
        with st.spinner("Initializing webcam and hand tracking... This may take a few moments, please wait."):
            try:
                st.session_state.hands_solution = mp_hands_module.Hands(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    model_complexity=1,
                    max_num_hands=1
                )
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    st.error("Failed to open webcam. Please check if it's connected and not used by another app.")
                    st.session_state.run_app = False
                    st.session_state.initializing = False # Reset initializing state
                    # Clean up partially initialized resources if any
                    if st.session_state.hands_solution:
                        st.session_state.hands_solution.close()
                        st.session_state.hands_solution = None
                else:
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    st.session_state.run_app = True # Successfully initialized
            except Exception as e:
                st.error(f"An error occurred during initialization: {e}")
                st.session_state.run_app = False
                # Ensure cleanup
                if st.session_state.cap and st.session_state.cap.isOpened(): st.session_state.cap.release(); st.session_state.cap = None
                if st.session_state.hands_solution: st.session_state.hands_solution.close(); st.session_state.hands_solution = None
            finally:
                st.session_state.initializing = False # Done initializing (success or fail)
                st.rerun() # Rerun to reflect new state (either running or error)

    elif st.session_state.run_app: # App is running
        if st.button("Stop Application", key="stop_app_button", type="primary", use_container_width=True):
            st.session_state.run_app = False
            st.session_state.initializing = False # Ensure this is reset
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.hands_solution is not None:
                st.session_state.hands_solution.close()
                st.session_state.hands_solution = None
            st.rerun()

# --- Video Processing Section ---
if st.session_state.run_app and not st.session_state.initializing: # Only process if running and not initializing
    if st.session_state.cap is not None and st.session_state.cap.isOpened() and st.session_state.hands_solution is not None:
        stframe = st.empty()
        while st.session_state.run_app:
            ret, img = st.session_state.cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam. Stopping application.")
                st.session_state.run_app = False
                if st.session_state.cap: st.session_state.cap.release(); st.session_state.cap = None
                if st.session_state.hands_solution: st.session_state.hands_solution.close(); st.session_state.hands_solution = None
                st.rerun()
                break
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = st.session_state.hands_solution.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp_hands_module.HandLandmark.INDEX_FINGER_TIP]
                    h_img, w_img, _ = img.shape
                    cursor_pos = (int(index_finger_tip.x * w_img), int(index_finger_tip.y * h_img))
                    draggable_object.update(cursor_pos)
            draggable_object.draw(img)
            stframe.image(img, channels="BGR", use_container_width=True)
        
        # Cleanup if loop exited due to run_app becoming False (e.g. Stop button)
        if not st.session_state.run_app:
             if st.session_state.cap and st.session_state.cap.isOpened(): st.session_state.cap.release(); st.session_state.cap = None
             if st.session_state.hands_solution: st.session_state.hands_solution.close(); st.session_state.hands_solution = None
             # No rerun here if already handled by stop button logic

    elif st.session_state.run_app: # Should not be hit if initializing is false and cap/hands are None
        st.error("Application is set to run, but resources are not available. Please try stopping and starting again.")

elif not st.session_state.run_app and not st.session_state.initializing:
    st.info("Application is currently stopped. Click 'Start Application' to begin.")
