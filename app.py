import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Streamlit UI setup
st.title("Virtual Drag and Drop (ACV)")
st.sidebar.markdown("### Controls")
run_app = st.sidebar.checkbox("Run Application", value=False)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1,
    max_num_hands=2
)
mp_draw = mp.solutions.drawing_utils

# DraggableObject class
class DraggableObject:
    def __init__(self, posCenter, size=(100, 100), color=(0, 215, 255), label=""):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.label = label

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def draw(self, img):
        cx, cy = self.posCenter
        w, h = self.size
        overlay = img.copy()
        cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, -1)
        cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)

# Initialize draggable object
draggable_object = DraggableObject([640, 360], size=(200, 200))

# Streamlit video processing
if run_app:
    # Streamlit state to manage application run
    if "run_flag" not in st.session_state:
        st.session_state.run_flag = True

    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    try:
        while st.session_state.run_flag:
            ret, img = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                    index_finger_tip = handLms.landmark[8]
                    h, w, c = img.shape
                    cursor = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    draggable_object.update(cursor)

            draggable_object.draw(img)
            stframe.image(img, channels="BGR", use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
        hands.close()
        st.info("Webcam and MediaPipe resources have been released.")

st.sidebar.info("Check the box above to start the app.")
