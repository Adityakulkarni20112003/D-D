import cv2
import mediapipe as mp
import numpy as np

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1,  # Added model complexity for optimization
    max_num_hands=2  # Limit number of hands detected
)
mp_draw = mp.solutions.drawing_utils

# OpenCV video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

color_yellow = (0, 215, 255)  # Yellow color for the square

# DraggableObject class for shapes
class DraggableObject:
    def __init__(self, posCenter, size=(100, 100), color=color_yellow, shape='rectangle', label=""):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.shape = shape
        self.label = label
        self.is_visible = True

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def draw(self, img):
        if not self.is_visible:
            return

        cx, cy = self.posCenter
        w, h = self.size

        # Make the square transparent (with the yellow color as you requested)
        overlay = img.copy()
        cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, -1)
        cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)  # Adjust transparency

# Initialize draggable objects with the square shape
draggable_objects = [
    DraggableObject([640, 360], size=(200, 200), color=color_yellow, shape='rectangle', label="Square")
]

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global menu_open, selected_object_index
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, obj in enumerate(draggable_objects):
            cx, cy = obj.posCenter
            w, h = obj.size
            if cx - w // 2 < x < cx + w // 2 and cy - h // 2 < y < cy + h // 2:
                obj.is_visible = not obj.is_visible

# OpenCV setup for normal window mode (resizable and with minimize/close options)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

# Try-except block for graceful exit and frame handling
try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        img = cv2.flip(img, 1)

        # Hand detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = handLms.landmark[8]
                h, w, c = img.shape
                cursor = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                # Update the position of the draggable object (square)
                draggable_objects[0].update(cursor)

        # Draw visible objects (only the square here)
        for obj in draggable_objects:
            obj.draw(img)

        # Show the frame
        cv2.imshow("Hand Tracking", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting gracefully.")

finally:
    cap.release()
    cv2.destroyAllWindows()
