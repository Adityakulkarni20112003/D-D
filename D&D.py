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

color_blue = (255, 0, 0)  # Set a consistent color for all objects
color_yellow = (0, 215, 255)  # Original color for the crown

# DraggableObject class for shapes
class DraggableObject:
    def __init__(self, posCenter, size=(100, 100), color=color_blue, shape='rectangle', label=""):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.shape = shape
        self.label = label
        self.is_visible = False

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

        if self.shape == 'rectangle':
            cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, -1)
        elif self.shape == 'circle':
            cv2.circle(img, (cx, cy), w // 2, self.color, -1)
        elif self.shape == 'crown':
            num_spikes = 5
            spike_width = w // num_spikes
            spike_height = h // 2
            for i in range(num_spikes):
                spike_center_x = cx - w // 2 + spike_width * i + spike_width // 2
                spike_top = (spike_center_x, cy - h // 2 - spike_height)
                spike_base_left = (spike_center_x - spike_width // 2, cy - h // 2)
                spike_base_right = (spike_center_x + spike_width // 2, cy - h // 2)
                cv2.drawContours(img, [np.array([spike_top, spike_base_left, spike_base_right])], -1, self.color, -1)
        elif self.shape == 'triangle':
            points = np.array([
                [cx, cy - h // 2],
                [cx - w // 2, cy + h // 2],
                [cx + w // 2, cy + h // 2]
            ])
            cv2.drawContours(img, [points], 0, self.color, -1)
        elif self.shape == 'hexagon':
            points = np.array([
                [cx, cy - h // 2],
                [cx + w // 2, cy - h // 4],
                [cx + w // 2, cy + h // 4],
                [cx, cy + h // 2],
                [cx - w // 2, cy + h // 4],
                [cx - w // 2, cy - h // 4]
            ])
            cv2.drawContours(img, [points], 0, self.color, -1)
        elif self.shape == 'star':
            points = np.array([
                [cx, cy - h // 2],
                [cx + w // 4, cy - h // 4],
                [cx + w // 2, cy - h // 4],
                [cx + w // 6, cy],
                [cx + w // 4, cy + h // 2],
                [cx, cy + h // 4],
                [cx - w // 4, cy + h // 2],
                [cx - w // 6, cy],
                [cx - w // 2, cy - h // 4],
                [cx - w // 4, cy - h // 4]
            ])
            cv2.drawContours(img, [points], 0, self.color, -1)
        elif self.shape == 'ellipse':
            cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, self.color, -1)

# Initialize draggable objects with the same color for all shapes except Crown
draggable_objects = [
    DraggableObject([640, 360], size=(200, 100), color=color_yellow, shape='crown', label="Crown"),
    DraggableObject([640, 360], size=(150, 150), color=color_yellow, shape='circle', label="Circle"),
    DraggableObject([640, 360], size=(200, 100), color=color_yellow, shape='rectangle', label="Rectangle"),
    DraggableObject([640, 360], size=(100, 100), color=color_yellow, shape='triangle', label="Triangle"),
    DraggableObject([640, 360], size=(150, 150), color=color_yellow, shape='hexagon', label="Hexagon"),
    DraggableObject([640, 360], size=(120, 120), color=color_yellow, shape='star', label="Star"),
    DraggableObject([640, 360], size=(150, 100), color=color_yellow, shape='ellipse', label="Ellipse")
]

menu_open = False
selected_object_index = None

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global menu_open, selected_object_index
    button_x, button_y, button_w, button_h = 20, 20, 100, 50

    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            menu_open = not menu_open
        if menu_open:
            for i, obj in enumerate(draggable_objects):
                item_y = button_y + (i + 1) * (button_h + 10)
                if button_x <= x <= button_x + button_w and item_y <= y <= item_y + button_h:
                    selected_object_index = i
                    for obj in draggable_objects:
                        obj.is_visible = False
                    draggable_objects[selected_object_index].is_visible = True
                    menu_open = False

cv2.namedWindow("Interactive Menu")
cv2.setMouseCallback("Interactive Menu", mouse_click)

# Try-except block for graceful exit and frame handling
try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        img = cv2.flip(img, 1)

        # Draw menu button
        button_x, button_y, button_w, button_h = 20, 20, 100, 50
        button_color = (200, 200, 200) if not menu_open else (150, 150, 150)
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)
        cv2.putText(img, "Menu", (button_x + 20, button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw objects in the menu if open
        if menu_open:
            for i, obj in enumerate(draggable_objects):
                item_y = button_y + (i + 1) * (button_h + 10)
                cv2.rectangle(img, (button_x, item_y), (button_x + button_w, item_y + button_h), obj.color, -1)
                cv2.putText(img, obj.label, (button_x + 10, item_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Hand detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = handLms.landmark[8]
                h, w, c = img.shape
                cursor = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                if selected_object_index is not None:
                    draggable_objects[selected_object_index].update(cursor)

        # Draw visible objects
        for obj in draggable_objects:
            obj.draw(img)

        # Show the frame
        cv2.imshow("Interactive Menu", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting gracefully.")

finally:
    cap.release()
    cv2.destroyAllWindows()
