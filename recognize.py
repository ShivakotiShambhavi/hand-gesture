import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Define finger tip landmarks (thumb tip is special)
FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
THUMB_TIP = 4
THUMB_IP = 3  # Used for thumb direction check

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Count fingers
            # For each finger tip, compare Y to PIP joint (tip should be above joint)
            for tip in FINGER_TIPS:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    finger_count += 1

            # Thumb logic: depends on x-coordinates (flipped for right hand)
            if landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x:
                finger_count += 1

    # Display result
    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Finger Counter (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
