import cv2  # OpenCV library for video capture and processing
import mediapipe as mp  # MediaPipe library for hand tracking and landmarks detection

# Initialize MediaPipe hands module for hand tracking
mp_hands = mp.solutions.hands  # MediaPipe hands module
mp_draw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks on frames
# Configure the hands module with minimum detection and tracking confidence
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video from the webcam (camera index 0)
cap = cv2.VideoCapture(0)

# List of indices corresponding to the tips of each finger in MediaPipe's hand landmarks
# 4: Thumb, 8: Index Finger, 12: Middle Finger, 16: Ring Finger, 20: Pinky
finger_tips = [4, 8, 12, 16, 20]

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is captured

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR (OpenCV format) to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks using MediaPipe
    result = hands.process(rgb_frame)

    total_fingers = 0  # Initialize a counter for the total number of fingers raised

    # Check if hands are detected in the frame
    if result.multi_hand_landmarks:
        # Loop through each detected hand and its associated metadata
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Draw the hand landmarks and connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if the hand is "Left" or "Right" based on classification
            hand_label = hand_info.classification[0].label  # Either 'Left' or 'Right'

            # List to keep track of whether each finger is up (1 for up, 0 for down)
            fingers_up = []

            # Get the coordinates of all landmarks for the current hand
            landmarks = hand_landmarks.landmark

            # Thumb detection logic:
            # Check if the x-coordinate of the thumb tip is further away from the palm
            # Adjust the condition based on whether it's a left or right hand
            if hand_label == 'Left':
                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                    fingers_up.append(1)  # Thumb is up
                else:
                    fingers_up.append(0)  # Thumb is down
            else:  # Right hand
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                    fingers_up.append(1)  # Thumb is up
                else:
                    fingers_up.append(0)  # Thumb is down

            # Logic for other fingers (Index, Middle, Ring, Pinky)
            # Compare the y-coordinate of the fingertip with the landmark two levels below it
            # If the fingertip is higher (smaller y-coordinate), the finger is up
            for i in range(1, 5):
                if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
                    fingers_up.append(1)  # Finger is up
                else:
                    fingers_up.append(0)  # Finger is down

            # Add the count of fingers up for this hand to the total count
            total_fingers += sum(fingers_up)

    # Overlay the total number of fingers detected on the video frame
    cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the processed video frame
    cv2.imshow("Finger Counting", frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
