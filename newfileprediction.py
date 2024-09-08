import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model
import traceback
import os

# Load the pre-trained model
model = load_model('/home/vikas/Downloads/Sign-Language-To-Text-and-Speech-Conversion-master/cnn8grps_rad1_model.h5')

# Create a white image
white = np.ones((400, 400, 3), np.uint8) * 255  # Ensure it's 3-channel (RGB)
cv2.imwrite("white.jpg", white)

# Initialize video capture and hand detector
capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Offset for cropping the hand
offset = 29

# Mapping from predicted class index to letters (A-Z)
classes = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

# Functions to calculate distances
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def distance_3d(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2) + ((x[2] - y[2]) ** 2))

while True:
    try:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        hands, frame = detector.findHands(frame)  # Detect hands
        
        if hands:
            hand = hands[0]  # Get the first detected hand
            x, y, w, h = hand['bbox']
            cropped_image = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            if cropped_image.size == 0:
                continue  # Ensure the cropped image is valid

            # Load the white background and resize the hand image
            white = cv2.imread("white.jpg")
            resized_hand = cv2.resize(cropped_image, (400, 400))

            # Draw hand landmarks on the white background
            hand_landmarks = hand['lmList']
            for i in range(21):
                cv2.circle(resized_hand, (hand_landmarks[i][0], hand_landmarks[i][1]), 5, (0, 0, 255), -1)

            cv2.imshow("Hand Gesture", resized_hand)

            # Prepare the input for the model
            resized_hand = resized_hand / 255.0  # Normalize the image
            input_image = np.expand_dims(resized_hand, axis=0)  # Add batch dimension

            # Predict with the model
            prediction = model.predict(input_image)

            # Print the full prediction array to see probabilities for all classes
            print(f"Prediction probabilities: {prediction}")

            predicted_class = np.argmax(prediction)  # Get the class with highest probability

            # Map the predicted class to the corresponding letter
            predicted_letter = classes.get(predicted_class, 'Unknown')
            print(f"Predicted letter: {predicted_letter}")

            # Display the predicted letter on the frame
            cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Video Feed", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

# Release video capture and close windows
capture.release()
cv2.destroyAllWindows()