import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model
import traceback

# Load the pre-trained model
model = load_model('/home/vikas/Downloads/Sign-Language-To-Text-and-Speech-Conversion-master/cnn8grps_rad1_model.h5')

# Create or read a white image (ensure it's 3-channel, color image)
white = np.ones((400, 400, 3), np.uint8) * 255  # Create a white image with 3 color channels
cv2.imwrite("white.jpg", white)

# Initialize video capture from webcam
capture = cv2.VideoCapture(0)

# Check if camera is initialized
if not capture.isOpened():
    print("Error: Camera not initialized")
    exit()

# Initialize hand detectors for detecting hands in the video feed
hd = HandDetector(maxHands=1)  # Detector for main hand
hd2 = HandDetector(maxHands=1)  # Detector for secondary hand if needed

offset = 29
step = 1
flag = False
suv = 0

def distance(x, y):
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def distance_3d(x, y):
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2) + ((x[2] - y[2]) ** 2))

bfh = 0
dicttt = dict()
count = 0
kok = []

while True:
    try:
        # Capture frame from camera
        ret, frame = capture.read()
        
        # Check if frame is captured correctly
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame to avoid mirror image
        frame = cv2.flip(frame, 1)

        # Detect hands in the frame
        hands = hd.findHands(frame, draw=False, flipType=True)
        
        # Check if any hands were detected
        if hands:
            hand = hands[0]  # Use the first detected hand
            lm_list = hand['lmList']  # List of 21 landmarks on the hand

            # Draw landmarks on the hand (you can customize as needed)
            for lm in lm_list:
                cv2.circle(frame, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)
            
            # Extract bounding box around the hand
            x, y, w, h = hand['bbox']  # x, y are top-left corner, w and h are width and height of the bounding box

            # Resize and prepare the image for model prediction
            # Resize the hand region to fit the model's input size (assumed to be 400x400)
            hand_img = frame[y:y+h, x:x+w]
            hand_img_resized = cv2.resize(hand_img, (400, 400))

            # Convert to the expected shape for the model
            hand_img_resized = np.reshape(hand_img_resized, (1, 400, 400, 3))  # Model expects (1, 400, 400, 3) shape

            # Make prediction using the model
            prob = model.predict(hand_img_resized)[0]  # Get the prediction probabilities
            ch1, ch2, ch3 = np.argsort(prob)[-3:][::-1]  # Get top 3 predictions
            
            # Display the prediction on the frame
            cv2.putText(frame, f"Predicted: {ch1}, {ch2}, {ch3}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()

# Release the video capture and destroy OpenCV windows
capture.release()
cv2.destroyAllWindows()
