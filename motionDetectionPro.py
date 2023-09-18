import cv2

# Open the camera (camera index 1)
cam = cv2.VideoCapture(1)

while cam.isOpened():
    
    # Read two consecutive frames from the camera
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Calculate the absolute difference between the two frames
    abs_diff = cv2.absdiff(frame1, frame2)

    # Convert the absolute difference image to grayscale
    gray = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to the grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to create a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Perform dilation to enhance object boundaries
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image (detect objects)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the detected contours
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw rectangles around detected objects
        cv2.rectangle(frame1, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)

    # Display the processed frame
    cv2.imshow('Sec Cam', frame1)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(10) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
