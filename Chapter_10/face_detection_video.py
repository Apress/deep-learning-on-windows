import numpy as np
import cv2
import dlib

# Create the video capture object for camera id '0'
video_capture = cv2.VideoCapture(0)
# Load the buil-in face dedector of Dlib 
detector = dlib.get_frontal_face_detector()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if (ret):
        # Create a grayscale copy of the captured frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the detected face bounding boxes, using the grayscale image
        rects = detector(gray, 0)

        # Loop over the bounding boxes, if there are more than one face
        for rect in rects:
            # Get the OpenCV coordinates from the Dlib rectangle objects
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            # Draw a rectangle around the face bounding box in OpenCV
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

    ch = 0xFF & cv2.waitKey(1)

    # press "q" to quit the program.
    if ch == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
