import numpy as np
import cv2

# Create the video capture object for camera id '0'
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if (ret):
        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

    ch = 0xFF & cv2.waitKey(1)

    # Press "q" to quit the program
    if ch == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
