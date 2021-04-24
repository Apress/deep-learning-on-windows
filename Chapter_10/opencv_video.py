import numpy as np
import cv2

# Create the video capture object for a video file
cap = cv2.VideoCapture(".\\videos\\GH010055.mp4")

while(cap.isOpened()):
    # Read frame-by-frame
    ret, frame = cap.read()

    if (ret):
        # Resize the frame
        res = cv2.resize(frame, (960, 540), interpolation = cv2.INTER_CUBIC)
        
        # Display the resulting frame
        cv2.imshow('Video', res)
    
    # Press "q" to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()