import numpy as np
import cv2
import dlib

# Load the buil-in face dedector of Dlib 
detector = dlib.get_frontal_face_detector()

# Load the image
img = cv2.imread('.//images//Face.jpg', cv2.IMREAD_COLOR)
# Create a grayscale copy of the image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get the detected face bounding boxes, using the grayscale image
rects = detector(img_gray, 0)

# Loop over the bounding boxes, if there are more than one face
for rect in rects:
    # Get the OpenCV coordinates from the Dlib rectangle objects
    x = rect.left()
    y = rect.top()
    x1 = rect.right()
    y1 = rect.bottom()

    # Draw a rectangle around the face bounding box in OpenCV
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

# Display the resulting image
cv2.imshow('Detected Faces', img)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
