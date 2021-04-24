import numpy as np
import cv2

# Read the image...
# cv2.IMREAD_COLOR - load a color image, without transparency
# cv2.IMREAD_GRAYSCALE - load image in grayscale mode
# cv2.IMREAD_UNCHANGED - load image as-is, including transparency if it is there
img = cv2.imread('.//images//Bird.jpg', cv2.IMREAD_COLOR)

# Display the image
cv2.imshow('Image', img)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()