import numpy as np
import cv2

# Read the image...
img = cv2.imread('.//images//Bird.jpg', cv2.IMREAD_COLOR)

# Extract the region-of-interest from the image
img_roi = img[50:250, 150:300]

# Save the region-of-interest as an image
cv2.imwrite('.//images//Bird_ROI.jpg', img_roi)

# Display the extracted region-of-interest 
cv2.imshow('Image ROI', img_roi)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()