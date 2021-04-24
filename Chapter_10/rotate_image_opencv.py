import numpy as np
import cv2

# Read the image...
img = cv2.imread('.//images//Bird.jpg', cv2.IMREAD_COLOR)

# Perform the rotation around the center point
rows,cols,channels = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
dst = cv2.warpAffine(img,M,(cols,rows))

# Display the image
cv2.imshow('Image', dst)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()