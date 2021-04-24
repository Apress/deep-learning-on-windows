import numpy as np
import cv2
from PIL import Image

# Read the image...
pil_image = Image.open('.//images//Bird.jpg')

# Convert image from RGB to BGR
opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow('Image', opencv_image)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()