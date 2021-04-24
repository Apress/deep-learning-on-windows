import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the ResNet50 model with the ImageNet weights
model = ResNet50(weights='imagenet')
# Create the video capture object
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if (ret):
        # Convert image from BGR to RGB
        rgb_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Resize the image to 224x224, the size required by ResNet50 model
        res_im = cv2.resize(rgb_im, (224, 224), interpolation = cv2.INTER_CUBIC)
        
        # Preprocess image
        prep_im = image.img_to_array(res_im)
        prep_im = np.expand_dims(prep_im, axis=0)
        prep_im = preprocess_input(prep_im)

        # Make the prediction
        preds = model.predict(prep_im)

        # Decode the prediction
        (class_name, class_description, score) = decode_predictions(preds, top=1)[0][0]

        # Display the predicted class and confidence
        print("Predicted: {0}, Confidence: {1:.2f}".format(class_description, score))
        cv2.putText(frame, "Predicted: {}".format(class_description), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Confidence: {0:.2f}".format(score), (10, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

    ch = 0xFF & cv2.waitKey(1)

    # press "q" to quit the program.
    if ch == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
