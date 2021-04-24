# Import the packages
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Loading the model from saved model file
model = load_model('data/lenet_model.h5')

# Visualizing the model
plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True, 
    show_layer_names=True,
    rankdir='TB', 
    expand_nested=False, 
    dpi=96
)

# Visualizing the model with no layer details
plot_model(
    model, 
    to_file='model_no_layer_details.png', 
    show_shapes=False, 
    show_layer_names=False,
    rankdir='TB', 
    expand_nested=False, 
    dpi=96
)

# Visualizing the model horizontally 
plot_model(
    model, 
    to_file='model_horizontal.png', 
    show_shapes=True, 
    show_layer_names=True,
    rankdir='LR', 
    expand_nested=False, 
    dpi=96
)