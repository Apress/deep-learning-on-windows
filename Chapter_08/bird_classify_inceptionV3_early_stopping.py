import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import math
import os
import os.path
import time

# utility functions
def graph_training_history(history):
    plt.rcParams["figure.figsize"] = (12, 9)

    plt.style.use('ggplot')

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()

    plt.show()

# util function to calculate the class weights based on the number of samples on each class
# this is useful with datasets that are higly skewed (datasets where 
# the number of samples in each class differs vastly)
def get_class_weights(class_data_dir):
    labels_count = dict()
    for img_class in [ic for ic in os.listdir(class_data_dir) if ic[0] != '.']:
        labels_count[img_class] = len(os.listdir(os.path.join(class_data_dir, img_class)))
    total_count = sum(labels_count.values())
    class_weights = {cls: total_count / count for cls, count in 
                    enumerate(labels_count.values())}
    return class_weights

# start time of the script
start_time = time.time()

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# number of epochs to train
epochs = 50

# batch size used by flow_from_directory
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# the number of classes/categories
num_classes = len(train_generator.class_indices)

# calculate the training steps
nb_train_samples = len(train_generator.filenames)
train_steps = int(math.ceil(nb_train_samples / batch_size))

# calculate the validation steps
nb_validation_samples = len(validation_generator.filenames)
validation_steps = int(math.ceil(nb_validation_samples / batch_size))

# get the class weights
class_weights = get_class_weights(train_data_dir)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_width, img_height, 3)))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

training_checkpoint_dir = 'checkpoints/training'

filepath = training_checkpoint_dir + "/model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(
                            filepath, 
                            monitor="val_acc", 
                            verbose=1, 
                            save_best_only=True, 
                            save_weights_only=False, 
                            mode="max"
                            )

early_stop = EarlyStopping(
                            monitor="val_acc", 
                            mode="max", 
                            verbose=1, 
                            patience=3, 
                            restore_best_weights=True
                            )

callbacks_list = [checkpoint, early_stop]

history = model.fit(
                    train_generator,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    class_weight=class_weights,
                    max_queue_size=15,
                    workers=8,
                    callbacks=callbacks_list
                    )

model.save('bird_classify_fine-tune_IV3_S1.h5')

(eval_loss, eval_accuracy) = model.evaluate(
    validation_generator, steps=validation_steps)

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# Run Fine-tuning on our model

# number of epochs to fine-tune
ft_epochs = 25

# reset our data generators
train_generator.reset()
validation_generator.reset()

# we chose to train the last convolution block from the base model
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), 
    loss='categorical_crossentropy', 
    metrics=['acc']
    )

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=ft_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    max_queue_size=15,
    workers=8
    )

model.save('bird_classify_finetune_IV3_final.h5')

(eval_loss, eval_accuracy) = model.evaluate(
    validation_generator, steps=validation_steps)

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

end_time = time.time()

training_duration = end_time - start_time
print("[INFO] Total Time for training: {} seconds".format(training_duration))
