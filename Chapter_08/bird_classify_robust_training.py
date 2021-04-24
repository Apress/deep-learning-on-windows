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
def graph_training_history(history, save_fig=False, save_path=None):
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

    if save_fig:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    # clear and close the current figure
    plt.clf()
    plt.close()

# util function to calculate the class weights based on the number of samples on each class
# this is useful with datasets that are highly skewed (datasets where 
# the number of samples in each class differs vastly)
def get_class_weights(class_data_dir):
    labels_count = dict()
    for img_class in [ic for ic in os.listdir(class_data_dir) if ic[0] != '.']:
        labels_count[img_class] = len(os.listdir(os.path.join(class_data_dir, img_class)))
    total_count = sum(labels_count.values())
    class_weights = {cls: total_count / count for cls, count in 
                    enumerate(labels_count.values())}
    return class_weights

# util function to get the initial epoch number from the checkpoint name
def get_init_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    filename = os.path.splitext(filename)[0]
    init_epoch = filename.split("-")[1]
    return int(init_epoch)

run_training = True
run_finetune = True

class_indices_path = 'class_indices.npy'
initial_model_path = 'bird_classify_finetune_initial.h5'
final_model_path = 'bird_classify_finetune_final.h5'

# check which of the training steps still need to complete
# if saved model file is already there, then that step is considered complete
if os.path.isfile(initial_model_path):
    run_training = False
    print("[Info] Initial model exists '{}'. Skipping training step.".format(initial_model_path))

if os.path.isfile(final_model_path):
    run_finetune = False
    print("[Info] Fine-tuned model exists '{}'. Skipping fine-tuning step.".format(final_model_path))

load_from_checkpoint_train = False

training_checkpoint_dir = 'checkpoints/training'
if run_training and len(os.listdir(training_checkpoint_dir)) > 0:
    # the checkpoint to load and continue from
    training_checkpoint = os.path.join(training_checkpoint_dir, os.listdir(training_checkpoint_dir)[len(os.listdir(training_checkpoint_dir))-1])
    load_from_checkpoint_train = True

init_epoch_train = 0
if load_from_checkpoint_train:
    # get the epoch number to continue from
    print(training_checkpoint)
    init_epoch_train = get_init_epoch(training_checkpoint)
    print("[Info] Training checkpoint found for epoch {}. Will continue from that step.".format(init_epoch_train))


load_from_checkpoint_finetune = False

finetune_checkpoint_dir = 'checkpoints/finetune'
if run_finetune and len(os.listdir(finetune_checkpoint_dir)) > 0:
    # the checkpoint to load and continue from
    finetune_checkpoint = os.path.join(finetune_checkpoint_dir, os.listdir(finetune_checkpoint_dir)[len(os.listdir(finetune_checkpoint_dir))-1])
    load_from_checkpoint_finetune = True

init_epoch_finetune = 0
if load_from_checkpoint_finetune:
    # get the epoch number to continue from
    init_epoch_finetune = get_init_epoch(finetune_checkpoint)
    print("[Info] Training checkpoint found for epoch {}. Will continue from that step.".format(init_epoch_finetune))


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

# save the class indices for use in the predictions
np.save(class_indices_path, train_generator.class_indices)

# calculate the training steps
nb_train_samples = len(train_generator.filenames)
train_steps = int(math.ceil(nb_train_samples / batch_size))

# calculate the validation steps
nb_validation_samples = len(validation_generator.filenames)
validation_steps = int(math.ceil(nb_validation_samples / batch_size))

# get the class weights
class_weights = get_class_weights(train_data_dir)

if run_training:
    if load_from_checkpoint_train:
        model = load_model(training_checkpoint)
    else:
        # create the base pre-trained model
        base_model = InceptionV3(
            weights='imagenet', 
            include_top=False, 
            input_tensor=Input(shape=(img_width, img_height, 3))
            )

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
                                patience=5, 
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
                        initial_epoch=init_epoch_train,
                        callbacks=callbacks_list
                        )

    model.save(initial_model_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_generator, steps=validation_steps)

    print("\n")

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    graph_training_history(history, save_fig=True, save_path='training.png')

else:
    # training step is already completed
    # load the already trained model
    model = load_model(initial_model_path)


# Run Fine-tuning on our model
if run_finetune:
    # number of epochs to fine-tune
    ft_epochs = 25

    # reset our data generators
    train_generator.reset()
    validation_generator.reset()

    if load_from_checkpoint_finetune:
        model = load_model(finetune_checkpoint)
    else:
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

    filepath = finetune_checkpoint_dir + "/model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5"
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
                                patience=5, 
                                restore_best_weights=True
                                )

    callbacks_list = [checkpoint, early_stop]
    
    history = model.fit(
                        train_generator,
                        steps_per_epoch=train_steps,
                        epochs=ft_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        class_weight=class_weights,
                        max_queue_size=15,
                        workers=8,
                        initial_epoch=init_epoch_finetune,
                        callbacks=callbacks_list
                        )

    model.save(final_model_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_generator, steps=validation_steps)

    print("\n")

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    graph_training_history(history, save_fig=True, save_path='finetune.png')

    end_time = time.time()

    training_duration = end_time - start_time
    print("[INFO] Total Time for training: {} seconds".format(training_duration))
