import tensorflow as tf

from tensorflow.keras import layers
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import cv2

# Load the image, crop just the face, and return image data as a numpy array
def load_image(image_file_path):
    img = PIL.Image.open(image_file_path)
    img = img.crop([25,65,153,193])
    img = img.resize((64,64))
    data = np.asarray( img, dtype="int32" )
    return data

dataset_path = "celeba-dataset/img_align_celeba/img_align_celeba/"

# load the list of training images
train_images = np.array(os.listdir(dataset_path))

BUFFER_SIZE = 2000
BATCH_SIZE = 8

# shuffle and list
np.random.shuffle(train_images)
# chunk the training images list in to batches
train_images = np.split(train_images[:BUFFER_SIZE],BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(4*4*1024, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4,4,1024)))
    assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, (5, 5), strides = (2,2), padding = "same", use_bias = False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5,5), strides = (2,2), padding = "same", use_bias = False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5,5), strides = (2,2), padding = "same", use_bias = False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), strides = (2,2), padding = "same", use_bias = False, activation = "tanh"))
    assert model.output_shape == (None, 64, 64, 3)

    return model

generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise, training = False)
plt.imshow(generated_image[0], interpolation="nearest")
plt.show()
plt.close()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
# output will be something like tf.Tensor([[-6.442342e-05]], shape=(1, 1), dtype=float32)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16

# setting the seed for the randomization, so that we can reproduce the results
tf.random.set_seed(1234)
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    # pre-process the images
    new_images = []
    for file_name in images:
        new_pic = load_image(dataset_path + file_name)
        new_images.append(new_pic)
    
    images = np.array(new_images)
    images = images.reshape(images.shape[0], 64, 64, 3).astype('float32')
    images = (images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    images = None

def train(dataset, epochs):  
    tf.print("Starting Training")
    train_start = time.time()
    
    for epoch in range(epochs):
        start = time.time()
        tf.print("Starting Epoch:", epoch)

        batch_count = 1
        for image_batch in dataset:
            train_step(image_batch)
            batch_count += 1

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        tf.print("Epoch:", epoch, "finished")
        tf.print()
        tf.print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        tf.print()

    # Save the model every epoch
    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for total training is {} sec'.format(time.time()-train_start))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False. 
    # This is so all layers run in inference mode.
    predictions = model(test_input, training=False).numpy()

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        image = predictions[i]
        plt.imshow(image)
        plt.axis('off')

    plt.savefig('output/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

train(train_images, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
cv2.destroyAllWindows()

anim_file = 'dcgan_faces.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('output/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
        cv2.imshow("Results", image)
        cv2.waitKey(100)
    image = imageio.imread(filename)
    writer.append_data(image)