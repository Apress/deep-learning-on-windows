from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# define the parameters for the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img('data/Bird.jpg')  # this is a PIL image

# convert image to numpy array with shape (3, width, height)
img_arr = img_to_array(img)

# convert to numpy array with shape (1, 3, width, height)
img_arr = img_arr.reshape((1,) + img_arr.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `data/augmented` directory
i = 0
for batch in datagen.flow(
        img_arr,
        batch_size=1,
        save_to_dir='data/augmented',
        save_prefix='Bird_A',
        save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
