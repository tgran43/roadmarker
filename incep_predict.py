from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import numpy

train_data_dir = 'D:/output2/train'
validation_data_dir = 'D:/output2/val'
test_data_dir = 'D:/output2/test'
img_width, img_height = 150, 150
batch_size = 1
num_img = 747

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_ds = test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_ds = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_ds = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model = keras.models.load_model('incepres_model_tuned_updated_output_layer_tuned')

predictions = model.predict(test_ds, steps=num_img)
print('predicted labels:')
print(predictions)
print('true labels:')
print(validation_ds.classes)
