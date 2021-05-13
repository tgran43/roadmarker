from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

train_data_dir = 'D:/output2/train'
validation_data_dir = 'D:/output2/val'
img_width, img_height = 150, 150
batch_size = 32
CLASSES = [0, 1]
CLASS_NAMES = ['marker',
               'no marker']
output_directory = 'predict'

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode="constant",
    shear_range=0.2,
    zoom_range=(0.5, 1),
    horizontal_flip=True,
    rotation_range=360,
    channel_shift_range=25,
    brightness_range=(0.75, 1.25))
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_ds = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_ds = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


def fit_model():
    global model
    print("making model")
    base_model = keras.applications.InceptionResNetV2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.
    x = base_model.output
    # Add a global average pooling layer
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # Add fully connected output layer with sigmoid activation for multi label classification
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    # Assemble the modified model
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    model.summary()
    model.save('incepres_model_deepweeds')


model = keras.models.load_model('incepres_model_deepweeds')
print("predicting...")
predictions = model.predict(validation_ds, 743 // batch_size + 1)
y_true = validation_ds.classes
y_pred = np.argmax(predictions, axis=1)
print(y_true)
print(y_pred)

# Generate and print classification metrics and confusion matrix
print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES))
report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, output_dict=True)
with open(output_directory + 'classification_report.csv', 'w') as f:
    for key in report.keys():
        f.write("%s,%s\n" % (key, report[key]))
conf_arr = confusion_matrix(y_true, y_pred, labels=CLASSES)
print(conf_arr)
np.savetxt(output_directory + "confusion_matrix.csv", conf_arr, delimiter=",")
