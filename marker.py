from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import easygui

train_data_dir = 'D:/output2/train'
validation_data_dir = 'D:/output2/val'
test_data_dir = 'D:/output2/test'
img_width, img_height = 299, 299
batch_size = 32
CLASSES = [0, 1]
CLASS_NAMES = ['Marker',
               'No Marker']
date = time.strftime("%Y%m%d-%H%M%S")
log_folder = "logs/" + date

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
    class_mode='categorical')

validation_ds = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_ds = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

def fit_model():
    print("making model")
    base_model = keras.applications.InceptionResNetV2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(299, 299, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.
    x = base_model.output
    # Add a global average pooling layer
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # Add fully connected output layer with softmax activation for binary label classification
    outputs = keras.layers.Dense(2, activation='softmax')(x)
    # Assemble the modified model
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    callbacks = [TensorBoard(log_dir=log_folder,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             update_freq='epoch',
                             profile_batch=2,
                             embeddings_freq=1)]
    epochs = input("Enter number of epochs:")
    checkpoint_filepath = 'models/' + date + '_inceptionResNet_epoch{epoch:02d}'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True,
        save_freq="epoch"
    )

    model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[model_checkpoint_callback, callbacks])
    # model.save('incepres_model_10_epochs' + date)


def model_predict(input_model, input_ds, ds_str):
    print("-------------------------------------------------------")
    print(ds_str + "predicting...")
    predictions = input_model.predict(input_ds, input_ds.samples // batch_size + 1)
    np.savetxt(date + ds_str + "predictions.csv", predictions, delimiter=",")
    y_true = input_ds.classes
    y_pred = np.argmax(predictions, axis=1)
    np.savetxt(date + ds_str + "predictions_argmax", y_pred, delimiter=",")
    print(y_true)
    print(predictions)
    print(y_pred)
    # Generate and print classification metrics and confusion matrix
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES))
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, output_dict=True)
    with open(date + ds_str + 'classification_report.csv', 'w') as f:
        for key in report.keys():
            f.write("%s,%s\n" % (key, report[key]))
    conf_arr = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print(conf_arr)
    np.savetxt(date + ds_str + "confusion_matrix.csv", conf_arr, delimiter=",")
    print("-------------------------------------------------------")


# def plot_confusion(predictions, y_pred, y_true):
#     np.set_printoptions(precision=2)
#     # Plot non-normalized confusion matrix
#     titles_options = [("Confusion matrix, without normalization", None),
#                       ("Normalized confusion matrix", 'true')]
#     for title, normalize in titles_options:
#         disp = plot_confusion_matrix(predictions, y_pred, y_true,
#                                      display_labels=CLASS_NAMES,
#                                      cmap=plt.cm.Blues,
#                                      normalize=normalize)
#         disp.ax_.set_title(title)
#
#         print(title)
#         print(disp.confusion_matrix)
#
#         plt.show()


i = 0
while i == 0:
    action = input("Enter 'fit', 'predict' or 'exit':")
    j = 0
    if action == "fit":
        print("Fitting model...")
        fit_model()
    elif action == "predict":
        print("Loading Model...")
        selected_model = easygui.diropenbox()
        model = keras.models.load_model(selected_model)
        print("Model Loaded")
        while j == 0:
            data_set = input("Which dataset do you want to predict: 'test', 'train', 'validation' or 'all'")
            if data_set == "test":
                model_predict(model, test_ds, "_test_")
                j = 1
            elif data_set == "train":
                model_predict(model, train_ds, "_train_")
                j = 1
            elif data_set == "validation":
                model_predict(model, validation_ds, "_validation_")
                j = 1
            elif data_set == 'all':
                model_predict(model, test_ds, "_test_")
                model_predict(model, train_ds, "_train_")
                model_predict(model, validation_ds, "_validation_")
                j = 1
            else:
                print("Invalid input")
    elif action == 'exit':
        i = 1
    else:
        print("Invalid action")
        action = input("Enter 'fit' or 'predict':")
