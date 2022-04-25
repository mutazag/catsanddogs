
import mlflow
# autolog your metrics, parameters, and model
mlflow.autolog()
mlflow.set_tag('logging', 'mlflow.autolog()')



## command line arguments, input_dataset to hold the path for mounted files
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset', dest='input_dataset', default=os.path.expanduser('/mnt/tmp/cats_dogs'))
args, _ = parser.parse_known_args()

zip_dir = args.input_dataset
print('################################## input dataset: {}'.format(zip_dir))
mlflow.set_tag('input_dataset', zip_dir)


import pathlib
parent_dir = pathlib.Path(zip_dir)

import os
print(os.listdir(zip_dir))

train_dir = parent_dir / 'train'
test_dir = parent_dir / 'validation'

print(train_dir)
print(test_dir)

print('############################### import tensorflow')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_img_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_img_gen = ImageDataGenerator(rescale=1./255)
batch_size=20


train_data_gen = train_img_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(100, 100),
    class_mode='binary')


test_data_gen = test_img_gen.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    target_size=(100, 100),
    class_mode='binary')

print(test_data_gen.class_indices)

total_train = 2000
total_val = 1000

from tensorflow.keras.applications import VGG16


input_shape = (100, 100, 3)

base_model = VGG16(
    input_shape=input_shape,
    weights='imagenet',
    include_top=False)

frozen_layers = 15


for layer in base_model.layers[:frozen_layers]:
    layer.trainable = False

base_model.summary()

from tensorflow.keras.layers import Flatten, Dense

tuned_model = tf.keras.Sequential([
    base_model,
    Flatten(),
    Dense(500, activation='relu'),
    Dense(1, activation='sigmoid')
])


optimizer = tf.keras.optimizers.Adam(0.00001)

tuned_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

tuned_model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.0000001)



print('############################# model checkpoint')
checkpoint_filepath = os.path.expanduser('/tmp/checkpoint')
print(checkpoint_filepath)

model_checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


history = tuned_model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=50,
    validation_data=test_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[early_stop_cb, reduce_lr, model_checkpoint_cb]
)


## log a plot

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.savefig("actuals_vs_predictions.png")
mlflow.log_artifact("actuals_vs_predictions.png")

