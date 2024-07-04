import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# from google.colab import files 
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.layers import Conv2D, MaxPooling2D,Dense,Activation,Dropout,Flatten,BatchNormalization
from keras.layers import Dense,Dropout,Input,MaxPooling2D,MaxPool2D,Conv2D,Flatten
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 256
BATCH_SIZE = 64

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)


# fix_gpu()

datagen = tf.keras.preprocessing.image.ImageDataGenerator()

clTrain = os.listdir(r"C:\Users\parvs\Downloads\train_and_test\train")
clTest = os.listdir(r"C:\Users\parvs\Downloads\train_and_test\test")

# Train generator for train folder
train_generator = datagen.flow_from_directory(
    r"C:\Users\parvs\Downloads\train_and_test\train",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    classes = clTrain
    )

# Test generator for test folder
test_generator = datagen.flow_from_directory(
    r"C:\Users\parvs\Downloads\train_and_test\test",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    classes = clTest
    )

print(len(clTest))

# Define image size
IMAGE_SIZE = 256

# Load pre-trained ResNet50 model without the top layer
resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Freeze base layers (optional, adjust based on training needs)
for layer in resnet.layers:
    layer.trainable = False

# Build the model
model = Sequential()

# Add pre-trained ResNet50 layers
model.add(resnet)

# Add custom layers for classification
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="softmax"))

# Print model summary
model.summary()

# Choose optimizer and compile the model
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Define callbacks
epochs = 10

checkpoint = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# Train the model with callbacks
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=len(test_generator),
                    callbacks=[checkpoint, early_stopping, learning_rate_scheduler])
