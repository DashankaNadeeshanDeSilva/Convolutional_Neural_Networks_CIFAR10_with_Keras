from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
# from keras.utils import print_summary, to_categorical
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sn
import pandas as pd

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras_sequential_ascii import sequential_model_to_ascii_printout

from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler

# Parameters
TRAINING_EPOCHS = 10


# Learning rate scheduling
def lr_schedule(epoch):
    l_rate = 0.001
    if epoch > 75:
        l_rate = 0.005
    if epoch > 100:
        l_rate = 0.003
    return l_rate


# Data preparation ----------------------------------------
# Training and test data arrange
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data normalization - Zero mean and unit variance
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

# Class label setup
NUM_CLASSES = 10
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

# Define the CNN model ----------------------------------------
# Includes:
# 1.3x3 convolution filters,
# 2.layer weight regularizers (L2 reg),
# 3.ELU activation,
# 4.Batch normalization,
# 5.Max pooling
# 6.Dropout
# Hyperparameters: placing of dropout with option of dropout ratio and batch normalization can be changed
WEIGHT_DECAY = 1e-4

model = Sequential()
# Conv layer 1
model.add(
    Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
# Conv layer 2
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# Conv layer 3
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
# Conv layer 4
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# Conv layer 5
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
# Conv layer 6
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
# Conv layer 7
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
# Conv layer 8
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
# Conv layer 9
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
# Conv layer 10
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
# Flatten
model.add(Flatten())
# Dense or fully connected layer
model.add(Dense(NUM_CLASSES, activation='softmax'))

print(model.summary())

# Data augmentation for more synthetic training data
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)

# Training ----------------------------------------
TRAINING_BATCH_SIZE = 128
# Optimizer and loss function introduction
opt_rms = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

# Time recording for the process
starttime = datetime.datetime.now()
cnn_model = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=TRAINING_BATCH_SIZE),
                                steps_per_epoch=x_train.shape[0] // TRAINING_BATCH_SIZE,
                                epochs=TRAINING_EPOCHS,
                                verbose=1,
                                validation_data=(x_test, y_test),
                                callbacks=[LearningRateScheduler(lr_schedule)])

endtime = datetime.datetime.now()
print(endtime - starttime)

# Model weights save ----
model_json = model.to_json()
with open('model,json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Testing ----------------------------------------
TEST_BATCH_SIZE = 128
scores = model.evaluate(x_test, y_test, batch_size=TEST_BATCH_SIZE, verbose=1)
# Results
print('Test loss:', scores[0])
print('Test accuracy:', scores[1] * 100, "%")

# Performance graphs ----
# Training accuracy vs Validation accuracy
plt.figure(0)
plt.plot(cnn_model.history['accuracy'], 'mediumaquamarine')
plt.plot(cnn_model.history['val_accuracy'], 'royalblue')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])
plt.savefig('accuracy_graphs.png')

# Training loss vs Validation loss
plt.figure(1)
plt.plot(cnn_model.history['loss'], 'mediumaquamarine')
plt.plot(cnn_model.history['val_loss'], 'royalblue')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])
plt.show()
plt.savefig('loss_graphs.png')

# Confusion matrix ----
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for i in range(10):
    print(i, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[i].sum())
con_mat = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# Draw confusion matrix
print(con_mat)

# Visualizing of confusion matrix
df_cm = pd.DataFrame(con_mat, range(10), range(10))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, cmap="BuPu", annot=True, annot_kws={"size": 12})  # font size
plt.savefig('confusion_matrix.png')
plt.show()
