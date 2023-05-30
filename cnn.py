import idx2numpy
import numpy as np
import pandas as pd
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt

emnist_labels = [
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57
]



def emnist_model():
    """ Модель сверточной нейронной сети. """

    model = Sequential()
    model.add(Convolution2D(
        filters=32, kernel_size=(3, 3), activation='relu',
        input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(
        filters=64, kernel_size=(3, 3), padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(
        filters=128, kernel_size=(3, 3), padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


def training(model):
    """ Обучение модели. """

    emnist_path = 'C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer'
    df_train = pd.read_csv(emnist_path + '\\emnist-digits-train.csv')
    df_test = pd.read_csv(emnist_path + '\\emnist-digits-test.csv')
    x_train =  df_train.iloc[:,1:]
    y_train =  df_train.iloc[:,0]
    x_test =  df_test.iloc[:,1:]
    y_test =  df_test.iloc[:,0]
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0

    y_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', patience=3,
        verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(
        x_train, y_train_cat, validation_data=(x_test, y_test_cat),
        callbacks=[learning_rate_reduction], batch_size=64, epochs=3)
    
if __name__ == '__main__':
#    model = emnist_model()
#    training(model)
#    model.save('C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer\\proj_model.h5')
    df_train = pd.read_csv('C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer\\emnist-digits-train.csv')
    x_train =  df_train.iloc[:,1:]
    y_train =  df_train.iloc[:,0]
    sample_image = x_train.iloc[35]
    sample_label = y_train.iloc[35]
    sample_image.shape, sample_label
    print("Label entry 42:", emnist_labels[sample_label])
    plt.imshow(sample_image.values.reshape(28, 28), cmap=plt.cm.gray)
    plt.show()