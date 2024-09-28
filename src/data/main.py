import tensorflow as tf
from tensorflow import keras
import numpy as np

def main():
    print("Training one simple neural network...")

    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\nAccuracies in test:', test_acc)

if __name__ == "__main__":
    main()