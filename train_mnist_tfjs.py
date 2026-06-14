#!/usr/bin/env python3
"""
Train a CNN on MNIST and export a TensorFlow.js model.

Usage:
  pip install -r requirements.txt
  python train_mnist_tfjs.py --epochs 15 --batch_size 128 --out_dir model
"""

import argparse
import os
import numpy as np
import tensorflow as tf


def build_model(input_shape=(28, 28, 1), num_classes=10):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                         Dropout, Flatten, Dense)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def preprocess_data(x, y):
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, -1)
    return x, y


def main(args):
    print('Loading MNIST dataset...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    val_split = 0.1
    val_count = int(len(x_train) * val_split)
    x_val = x_train[:val_count]
    y_val = y_train[:val_count]
    x_train = x_train[val_count:]
    y_train = y_train[val_count:]

    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, mode='max'),
        tf.keras.callbacks.ModelCheckpoint('best_mnist.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.10,
        height_shift_range=0.10,
        rotation_range=8
    )
    datagen.fit(x_train)

    steps_per_epoch = max(100, len(x_train) // args.batch_size)
    print(f'Training for {args.epochs} epochs with batch size {args.batch_size}...')

    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc * 100:.4f}%')

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Saving Keras model to mnist_keras.h5 and SavedModel to mnist_saved_model...')
    model.save('mnist_keras.h5')
    model.save('mnist_saved_model')

    try:
        import tensorflowjs as tfjs
    except ImportError:
        print('ERROR: tensorflowjs is not installed. Install with `pip install tensorflowjs`.')
        return

    print(f'Exporting TensorFlow.js model to {args.out_dir} ...')
    tfjs.converters.save_keras_model(model, args.out_dir)
    print('Export complete. Copy the `model.json` and shard files into the frontend `model/` folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST CNN and export TF.js model')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default='model')
    args = parser.parse_args()
    main(args)
