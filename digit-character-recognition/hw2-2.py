import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import time
import os
import urllib.request
import ssl

from HodaDatasetReader import read_hoda_cdb

# Load and process Hoda Digits dataset
def load_hoda_cdb(dataset_path='hoda_dataset/Digits', target_size=(28, 28)):
    print("\nLoading Hoda Dataset from .cdb files...")
    
    train_path = os.path.join(dataset_path, 'Train 60000.cdb')
    test_path = os.path.join(dataset_path, 'Test 20000.cdb')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Could not find files in {dataset_path}")
        return None, None

    print("Reading Train 60000.cdb (This might take a moment)...")
    train_images, train_labels = read_hoda_cdb(train_path)
    
    print("Reading Test 20000.cdb ...")
    test_images, test_labels = read_hoda_cdb(test_path)
    
    # Resize images to 28x28 and normalize
    def process_images(raw_images):
        processed = []
        for img in raw_images:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            processed.append(resized)
        
        processed_array = np.array(processed, dtype='float32') / 255.0
        processed_array = np.expand_dims(processed_array, -1)
        return processed_array

    print("Resizing and processing images...")
    x_train = process_images(train_images)
    x_test = process_images(test_images)
    
    y_train = np.array(train_labels, dtype='int')
    y_test = np.array(test_labels, dtype='int')
    
    print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# Build CNN model dynamically (10 classes for digits)
def build_model(input_shape, use_pooling=True, conv_stride=(1, 1), padding_type='valid'):
    model = models.Sequential()
    
    model.add(layers.InputLayer(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type))
    
    if use_pooling:
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.Conv2D(32, (3, 3), strides=conv_stride, activation='relu', padding=padding_type))
        
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type))
    
    if use_pooling:
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.Conv2D(64, (3, 3), strides=conv_stride, activation='relu', padding=padding_type))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax')) # 10 classes (0-9)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main execution block
train_data, test_data = load_hoda_cdb()

if train_data and test_data:
    x_train, y_train = train_data
    x_test, y_test = test_data
    input_shape = x_train.shape[1:]

    configs = [
        {"name": "Baseline (Max Pooling)", "use_pooling": True, "stride": (1,1), "pad": "valid"},
        {"name": "No Pooling, Stride (1,2)", "use_pooling": False, "stride": (1,2), "pad": "valid"},
        {"name": "No Pooling, Stride (2,1)", "use_pooling": False, "stride": (2,1), "pad": "valid"},
        {"name": "No Pooling, Stride (2,2)", "use_pooling": False, "stride": (2,2), "pad": "valid"},
        {"name": "No Pooling, Stride (2,2) + Same Padding", "use_pooling": False, "stride": (2,2), "pad": "same"},
    ]

    epochs = 5

    for config in configs:
        print(f"\n--- Training Model: {config['name']} ---")
        model = build_model(input_shape=input_shape, 
                            use_pooling=config['use_pooling'], 
                            conv_stride=config['stride'], 
                            padding_type=config['pad'])
        
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_data=(x_test, y_test), verbose=1)
        end_time = time.time()
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Result for {config['name']}:")
        print(f"Accuracy: {test_acc:.4f} | Training Time: {end_time - start_time:.2f} seconds")