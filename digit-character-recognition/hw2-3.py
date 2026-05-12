import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import time
import os
import urllib.request
import ssl

from HodaDatasetReader import read_hoda_cdb

def load_combined_hoda(dataset_path='hoda_dataset', target_size=(28, 28)):
    print("\nLoading Digits and Letters datasets...")
    
    # مسیرهای فایل‌های ارقام و حروف
    d_train_path = os.path.join(dataset_path, 'Digits', 'Train 60000.cdb')
    d_test_path = os.path.join(dataset_path, 'Digits', 'Test 20000.cdb')
    l_train_path = os.path.join(dataset_path, 'Letters', 'Persian-Character-DB-Training.cdb')
    l_test_path = os.path.join(dataset_path, 'Letters', 'Persian-Character-DB-Test.cdb')
    
    # خواندن ارقام
    print("Reading Digits (.cdb files)...")
    d_train_img, d_train_lbl = read_hoda_cdb(d_train_path)
    d_test_img, d_test_lbl = read_hoda_cdb(d_test_path)
    
    # خواندن حروف
    print("Reading Letters (.cdb files)...")
    l_train_img, l_train_lbl = read_hoda_cdb(l_train_path)
    l_test_img, l_test_lbl = read_hoda_cdb(l_test_path)
    
    # --- اصلاح و شیفت دادن برچسب‌ها ---
    # ارقام رو بین 0 تا 9 و حروف رو از 10 به بعد شماره‌گذاری می‌کنیم تا تداخل نداشته باشن
    d_unique = np.unique(np.concatenate((d_train_lbl, d_test_lbl)))
    d_map = {val: i for i, val in enumerate(d_unique)}
    
    l_unique = np.unique(np.concatenate((l_train_lbl, l_test_lbl)))
    l_map = {val: i + len(d_unique) for i, val in enumerate(l_unique)}
    
    d_train_lbl = np.array([d_map[lbl] for lbl in d_train_lbl])
    d_test_lbl = np.array([d_map[lbl] for lbl in d_test_lbl])
    
    l_train_lbl = np.array([l_map[lbl] for lbl in l_train_lbl])
    l_test_lbl = np.array([l_map[lbl] for lbl in l_test_lbl])
    
    # --- ترکیب داده‌ها ---
    train_images_raw = d_train_img + l_train_img
    test_images_raw = d_test_img + l_test_img
    
    y_train = np.concatenate([d_train_lbl, l_train_lbl])
    y_test = np.concatenate([d_test_lbl, l_test_lbl])
    
    # تابعی برای تغییر اندازه و پیش‌پردازش
    def process_images(raw_images):
        processed = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in raw_images]
        processed_array = np.array(processed, dtype='float32') / 255.0
        return np.expand_dims(processed_array, -1)

    print("Resizing and combining images (This will take a minute)...")
    x_train = process_images(train_images_raw)
    x_test = process_images(test_images_raw)
    
    # بُر زدن (Shuffle) داده‌های آموزشی تا حروف و ارقام با هم مخلوط شوند
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    print(f"Combined Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# 3. تابع ساخت مدل (تعداد کلاس‌ها به صورت خودکار تعیین می‌شود)
def build_model(input_shape, num_classes, use_pooling=True, conv_stride=(1, 1), padding_type='valid'):
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
    
    # لایه خروجی برای ترکیب حروف و ارقام
    model.add(layers.Dense(num_classes, activation='softmax')) 
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 4. اجرای تست‌ها
train_data, test_data = load_combined_hoda()

if train_data and test_data:
    x_train, y_train = train_data
    x_test, y_test = test_data
    input_shape = x_train.shape[1:]
    
    num_classes = int(max(np.max(y_train), np.max(y_test))) + 1
    print(f"\nDetected {num_classes} unique classes (Digits + Letters).")

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
                            num_classes=num_classes,
                            use_pooling=config['use_pooling'], 
                            conv_stride=config['stride'], 
                            padding_type=config['pad'])
        
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_data=(x_test, y_test), verbose=1)
        end_time = time.time()
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Result for {config['name']}:")
        print(f"Accuracy: {test_acc:.4f} | Training Time: {end_time - start_time:.2f} seconds")