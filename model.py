import os
import json
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_FROZEN = 150

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Class indices:", train_generator.class_indices)

train_dir = 'data/train'
class_names = sorted(os.listdir(train_dir))

class_counts = {
    i: len(os.listdir(os.path.join(train_dir, name)))
    for i, name in enumerate(class_names)
}

total = sum(class_counts.values())
n_classes = len(class_counts)

class_weights = {
    i: total / (n_classes * count)
    for i, count in class_counts.items()
}

print("Class counts:", class_counts)
print("Class weights:", class_weights)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='elu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath='model/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\n--- Phase 1: Training with frozen base ---")

history_frozen = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_phase1
)

shutil.copy('model/best_model.keras', 'model/best_model_ft.keras')
print("\nBest model copied to model/best_model_ft.keras")

model.save('model/friend_classifier.keras')
print("Model saved to model/friend_classifier.keras")

history_combined = {
    'accuracy': history_frozen.history['accuracy'],
    'val_accuracy': history_frozen.history['val_accuracy'],
    'loss': history_frozen.history['loss'],
    'val_loss': history_frozen.history['val_loss'],
    'phase_split': len(history_frozen.history['accuracy'])
}

with open('model/history.json', 'w') as f:
    json.dump(history_combined, f)

print("Training history saved to model/history.json")