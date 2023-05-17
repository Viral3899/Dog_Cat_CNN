from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

# Define the model
classifier = Sequential([
    Conv2D(64, 3, padding='same', activation='relu', input_shape=(64, 64, 1)),  # Update input shape to (64, 64, 1)
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'cats_and_dogs_filtered/train',
    target_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',  # Set color_mode to grayscale
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'cats_and_dogs_filtered/validation',
    target_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',  # Set color_mode to grayscale
    class_mode='binary'
)

checkpoint_path = 'model_checkpoint.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model with checkpoints
try:
    classifier.load_weights(checkpoint_path)
    print('Loaded model from checkpoint')
except (OSError, ValueError):
    print('No saved model found. Training from scratch.')

# Train the model with checkpoints
model = classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=100,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[checkpoint]
)

# Save the model
classifier.save('model.h5')
print('Saved model to disk')
