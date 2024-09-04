import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Define the path to your dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'Training_Images')

# Define the batch size
batch_size = 2

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.20,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data without resizing
train_generator = datagen.flow_from_directory(
    dataset_path,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data without resizing
validation_generator = datagen.flow_from_directory(
    dataset_path,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Print the number of examples found for each class
print("Number of examples found per class in the training set:")
for class_name, count in train_generator.class_indices.items():
    print(f"{class_name}: {np.sum(train_generator.labels == count)}")

print("\nNumber of examples found per class in the validation set:")
for class_name, count in validation_generator.class_indices.items():
    print(f"{class_name}: {np.sum(validation_generator.labels == count)}")

# Define the deep model with increased layers and regularization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(5, activation='softmax')  # 5 output classes: Mint, Excellent, Good, Played, Poor
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Implement Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

# Train the model with Early Stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=60,  # Increased number of epochs
    callbacks=[early_stopping]  # Only use early stopping
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc:.2f}")

# Save the trained model
model.save('card_condition_classifier.keras')

# Function to make predictions on new images
def predict_image(img_path, model):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Remember to rescale as done during training
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    classes = ['Mint', 'Excellent', 'Good', 'Played', 'Poor']
    return classes[class_idx]

# Example usage of the prediction function
# img_path = 'path_to_some_image.jpg'
# prediction = predict_image(img_path, model)
# print(f"The image is predicted as: {prediction}")
