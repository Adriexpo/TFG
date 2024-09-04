import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'Training_Images')

# Define the batch size
batch_size = 5

# Define your custom class order
custom_class_order = ['Poor', 'Played', 'Good', 'Excellent', 'Mint']

# Create a dictionary with the desired mapping
custom_class_indices = {class_name: i for i, class_name in enumerate(custom_class_order)}

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

# Load training data with custom class indices
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Specify a fixed size for the images
    batch_size=batch_size,
    class_mode='sparse',  # Use sparse mode for regression (numerical labels)
    subset='training',
    classes=custom_class_order  # Specify the custom class order
)

# Load validation data with custom class indices
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Specify a fixed size for the images
    batch_size=batch_size,
    class_mode='sparse',  # Use sparse mode for regression (numerical labels)
    subset='validation',
    classes=custom_class_order  # Specify the custom class order
)


# Print the custom mapping to verify
print("Custom class indices mapping:")
for class_name, index in custom_class_indices.items():
    print(f"Class '{class_name}' is mapped to label '{index}'")

# Print the number of examples found for each class
print("Number of examples found per class in the training set:")
for class_name, count in train_generator.class_indices.items():
    print(f"{class_name}: {np.sum(train_generator.labels == count)}")

print("\nNumber of examples found per class in the validation set:")
for class_name, count in validation_generator.class_indices.items():
    print(f"{class_name}: {np.sum(validation_generator.labels == count)}")

# Define the deep model for regression
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
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
    Dense(1, activation='linear')  # Single output for regression.
])

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Compile the model for regression
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule), 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error'])

# Summary of the model
model.summary()

# Implement Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

# Define custom callback to skip saving certain epochs
class SkipEpochSaving(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_mae = logs.get('val_mean_absolute_error')
        if loss == 0.0 or loss == 1 or val_mae == 0.0 or val_mae == 1.0:
            print(loss)
            print(f"Skipping epoch {epoch+1} due to unacceptable validation metrics (loss={loss}, val_mae={val_mae}).")
            # Skip saving this epoch but continue training
            self.model.stop_training = False  # Ensure training continues

# Save the best model only when valid
model_checkpoint = ModelCheckpoint(
    'card_condition_regressor.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

steps_per_epoch = train_generator.samples // batch_size
if steps_per_epoch < train_generator.samples / batch_size:
    steps_per_epoch += 1

# Train the model with Early Stopping and SkipEpochSaving
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20,  # Increased number of epochs
    callbacks=[early_stopping, SkipEpochSaving(), model_checkpoint]  # Include custom callback
)

# Evaluate the model on validation data
val_loss, val_mae = model.evaluate(validation_generator)
print(f"Validation Mean Absolute Error: {val_mae:.2f}")

# Create confusion matrix based on rounded predictions
# Get the true labels and predictions
true_labels = validation_generator.classes
predictions = model.predict(validation_generator)
rounded_predictions = np.round(predictions).astype(int)

# Create the confusion matrix
conf_matrix = confusion_matrix(true_labels, rounded_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix based on Rounded Predictions')

# Save the plot to a file instead of showing it interactively
plt.savefig('confusion_matrix.png')

# Optional: Display the file path where the plot was saved
print("Confusion matrix plot saved as 'confusion_matrix.png'")