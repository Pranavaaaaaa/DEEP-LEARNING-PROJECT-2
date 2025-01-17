import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
import tensorflow as tf

# Function to load individual batches
def load_batch(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        labels = np.array(labels)
        return images, labels

# Function to load the entire CIFAR-10 dataset
def load_cifar10(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):  # Load batches 1-5 for training
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_batch(batch_file)
        x_train.append(images)
        y_train.append(labels)
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    # Load the test batch
    test_file = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_batch(test_file)

    return x_train, y_train, x_test, y_test

# Set the folder path where you extracted the CIFAR-10 dataset
data_dir = r'C:\Users\apran\Downloads\cifar-10-python\cifar-10-batches-py'

# Load the dataset
x_train, y_train, x_test, y_test = load_cifar10(data_dir)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Display the first 9 images with their labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

# Build a CNN model
model = models.Sequential([
    Input(shape=(32, 32, 3)),  # Define the input shape here
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
