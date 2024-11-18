import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from PIL import Image
from keras import Sequential
from keras.src.saving import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data exploration and preparation
def load_and_preprocess_images(main_dir, color_mode='grayscale'):
    X_train, y_train = [], []
    X_val, y_val = [], []

    # Load training data
    for label in ['male', 'female']:
        folder_path = os.path.join(main_dir, 'Training', label)
        images, labels = preprocess_images_folder(folder_path, 1 if label == 'male' else 0, color_mode)
        X_train.extend(images)
        y_train.extend(labels)

    # Load validation data
    for label in ['male', 'female']:
        folder_path = os.path.join(main_dir, 'Validation', label)
        images, labels = preprocess_images_folder(folder_path, 1 if label == 'male' else 0, color_mode)
        X_val.extend(images)
        y_val.extend(labels)

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)
# Data exploration and preparation
def preprocess_images_folder(folder_path, label, color_mode='grayscale'):
    processed_images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            processed_image = preprocess_image(image_path, color_mode)
            if processed_image is not None:
                processed_images.append(processed_image)
                labels.append(label)
    return processed_images, labels

# Data exploration and preparation
def preprocess_image(image_path, color_mode='grayscale'):
    try:
        if color_mode == 'grayscale':
            image = Image.open(image_path).resize((64, 64)).convert('L')
            normalized_image = np.array(image) / 255.0
            normalized_image = np.expand_dims(normalized_image, axis=-1)  # Add channel dimension for grayscale
        else:
            image = Image.open(image_path).resize((64, 64)).convert('RGB')
            normalized_image = np.array(image) / 255.0

        print(
            f"Successfully processed image at: {image_path}, Image size: {image.size}, Image shape: {normalized_image.shape}")
        return normalized_image
    except Exception as e:
        print(f"Error processing image at: {image_path} - {e}")
        return None


def build_first_NN(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation='relu'),
        Dense(256, activation='tanh'),
        Dense(128, activation='relu'),
        Dense(64, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_second_NN(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='tanh'),
        Dense(128, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.suptitle(title)
    plt.show()

def plot_history3(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.suptitle(title)
    plt.show()

def create_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history3(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.suptitle(title)
    plt.show()

def firstExperiment(X_train, X_test, y_train, y_test):

    # Image Flattening
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Train an SVM model on the grayscale images
    svm = LinearSVC()
    svm.fit(X_train_flat, y_train)

    y_test_pred = svm.predict(X_test_flat)

    # Confusion matrix and the average f-1 scores
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Output results to a file named output.txt
    with open("output.txt", "w") as file:
        file.write("Experiment 1:\nConfusion Matrix:\n")
        file.write(np.array2string(test_conf_matrix))
        file.write("\n")
        file.write("Average F1 Score: " + str(test_f1) + "\n\n\n")

    print("Confusion Matrix:\n", test_conf_matrix)
    print("Average F1 Score:", test_f1)

def secondExperiment(X_train, X_test, y_train, y_test,X_val, y_val):
    input_shape = X_train[0].shape

    model1 = build_first_NN(input_shape)
    history1 = model1.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    plot_history(history1, 'First Model')
    val_accuracy1 = max(history1.history['val_accuracy'])

    model2 = build_second_NN(input_shape)
    history2 = model2.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    plot_history(history2, 'Second Model')
    val_accuracy2 = max(history2.history['val_accuracy'])

    if val_accuracy1 > val_accuracy2:
        best_model = model1
        best_model.save('best_model.h5')
        print('First model saved as the best model.')
    else:
        best_model = model2
        best_model.save('best_model.h5')
        print('Second model saved as the best model.')

    best_model = load_model('best_model.h5')
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(cm)

    f1 = f1_score(y_test, y_pred)
    print('F1 Score:', f1)

    with open("output.txt", "a") as file:
        file.write("Experiment 2:\nConfusion Matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n")
        file.write("Average F1 Score: " + str(f1) + "\n\n\n")

def thirdExperiment():
    X_train_gray, y_train_gray, X_val_gray, y_val_gray = load_and_preprocess_images(main_dir, color_mode='grayscale')

    # Split the training data into 80% training and 20% testing for grayscale
    X_train_gray, X_test_gray, y_train_gray, y_test_gray = train_test_split(X_train_gray, y_train_gray, test_size=0.2,
                                                                            random_state=42)

    # Build, train, and evaluate the grayscale model
    input_shape_gray = X_train_gray[0].shape
    model_gray = create_cnn(input_shape_gray)
    history_gray = model_gray.fit(X_train_gray, y_train_gray, epochs=5, validation_data=(X_val_gray, y_val_gray))
    plot_history3(history_gray, 'Grayscale Model')
    val_accuracy_gray = max(history_gray.history['val_accuracy'])

    # Load and preprocess images for RGB
    X_train_rgb, y_train_rgb, X_val_rgb, y_val_rgb = load_and_preprocess_images(main_dir, color_mode='rgb')

    # Split the training data into 80% training and 20% testing for RGB
    X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_split(X_train_rgb, y_train_rgb, test_size=0.2,
                                                                        random_state=42)

    # Build, train, and evaluate the RGB model
    input_shape_rgb = X_train_rgb[0].shape
    model_rgb = create_cnn(input_shape_rgb)
    history_rgb = model_rgb.fit(X_train_rgb, y_train_rgb, epochs=5, validation_data=(X_val_rgb, y_val_rgb))
    plot_history3(history_rgb, 'RGB Model')
    val_accuracy_rgb = max(history_rgb.history['val_accuracy'])

    # Compare the validation accuracy and save the best model
    if val_accuracy_gray > val_accuracy_rgb:
        best_model = model_gray
        best_model.save('best_model.h5')
        print('Grayscale model saved as the best model.')
        X_test, y_test = X_test_gray, y_test_gray
    else:
        best_model = model_rgb
        best_model.save('best_model.h5')
        print('RGB model saved as the best model.')
        X_test, y_test = X_test_rgb, y_test_rgb

    # Load the best model and evaluate it on the test set
    best_model = load_model('best_model.h5')
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(cm)

    f1 = f1_score(y_test, y_pred)
    print('F1 Score:', f1)

    with open("output.txt", "a") as file:
        file.write("Experiment 3:\nConfusion Matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n")
        file.write("Average F1 Score: " + str(f1) + "\n\n\n")


if __name__ == '__main__':
    # Replace with the actual directory of your dataset
    main_dir = 'example:\\example\\example'
    X_train, y_train, X_val, y_val = load_and_preprocess_images(main_dir)

    # Split the data into training and testing datasets (if there is no testing dataset)
    # Training 80% - Testing 20%
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=7)

    # First Experiment
    firstExperiment(X_train, X_test, y_train, y_test)
    # Second Experiment
    secondExperiment(X_train, X_test, y_train, y_test, X_val, y_val)
    # Third Experiment
    thirdExperiment()
