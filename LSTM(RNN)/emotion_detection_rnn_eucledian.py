import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Function to extract features from images
def extract_features(image_path):
    image = imread(image_path, as_gray=True)
    resized_image = resize(image, (100, 100))
    feature_vector = resized_image.flatten()
    return feature_vector

# Function to create sequences for LSTM
def create_sequences(data_dir, num_samples_per_category):
    sequences = []
    labels = []
    emotions = os.listdir(data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(emotions)
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_dir):
            image_files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if len(image_files) >= num_samples_per_category:
                selected_image_files = np.random.choice(image_files, size=num_samples_per_category, replace=False)
                for image_file in selected_image_files:
                    image_path = os.path.join(emotion_dir, image_file)
                    feature_vector = extract_features(image_path)
                    sequences.append(feature_vector)
                    labels.append(label_encoder.transform([emotion])[0])
    sequences = pad_sequences(sequences, maxlen=10000)  # Pad sequences to the same length
    labels = np.array(labels)
    return sequences, labels, label_encoder

# Function to create and compile the LSTM model
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=(input_shape, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the LSTM model
def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    model = create_lstm_model(input_shape, num_classes)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for LSTM
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)  # Reshape for LSTM
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model

# Function to calculate accuracy and precision
def calculate_metrics(model, X_val, y_val):
    y_pred = model.predict(X_val).argmax(axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    return accuracy, precision

# Function to predict emotion for a new image
def predict_emotion(lstm_model, image_path):
    feature_vector = extract_features(image_path)
    feature_vector = pad_sequences([feature_vector], maxlen=10000)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1], 1)  # Reshape for LSTM
    prediction = lstm_model.predict(feature_vector)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to handle file drop event
def on_drop(event):
    try:
        # Get the file paths of dropped files
        file_paths = event.data.split()

        # Predict emotion for each dropped image
        for file_path in file_paths:
            predicted_class = predict_emotion(lstm_model, file_path.decode("utf-8"))
            predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to open file dialog for selecting images
def open_file_dialog():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_paths:
        for file_path in file_paths:
            predicted_class = predict_emotion(lstm_model, file_path)
            predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

# Create the main window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("600x400")  # Set window size

# Create a label for instructions
instructions_label = tk.Label(root, text="Drag and drop images or click the button to select images \n (LSTM with Euclidean Distance metrics)", font=("Arial", 12))
instructions_label.pack(pady=10)

# Create a button to open file dialog
open_button = tk.Button(root, text="Select Images", command=open_file_dialog, font=("Arial", 12))
open_button.pack(pady=5)

# Labels to display accuracy and precision
accuracy_label = tk.Label(root, text="Validation Accuracy: N/A", font=("Arial", 12))
accuracy_label.pack(pady=5)
precision_label = tk.Label(root, text="Precision: N/A", font=("Arial", 12))
precision_label.pack(pady=5)

# Load and prepare the data
data_dir = "C:\\Users\\pro_j\\Desktop\\project\\new\\GUI\\CK+48"
num_samples_per_category = 75
X, y, label_encoder = create_sequences(data_dir, num_samples_per_category)
input_shape = X.shape[1]
num_classes = len(np.unique(y))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LSTM model
lstm_model = train_lstm_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

# Calculate accuracy and precision
accuracy, precision = calculate_metrics(lstm_model, X_val, y_val)

# Update accuracy and precision labels
accuracy_label.config(text=f"Validation Accuracy: {accuracy:.2f}")
precision_label.config(text=f"Precision: {precision:.2f}")

# Run the application
root.mainloop()
