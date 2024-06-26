import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.io import imread
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Extracting features from images
def extract_features(image_path):
    image = imread(image_path, as_gray=True)
    resized_image = resize(image, (100, 100))
    feature_vector = resized_image.flatten()
    return feature_vector

# Creating adjacency list
def create_adjacency_list(data_dir, num_samples_per_category):
    adjacency_list = {}
    emotions = os.listdir(data_dir)
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_dir):
            image_files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if len(image_files) >= num_samples_per_category:
                selected_image_files = np.random.choice(image_files, size=num_samples_per_category, replace=False)
                for image_file in selected_image_files:
                    image_path = os.path.join(emotion_dir, image_file)
                    feature_vector = extract_features(image_path)
                    adjacency_list[image_path] = {
                        "feature_vector": feature_vector,
                        "emotion": emotion
                    }
    return adjacency_list

# Function to train KNN model
def train_knn_model(adjacency_list):
    X = np.array([data["feature_vector"] for data in adjacency_list.values()])
    y = np.array([data["emotion"] for data in adjacency_list.values()])
    knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_model.fit(X, y)
    return knn_model

# Function to predict emotion for a new image
def predict_emotion(knn_model, image_path):
    feature_vector = extract_features(image_path)
    predicted_emotion = knn_model.predict([feature_vector])[0]
    return predicted_emotion

# Function to calculate accuracy and precision
def calculate_metrics(knn_model, adjacency_list):
    X_test = np.array([data["feature_vector"] for data in adjacency_list.values()])
    y_test = np.array([data["emotion"] for data in adjacency_list.values()])
    y_pred = knn_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    return accuracy, precision

# Function to handle file drop event
def on_drop(event):
    try:
        # Get the file paths of dropped files
        file_paths = event.data.split()

        # Predict emotion for each dropped image
        for file_path in file_paths:
            predicted_emotion = predict_emotion(knn_model, file_path.decode("utf-8"))
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to open file dialog for selecting images
def open_file_dialog():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_paths:
        for file_path in file_paths:
            predicted_emotion = predict_emotion(knn_model, file_path)
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

# Create the main window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("600x400")  # Set window size

# Create a label for instructions
instructions_label = tk.Label(root, text="Drag and drop images or click the button to select images \n (KNN with Euclidean Distance metrics)", font=("Arial", 12))
instructions_label.pack(pady=10)

# Create a button to open file dialog
open_button = tk.Button(root, text="Select Images", command=open_file_dialog, font=("Arial", 12))
open_button.pack(pady=5)

# Labels to display accuracy and precision
accuracy_label = tk.Label(root, text="Accuracy: N/A", font=("Arial", 12))
accuracy_label.pack(pady=5)
precision_label = tk.Label(root, text="Precision: N/A", font=("Arial", 12))
precision_label.pack(pady=5)

# Load and train the model
data_dir = "C:\\Users\\pro_j\\Desktop\\project\\new\\GUI\\CK+48"
num_samples_per_category = 100
adjacency_list = create_adjacency_list(data_dir, num_samples_per_category)
knn_model = train_knn_model(adjacency_list)

# Calculate accuracy and precision
accuracy, precision = calculate_metrics(knn_model, adjacency_list)

# Update accuracy and precision labels
accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
precision_label.config(text=f"Precision: {precision:.2f}")

# Run the application
root.mainloop()
