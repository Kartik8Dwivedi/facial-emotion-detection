import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from sklearn.preprocessing import LabelEncoder

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data(data_dir, num_samples_per_category):
    X = []
    y = []
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
                    image = imread(image_path, as_gray=True)
                    resized_image = resize(image, (100, 100))
                    resized_image = np.expand_dims(resized_image, axis=-1)  
                    X.append(resized_image)
                    y.append(label_encoder.transform([emotion])[0])
    X = np.array(X)
    y = np.array(y)
    return X, y, label_encoder

def train_cnn_model(X_train, y_train, input_shape, num_classes):
    model = create_cnn_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

def predict_emotion(cnn_model, image):
    resized_image = resize(image, (100, 100))
    resized_image = np.expand_dims(resized_image, axis=-1) 
    resized_image = np.expand_dims(resized_image, axis=0)  
    prediction = cnn_model.predict(resized_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

def on_drop(event):
    try:
        file_paths = event.data.split()

        for file_path in file_paths:
            image = imread(file_path.decode("utf-8"), as_gray=True)
            predicted_class = predict_emotion(cnn_model, image)
            predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def open_file_dialog():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_paths:
        for file_path in file_paths:
            image = imread(file_path, as_gray=True)
            predicted_class = predict_emotion(cnn_model, image)
            predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
            messagebox.showinfo("Emotion Prediction", f"The emotion displayed in the image is: {predicted_emotion}")

root = tk.Tk()
root.title("Emotion Detection")
root.geometry("600x400")  

instructions_label = tk.Label(root, text="Drag and drop images or click the button to select images \n (CNN with Euclidean Distance metrics)", font=("Arial", 12))
instructions_label.pack(pady=10)

open_button = tk.Button(root, text="Select Images", command=open_file_dialog, font=("Arial", 12))
open_button.pack(pady=5)

data_dir = "C:\\Users\\pro_j\\Desktop\\project\\new\\GUI\\CK+48"
num_samples_per_category = 75
X, y, label_encoder = prepare_data(data_dir, num_samples_per_category)
input_shape = X.shape[1:]
num_classes = len(np.unique(y))

cnn_model = train_cnn_model(X, y, input_shape, num_classes)

root.mainloop()
