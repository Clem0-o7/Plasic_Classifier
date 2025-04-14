import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import tensorflow as tf
import cv2

def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.uint8)
    return segment_image(img)

def segment_image(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return segmented
    else:
        return image

def predict_plastic_type(image_path, model_path):
    label_map = {0: 'HDPE', 1: 'LDPE', 2: 'Other', 3: 'PET', 4: 'PP', 5: 'PS', 6: 'PVC'}
    processed_image = preprocess_image(image_path)
    input_image = np.expand_dims(processed_image, axis=0).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction[0])
    predicted_label = label_map[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        entry_image_path.delete(0, tk.END)
        entry_image_path.insert(0, file_path)

def run_prediction():
    image_path = entry_image_path.get()
    model_path = entry_model_path.get()
    if not image_path or not model_path:
        messagebox.showwarning("Input Error", "Please provide both image and model paths.")
        return
    try:
        predicted_type, confidence = predict_plastic_type(image_path, model_path)
        lbl_result.config(text=f"Predicted: {predicted_type} (Confidence: {confidence:.2%})")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict plastic type: {str(e)}")

# Create GUI window
root = tk.Tk()
root.title("Plastic Type Classifier")
root.geometry("500x300")

# Image path input
lbl_image_path = tk.Label(root, text="Image Path:")
lbl_image_path.pack(pady=5)
entry_image_path = tk.Entry(root, width=50)
entry_image_path.pack(pady=5)
btn_browse_image = tk.Button(root, text="Browse", command=open_file_dialog)
btn_browse_image.pack(pady=5)

# Model path input
lbl_model_path = tk.Label(root, text="Model Path:")
lbl_model_path.pack(pady=5)
entry_model_path = tk.Entry(root, width=50)
entry_model_path.insert(0, "D:/Projects/Plastic_tflite/Plastic/Model1/plastic_classifier.tflite")  # Default model path
entry_model_path.pack(pady=5)

# Prediction button
btn_predict = tk.Button(root, text="Predict Plastic Type", command=run_prediction)
btn_predict.pack(pady=10)

# Result display
lbl_result = tk.Label(root, text="Result will appear here", font=("Arial", 14))
lbl_result.pack(pady=20)

# Run the GUI
root.mainloop()
