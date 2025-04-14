import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
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
        # Display image on the screen
        show_image(file_path)
        # Store the image path
        image_path.set(file_path)

def show_image(image_path):
    img = Image.open(image_path)
    img = img.resize((200, 200), Image.Resampling.LANCZOS)  # Updated line
    img_tk = ImageTk.PhotoImage(img)
    
    lbl_image.config(image=img_tk)
    lbl_image.image = img_tk


def run_prediction():
    image_path_value = image_path.get() 
    if not image_path_value:
        messagebox.showwarning("Input Error", "Please choose an image.")
        return
    try:
        predicted_type, confidence = predict_plastic_type(image_path_value, model_path)
        lbl_result.config(text=f"Predicted: {predicted_type}\nConfidence: {confidence:.2%}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict plastic type: {str(e)}")


# Create GUI window
root = tk.Tk()
root.title("Plastic Classifier")
root.geometry("320x640")
root.configure(bg="#f8f9fa")

# Style
header_font = ("Helvetica", 16, "bold")
label_font = ("Helvetica", 12)
button_font = ("Helvetica", 12, "bold")
result_font = ("Helvetica", 14)

# Header
header = tk.Label(root, text="Plastic Classifier", font=header_font, bg="#007bff", fg="white", height=2)
header.pack(fill=tk.X)

# Choose image button and display
frame_image = tk.Frame(root, bg="#f8f9fa", pady=10)
frame_image.pack(fill=tk.X)
btn_choose_image = tk.Button(frame_image, text="Choose Image", font=button_font, command=open_file_dialog, width=20)
btn_choose_image.pack(pady=5)

# Image preview display
lbl_image = tk.Label(root, bg="#f8f9fa")
lbl_image.pack(pady=20)

# Prediction button
btn_predict = tk.Button(root, text="Predict", font=button_font, bg="#28a745", fg="white", command=run_prediction)
btn_predict.pack(pady=20, ipadx=10, ipady=5)

# Result display
lbl_result = tk.Label(root, text="Result will appear here", font=result_font, bg="#f8f9fa", fg="#212529", wraplength=300, justify="center")
lbl_result.pack(pady=20, fill=tk.BOTH)

# Footer
footer = tk.Label(root, text="© 2025 Plastic Classifier", font=("Helvetica", 10), bg="#007bff", fg="white", height=1)
footer.pack(side=tk.BOTTOM, fill=tk.X)

# To store the image path
image_path = tk.StringVar()

# Model path (no longer part of the GUI but set in code)
model_path = "D:/Projects/Plastic_tflite/Plastic/Model1/plastic_classifier.tflite"

# Run the GUI
root.mainloop()
