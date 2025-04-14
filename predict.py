import numpy as np
import tensorflow as tf
import cv2
import os

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess a single image using the same steps as training
    """
    # Read and check image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Ensure uint8 format
    img = img.astype(np.uint8)
    
    # Apply segmentation
    return segment_image(img)

def segment_image(image):
    """
    Segment the image using the same method as training
    """
    # Ensure image is in uint8 format
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
    """
    Make a prediction using TFLite model
    """
    # Label mapping (same as training)
    label_map = {0: 'HDPE', 1: 'LDPE', 2: 'Other', 3: 'PET', 4: 'PP', 5: 'PS', 6: 'PVC'}
    
    # Load and preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Add batch dimension and convert to float32
    input_image = np.expand_dims(processed_image, axis=0).astype(np.float32)
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction results
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction[0])
    predicted_label = label_map[predicted_class]
    confidence = prediction[0][predicted_class]
    
    return predicted_label, confidence

if __name__ == "__main__":
    # Example usage
    image_path = "D:/Projects/Plastic_tflite/Plastic/image.jpg"  # Replace with your image path
    model_path = "D:/Projects/Plastic_tflite/Plastic/Model1/plastic_classifier.tflite"
    
    try:
        predicted_type, confidence = predict_plastic_type(image_path, model_path)
        print(f"Predicted plastic type: {predicted_type}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")