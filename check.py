import tensorflow as tf

# Check if TensorFlow Lite is accessible
try:
    print("TensorFlow version:", tf.__version__)
    lite = tf.lite
    print("TensorFlow Lite is available.")
except AttributeError as e:
    print("TensorFlow Lite is not accessible:", e)
