from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# Define custom loss function
def emd_loss(y_true, y_pred):
    y_true_normalized = tf.nn.softmax(y_true, axis=-1)
    y_pred_normalized = tf.nn.softmax(y_pred, axis=-1)
    cdf_true = tf.cumsum(y_true_normalized, axis=-1)
    cdf_pred = tf.cumsum(y_pred_normalized, axis=-1)
    emd = tf.reduce_mean(tf.reduce_sum(tf.abs(cdf_true - cdf_pred), axis=-1))
    return emd

# Initialize Flask app
app = Flask(__name__)

# Load the model at startup
print("Loading model...")
loaded_model = load_model('resnet50_emd', custom_objects={'emd_loss': emd_loss})
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Save the uploaded file temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    try:
        # Read and preprocess the image
        img = tf.io.read_file(temp_path)
        img = tf.image.decode_image(img, channels=3)
        
        # Handle images with unknown dimensions
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        
        # Convert back to uint8 for proper preprocessing
        img = tf.cast(img * 255.0, tf.uint8)
        
        # Apply ResNet50 preprocessing
        img_preprocessed = preprocess_input(img.numpy())
        
        # Add batch dimension
        img_batch = np.expand_dims(img_preprocessed, axis=0)
         
        # Make prediction
        prediction = loaded_model.predict(img_batch)[0]
        prediction = tf.nn.softmax(prediction, axis=-1)
        prediction = prediction.numpy().tolist()  # Convert to list for JSON serialization
        
        # Remove temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        # Clean up and return error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)