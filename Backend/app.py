from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import io
import os

# Disable GPU usage for TensorFlow (optional, for CPU-based inference)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app, resources={r"/*": {"origins": "*"}})  # Fully enable CORS

# Set max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model-2.h5")
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return "Hello, Flask is running successfully!"

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({"error": "Empty file uploaded"}), 400

        try:
            # Open the image file
            img = Image.open(file)
        except Exception:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocess the image (resize & normalize)
        img = img.resize((224, 224))  # Adjust based on your model's input size
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img).tolist()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
