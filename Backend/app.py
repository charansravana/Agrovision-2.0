from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Initialize Flask app
app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing
CORS(app, resources={r"/*": {"origins": "*"}})  # Fully enable CORS

# Set max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# Load the trained model
# model = load_model(r"C:/Agrovision 2.0/Backend/model-2.h5")

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
        
        # Open the image file
        img = Image.open(io.BytesIO(file.read()))
        
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
    app.run(host='0.0.0.0', port=10000,debug=True)
