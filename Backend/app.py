from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the trained model
model = load_model(r"C:/SIH/agrovision/Backend/plant_disease_model.h5")

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
    app.run(debug=True)
