# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from flask_cors import CORS
# from PIL import Image
# import io
# import os

# # Disable GPU usage for TensorFlow (optional, for CPU-based inference)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Initialize Flask app
# app = Flask(__name__)

# # Enable Cross-Origin Resource Sharing (CORS)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Fully enable CORS

# # Set max upload size to 16MB
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# # Load the trained model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "model-2.h5")
# model = load_model(MODEL_PATH)

# @app.route('/')
# def home():
#     return "Hello, Flask is running successfully!"

# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     try:
#         # Check if an image file was uploaded
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files['file']

#         # Check if the file is empty
#         if file.filename == '':
#             return jsonify({"error": "Empty file uploaded"}), 400

#         try:
#             # Open the image file
#             img = Image.open(file)
#         except Exception:
#             return jsonify({"error": "Invalid image format"}), 400

#         # Preprocess the image (resize & normalize)
#         img = img.resize((224, 224))  # Adjust based on your model's input size
#         img = np.array(img) / 255.0   # Normalize pixel values
#         img = np.expand_dims(img, axis=0)  # Add batch dimension

#         # Make prediction
#         prediction = model.predict(img).tolist()

#         return jsonify({"prediction": prediction})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)





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

# Enable CORS with specific settings
CORS(app, resources={r"/predict": {"origins": "*"}})

# Set max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the trained model only once
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model-2.h5")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Ensure the app doesn't crash

@app.route('/')
def home():
    return "✅ Flask server is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the server logs."}), 500

    try:
        # Validate if a file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Validate file name
        if file.filename == '':
            return jsonify({"error": "Empty file uploaded"}), 400

        # Validate image format
        try:
            img = Image.open(file).convert("RGB")  # Convert to RGB (handle grayscale)
        except Exception:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocess image (resize & normalize)
        img = img.resize((224, 224))  # Adjust to match model's expected input
        img = np.array(img) / 255.0   # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run prediction
        prediction = model.predict(img).tolist()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
