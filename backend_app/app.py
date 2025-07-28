import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np

# --- Initialization ---
app = Flask(__name__)

# --- Load the Trained Model ---
# Make sure to provide the correct path to your best model file
MODEL_PATH = 'efficientnet50_model_data_aug.keras'  # Adjust this path
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Define Preprocessing and Prediction Logic ---
# These are your class names in the correct order
IMAGE_CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_HEIGHT = 224  # The height your model expects
IMG_WIDTH = 224  # The width your model expects


def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    # Load the image
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
    # Convert to numpy array
    img_array = tf.keras.utils.img_to_array(img)
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    # Apply model-specific preprocessing
    preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return preprocessed_img


# --- Define Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Render the main web page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file to a temporary location
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)  # Create 'uploads' folder if it doesn't exist
        file.save(filepath)

        # Preprocess the image and make a prediction
        try:
            processed_image = preprocess_image(filepath)
            prediction_probs = model.predict(processed_image)

            # Get the top predicted class and its confidence
            predicted_class_index = np.argmax(prediction_probs[0])
            predicted_class_name = IMAGE_CLASSES[predicted_class_index]
            confidence = np.max(prediction_probs[0]) * 100

            # Return the result as JSON
            return jsonify({
                'emotion': predicted_class_name,
                'confidence': f'{confidence:.2f}%'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {e}'}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)


# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True)