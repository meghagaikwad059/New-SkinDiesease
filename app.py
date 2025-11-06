from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Flask
app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "skin_disease_model.h5")
model = load_model(MODEL_PATH)

# Class index mapping
short_label_map = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

full_label_map = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# ✅ Manual disease descriptions
disease_descriptions = {
    'Actinic keratoses and intraepithelial carcinoma': (
        "A rough, scaly patch on the skin caused by years of sun exposure. "
        "It can sometimes progress to skin cancer if untreated."
    ),
    'Basal cell carcinoma': (
        "A common form of skin cancer that arises from sun exposure. "
        "It grows slowly and rarely spreads but needs treatment."
    ),
    'Benign keratosis-like lesions': (
        "Non-cancerous skin growths that may appear as warty or waxy spots. "
        "Usually harmless and common with aging."
    ),
    'Dermatofibroma': (
        "A benign skin nodule, often firm and raised. "
        "Typically develops after minor skin trauma."
    ),
    'Melanoma': (
        "A serious type of skin cancer that develops from pigment-producing cells. "
        "Early detection is critical for successful treatment."
    ),
    'Melanocytic nevi': (
        "Commonly known as moles, these are usually benign. "
        "Changes in size, shape, or color may need medical attention."
    ),
    'Vascular lesions': (
        "Abnormal clusters of blood vessels visible on the skin. "
        "Often benign and can include conditions like hemangiomas."
    )
}

def preprocess_image(file_storage):
    img = cv2.imdecode(np.frombuffer(file_storage.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def get_disease_description(disease_name):
    return disease_descriptions.get(disease_name, "No detailed description available at the moment.")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image = preprocess_image(image_file)

        preds = model.predict(image)
        top_index = np.argmax(preds)
        short_label = short_label_map.get(top_index, "unknown")
        full_label = full_label_map.get(short_label, "Unknown Disease")
        confidence = float(preds[0][top_index]) * 100

        # Use local description
        info = get_disease_description(full_label)

        return jsonify({
            'label': full_label,
            'score': round(confidence, 2),
            'info': info,
            'status': 200
        })

    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
