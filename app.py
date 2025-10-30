from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import librosa
import tensorflow as tf
from flask_cors import CORS
import os
from tensorflow.keras.preprocessing import image
import pickle
import tensorflow_hub as hub


app = Flask(__name__)
CORS(app)
# Görsel model
IMG_MODEL_PATH = "bee_disease_model.h5"
IMG_CLASS_PATH = "class_indices.pkl"


# --- 🔧 Eksik fonksiyon tanımı (buraya ekle) ---
def hub_feature_extractor(x):
    """Modelde kullanılan hub tabanlı feature extractor"""
    layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1")
    return layer(x)
# ------------------------------------------------

# Custom layer tanıtımı
custom_objects = {
    'KerasLayer': hub.KerasLayer,
    'hub_feature_extractor': hub_feature_extractor
}

# Modeli yükle
image_model = tf.keras.models.load_model(IMG_MODEL_PATH, custom_objects=custom_objects)

with open(IMG_CLASS_PATH, "rb") as f:
    class_indices = pickle.load(f)

inv_class_indices = {v: k for k, v in class_indices.items()}

def prepare_image(file_path):
    img = image.load_img(file_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# Modeli yükle
MODEL_PATH = "bee_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------------
# Ses özellik çıkarma
# --------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# --------------------------
# Ses analizi
# --------------------------
@app.route('/analyze/audio', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400

    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)  # (1, 40)
        predictions = model.predict(features)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return jsonify({
            'queen_status': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --------------------------
# Görsel analizi (dummy)
# --------------------------
@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Görsel yüklenmedi'}), 400

    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        img = prepare_image(file_path)
        preds = image_model.predict(img)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        class_name = inv_class_indices[predicted_class]

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# --------------------------
# HTML sunucu
# --------------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# --------------------------
# Ana çalıştırma
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
