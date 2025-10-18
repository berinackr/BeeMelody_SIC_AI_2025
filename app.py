from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import librosa
import tensorflow as tf
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

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
    return jsonify({
        'queen_status': 0,
        'confidence': 0.95
    })

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
