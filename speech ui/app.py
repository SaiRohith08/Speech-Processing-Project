import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from scipy.stats import skew, kurtosis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# Parameters
SAMPLE_RATE = 16000
N_MFCC = 20
TARGET_SHAPE = (16, 8)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_stats = [
        np.mean(zcr), np.std(zcr), np.min(zcr),
        np.max(zcr), skew(zcr), kurtosis(zcr)
    ]

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_flat = librosa.feature.spectral_flatness(y=y)[0]

    spectral_features = []
    for f in [spec_centroid, spec_bw, spec_rolloff, spec_flat]:
        spectral_features.extend([np.mean(f), np.std(f)])

    features = np.array(mfcc_mean.tolist() + mfcc_std.tolist() + delta_mean.tolist() + zcr_stats + spectral_features)
    return features

def prepare_input(features):
    features_scaled = scaler.transform([features])
    num_features = features_scaled.shape[1]
    required_features = TARGET_SHAPE[0] * TARGET_SHAPE[1]

    if num_features < required_features:
        padding_size = required_features - num_features
        padded = np.pad(features_scaled, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
    else:
        padded = features_scaled

    reshaped = padded.reshape(-1, TARGET_SHAPE[0], TARGET_SHAPE[1], 1)
    return reshaped

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        if file and file.filename.endswith('.flac'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            try:
                features = extract_features(path)
                input_data = prepare_input(features)
                prediction = model.predict(input_data)[0]
                label = "Real" if np.argmax(prediction) == 1 else "Fake"
                confidence = float(np.max(prediction))

                return render_template('index.html', result=label, confidence=confidence, filename=file.filename)
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
