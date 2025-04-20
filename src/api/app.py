from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys
from pathlib import Path

# Get the absolute path to the project root
def get_project_root():
    current = Path(__file__).resolve()
    # Go up until we find the project root (contains src/ and models/)
    while not (current / 'src').exists() and not (current / 'models').exists() and current.parent != current:
        current = current.parent
    return current

# Set up paths
project_root = get_project_root()
sys.path.append(str(project_root))

# Now import using absolute path
try:
    from src.features.extract_features import extract_features
except ImportError:
    raise ImportError(
        "Could not import extract_features. Please ensure: \n"
        "1. Your project structure is correct\n"
        "2. You're running from the project root\n"
        "3. All required packages are installed"
    )

MODEL_DIR = project_root / 'models'

# Verify model files exist
required_files = {
    'model': MODEL_DIR / 'model.pkl',
    'scaler': MODEL_DIR / 'scaler.pkl',
    'encoder': MODEL_DIR / 'encoder.pkl'
}

missing_files = [name for name, path in required_files.items() if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        f"Missing model files: {', '.join(missing_files)}. \n"
        "Please train the model first by running:\n"
        "python -m src.training.train_model"
    )

# Load artifacts
model = joblib.load(required_files['model'])
scaler = joblib.load(required_files['scaler'])
encoder = joblib.load(required_files['encoder'])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    
    try:
        # Extract features
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        
        # Scale and predict
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)
        attack_type = encoder.inverse_transform(prediction)[0]
        confidence = float(model.predict_proba(scaled_features).max())
        
        return jsonify({
            "url": url,
            "is_malicious": attack_type != "benign",
            "attack_type": attack_type,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)