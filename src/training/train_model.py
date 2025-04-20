import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import sys
from pathlib import Path

# Fix relative import issue
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.extract_features import extract_features

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "malicious_urls.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("\nClass distribution in dataset:")
    print(df['label'].value_counts())

    # Extract features
    X = df['url'].apply(extract_features).apply(pd.Series)
    y = df['label']

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        labels=range(len(encoder.classes_)),
        zero_division=0
    ))

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))

    print("\nModel training complete. Artifacts saved to models/ directory.")

if __name__ == "__main__":
    main()