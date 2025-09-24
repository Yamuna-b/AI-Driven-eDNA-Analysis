import joblib
import os
import numpy as np

# Path to your trained model
MODEL_PATH = "../models/edna_model_enhanced.pkl"

# Global components
model_bundle = None
vocab = None
max_len = None
model = None
label_encoder = None

def load_model():
    """Load model and extract components"""
    global model_bundle, vocab, max_len, model, label_encoder

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return False

    try:
        model_bundle = joblib.load(MODEL_PATH)
        print(f"Model loaded. Keys: {list(model_bundle.keys())}")

        # Vocabulary
        for key in ['vocab', 'vocabulary', 'char_to_idx', 'tokenizer']:
            if key in model_bundle:
                vocab = model_bundle[key]
                print(f"Found vocabulary: {key}")
                break
        if vocab is None:
            vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'N': 5}
            print("No vocab found, using default")

        # Max sequence length
        for key in ['max_len', 'max_length', 'sequence_length', 'maxlen']:
            if key in model_bundle:
                max_len = model_bundle[key]
                print(f"Found max_len: {key} = {max_len}")
                break
        if max_len is None:
            max_len = 100
            print("No max_len found, using default = 100")

        # Model
        for key in ['model', 'classifier', 'lstm_model']:
            if key in model_bundle:
                model = model_bundle[key]
                print(f"Found model: {key}")
                break

        # Label encoder
        for key in ['label_encoder', 'encoder', 'classes', 'class_names']:
            if key in model_bundle:
                label_encoder = model_bundle[key]
                print(f"Found label encoder: {key}")
                break

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def encode_seq(seq, vocab_dict):
    """Convert DNA sequence to integer list"""
    return [vocab_dict.get(ch.upper(), vocab_dict.get('N', 5)) for ch in seq]

def extract_features(sequence):
    """Extract comprehensive features from DNA sequence"""
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return np.zeros(12)
    
    # Basic nucleotide frequencies
    a_freq = seq.count('A') / length
    t_freq = seq.count('T') / length
    c_freq = seq.count('C') / length
    g_freq = seq.count('G') / length
    
    # GC content
    gc_content = c_freq + g_freq
    
    # Dinucleotide frequencies
    dinucs = ['AT', 'GC', 'CG', 'TA', 'AA', 'TT', 'CC']
    dinuc_freqs = []
    for dinuc in dinucs:
        count = 0
        for i in range(len(seq) - 1):
            if seq[i:i+2] == dinuc:
                count += 1
        dinuc_freqs.append(count / max(1, length - 1))
    
    # Combine all features
    features = [a_freq, t_freq, c_freq, g_freq, gc_content] + dinuc_freqs
    return np.array(features)

def run_lstm(sequence):
    """Run prediction on a DNA sequence"""
    global model_bundle, vocab, max_len, model, label_encoder

    if model_bundle is None:
        if not load_model():
            return {"Species": "Model Error", "Confidence": 0.0, "Error": "Failed to load model"}

    try:
        if vocab is None or model is None:
            return {"Species": "Model Error", "Confidence": 0.0, "Error": "Missing model components"}

        # Feature extraction
        features = extract_features(sequence)
        feature_array = np.array([features])  # Shape: (1, 5)

        # Prediction
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(feature_array)[0]
        else:
            try:
                probs = model.predict(feature_array, verbose=0)[0]
            except TypeError:
                probs = model.predict(feature_array)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # Decode species name
        if label_encoder:
            if hasattr(label_encoder, 'inverse_transform'):
                species = str(label_encoder.inverse_transform([idx])[0])
            elif hasattr(label_encoder, 'classes_'):
                species = str(label_encoder.classes_[idx])
            elif isinstance(label_encoder, (list, tuple)):
                species = str(label_encoder[idx]) if idx < len(label_encoder) else f"Species_{idx}"
            else:
                species = f"Species_{idx}"
        else:
            species = f"Species_{idx}"

        return {"Species": species, "Confidence": round(confidence, 4)}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Species": "Prediction Error", "Confidence": 0.0, "Error": str(e)}

# Optional: Load model on import
load_model()