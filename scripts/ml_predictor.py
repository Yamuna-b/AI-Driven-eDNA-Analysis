import joblib
import os
import sys

MODEL_PATH = "../models/edna_model.pkl"

# Global variables to store model components
model_bundle = None
vocab = None
max_len = None
model = None
label_encoder = None

def load_model():
    """Load the model and handle different possible structures"""
    global model_bundle, vocab, max_len, model, label_encoder
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            return False
            
        model_bundle = joblib.load(MODEL_PATH)
        print(f"Model loaded. Available keys: {list(model_bundle.keys()) if isinstance(model_bundle, dict) else 'Not a dictionary'}")
        
        # Try different possible key names for vocabulary
        vocab_keys = ['vocab', 'vocabulary', 'char_to_idx', 'tokenizer']
        vocab = None
        for key in vocab_keys:
            if key in model_bundle:
                vocab = model_bundle[key]
                print(f"Found vocabulary with key: {key}")
                break
        
        # If no vocab found, create a default one
        if vocab is None:
            print("No vocabulary found, creating default DNA vocabulary")
            vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'N': 5}
        
        # Try different possible key names for max_len
        max_len_keys = ['max_len', 'max_length', 'sequence_length', 'maxlen']
        max_len = None
        for key in max_len_keys:
            if key in model_bundle:
                max_len = model_bundle[key]
                print(f"Found max_len with key: {key}, value: {max_len}")
                break
        
        # Default max_len if not found
        if max_len is None:
            max_len = 100  # Default sequence length
            print(f"No max_len found, using default: {max_len}")
        
        # Try to get the model
        model_keys = ['model', 'classifier', 'lstm_model']
        model = None
        for key in model_keys:
            if key in model_bundle:
                model = model_bundle[key]
                print(f"Found model with key: {key}")
                break
        
        # Try to get label encoder
        encoder_keys = ['label_encoder', 'encoder', 'classes', 'class_names']
        label_encoder = None
        for key in encoder_keys:
            if key in model_bundle:
                label_encoder = model_bundle[key]
                print(f"Found label encoder with key: {key}")
                break
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def encode_seq(seq, vocab_dict):
    """Encode DNA sequence using vocabulary"""
    return [vocab_dict.get(ch.upper(), vocab_dict.get('N', 5)) for ch in seq]

def extract_features(sequence):
    """Extract statistical features from DNA sequence for Random Forest model"""
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return [0, 0, 0, 0, 0]
    
    # Calculate nucleotide frequencies (this gives us 4 features)
    a_freq = seq.count('A') / length
    t_freq = seq.count('T') / length  
    c_freq = seq.count('C') / length
    g_freq = seq.count('G') / length
    
    # Calculate GC content (5th feature)
    gc_content = (c_freq + g_freq)
    
    return [a_freq, t_freq, c_freq, g_freq, gc_content]

def run_lstm(sequence):
    """Run ML prediction on a DNA sequence (kept name for compatibility)"""
    return run_ml_prediction(sequence)

def run_ml_prediction(sequence):
    """Run ML prediction on a DNA sequence"""
    global model_bundle, vocab, max_len, model, label_encoder
    
    # Load model if not already loaded
    if model_bundle is None:
        if not load_model():
            return {
                "Species": "Model Error",
                "Confidence": 0.0,
                "Error": "Failed to load model"
            }
    
    try:
        # Check if we have all required components
        if model is None:
            return {
                "Species": "Model Error", 
                "Confidence": 0.0,
                "Error": "Missing model components"
            }
        
        # Extract features for Random Forest model
        features = extract_features(sequence)
        
        # Convert to numpy array with proper shape
        import numpy as np
        feature_array = np.array([features])  # Shape: (1, 5)
        
        # Make prediction
        try:
            # Check if it's a neural network or traditional ML model
            if hasattr(model, 'predict_proba'):
                # Traditional ML model (like Random Forest)
                probs = model.predict_proba(feature_array)[0]
                idx = probs.argmax()
                confidence = float(probs[idx])
            elif hasattr(model, 'predict'):
                # Try neural network prediction
                try:
                    probs = model.predict(feature_array, verbose=0)[0]
                    idx = probs.argmax()
                    confidence = float(probs[idx])
                except TypeError:
                    # If verbose parameter not supported
                    probs = model.predict(feature_array)[0]
                    idx = probs.argmax()
                    confidence = float(probs[idx])
            else:
                raise Exception("Model doesn't have predict or predict_proba method")
            
            # Get species name
            if label_encoder is not None:
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
            
            return {
                "Species": species,
                "Confidence": round(confidence, 4)
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                "Species": "Prediction Error",
                "Confidence": 0.0,
                "Error": str(e)
            }
            
    except Exception as e:
        print(f"ML processing error: {str(e)}")
        return {
            "Species": "Processing Error",
            "Confidence": 0.0,
            "Error": str(e)
        }

# Initialize model on import
load_model()
