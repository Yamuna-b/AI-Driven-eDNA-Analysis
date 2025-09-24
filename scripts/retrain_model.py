#!/usr/bin/env python3
"""
Retrain the ML model with BLAST database species
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_features(sequence):
    """Extract statistical features from DNA sequence"""
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return [0, 0, 0, 0, 0]
    
    # Calculate nucleotide frequencies
    a_freq = seq.count('A') / length
    t_freq = seq.count('T') / length  
    c_freq = seq.count('C') / length
    g_freq = seq.count('G') / length
    
    # Calculate GC content
    gc_content = (c_freq + g_freq)
    
    return [a_freq, t_freq, c_freq, g_freq, gc_content]

def create_training_data():
    """Create training data using BLAST database sequences"""
    
    # Get sequences from BLAST database
    import subprocess
    result = subprocess.run([
        "blastdbcmd", "-db", "../blast_db/blast_db", "-entry", "all"
    ], capture_output=True, text=True, cwd=".")
    
    sequences = []
    species = []
    current_species = None
    current_seq = ""
    
    for line in result.stdout.split('\n'):
        if line.startswith('>'):
            if current_species and current_seq:
                sequences.append(current_seq)
                species.append(current_species)
            current_species = line[1:].strip()
            current_seq = ""
        else:
            current_seq += line.strip()
    
    # Add the last sequence
    if current_species and current_seq:
        sequences.append(current_seq)
        species.append(current_species)
    
    # Create variations of each sequence to have more training data
    expanded_sequences = []
    expanded_species = []
    
    for seq, sp in zip(sequences, species):
        # Add original sequence
        expanded_sequences.append(seq)
        expanded_species.append(sp)
        
        # Add variations (simulate mutations)
        for i in range(5):  # Create 5 variations per sequence
            if len(seq) > 10:
                # Create slight variations
                variation = list(seq)
                # Change 1-2 random nucleotides
                import random
                for _ in range(random.randint(1, 2)):
                    pos = random.randint(0, len(variation)-1)
                    variation[pos] = random.choice(['A', 'T', 'C', 'G'])
                expanded_sequences.append(''.join(variation))
                expanded_species.append(sp)
    
    return expanded_sequences, expanded_species

def train_model():
    """Train a new Random Forest model"""
    print("Creating training data...")
    sequences, species = create_training_data()
    
    print(f"Training data: {len(sequences)} sequences, {len(set(species))} species")
    print(f"Species: {set(species)}")
    
    # Extract features
    print("Extracting features...")
    features = [extract_features(seq) for seq in sequences]
    X = np.array(features)
    
    # Encode species labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(species)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model
    model_bundle = {
        'model': model,
        'encoder': label_encoder
    }
    
    joblib.dump(model_bundle, '../models/edna_model_retrained.pkl')
    print("Model saved as edna_model_retrained.pkl")
    
    return model, label_encoder

if __name__ == "__main__":
    train_model()
