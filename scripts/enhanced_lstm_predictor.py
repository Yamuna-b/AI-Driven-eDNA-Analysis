#!/usr/bin/env python3
"""
Enhanced LSTM Predictor with Clustering for eDNA Species Classification
Works specifically with labeled_sequences.csv training data
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import argparse
import sys

# Configuration
MODEL_PATH = "../models/edna_model.pkl"
TRAINING_DATA_PATH = "../models/labeled_sequences.csv"
CLUSTER_MODEL_PATH = "../models/cluster_model.pkl"

class eDNAPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.cluster_model = None
        self.pca_model = None
        self.species_list = []
        
    def extract_features(self, sequence):
        """Extract comprehensive features from DNA sequence"""
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return np.zeros(12)  # Extended feature set
        
        # Basic nucleotide frequencies
        a_freq = seq.count('A') / length
        t_freq = seq.count('T') / length
        c_freq = seq.count('C') / length
        g_freq = seq.count('G') / length
        
        # GC content
        gc_content = c_freq + g_freq
        
        # Dinucleotide frequencies (top 7 most informative)
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
    
    def load_training_data(self):
        """Load and prepare training data from CSV"""
        print("Loading training data...")
        
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"Training data not found: {TRAINING_DATA_PATH}")
            return None, None
        
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"Loaded {len(df)} sequences with {df['species'].nunique()} species")
        
        # Extract features
        print("Extracting features...")
        features = []
        labels = []
        
        for idx, row in df.iterrows():
            try:
                seq_features = self.extract_features(row['sequence'])
                features.append(seq_features)
                labels.append(row['species'])
            except Exception as e:
                print(f"Error processing sequence {idx}: {e}")
                continue
        
        X = np.array(features)
        y = np.array(labels)
        
        self.species_list = sorted(list(set(y)))
        print(f"Species found: {self.species_list}")
        
        return X, y
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Training model...")
        
        X, y = self.load_training_data()
        if X is None:
            return False
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return True
    
    def train_clustering(self):
        """Train clustering model for species grouping"""
        print("🎯 Training clustering model...")
        
        X, y = self.load_training_data()
        if X is None:
            return False
        
        # PCA for dimensionality reduction
        self.pca_model = PCA(n_components=5)
        X_pca = self.pca_model.fit_transform(X)
        
        # K-means clustering
        n_clusters = min(8, len(self.species_list))  # Reasonable number of clusters
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(X_pca)
        
        # Analyze clusters
        cluster_species = {}
        for i in range(n_clusters):
            mask = cluster_labels == i
            species_in_cluster = [y[j] for j in range(len(y)) if mask[j]]
            cluster_species[i] = list(set(species_in_cluster))
        
        print(f"✅ Created {n_clusters} clusters:")
        for cluster_id, species in cluster_species.items():
            print(f"  Cluster {cluster_id}: {species}")
        
        return True
    
    def save_models(self):
        """Save trained models"""
        print("💾 Saving models...")
        
        model_bundle = {
            'model': self.model,
            'encoder': self.label_encoder,
            'cluster_model': self.cluster_model,
            'pca_model': self.pca_model,
            'species_list': self.species_list
        }
        
        joblib.dump(model_bundle, MODEL_PATH)
        print(f"✅ Models saved to {MODEL_PATH}")
    
    def load_models(self):
        """Load trained models"""
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
        
        try:
            model_bundle = joblib.load(MODEL_PATH)
            self.model = model_bundle.get('model')
            self.label_encoder = model_bundle.get('encoder')
            self.cluster_model = model_bundle.get('cluster_model')
            self.pca_model = model_bundle.get('pca_model')
            self.species_list = model_bundle.get('species_list', [])
            
            print(f"✅ Models loaded successfully")
            print(f"🧬 Available species: {len(self.species_list)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_species(self, sequence):
        """Predict species for a DNA sequence"""
        if self.model is None or self.label_encoder is None:
            return {"Species": "Model Error", "Confidence": 0.0, "Error": "Model not loaded"}
        
        try:
            # Extract features
            features = self.extract_features(sequence)
            feature_array = features.reshape(1, -1)
            
            # Predict
            probabilities = self.model.predict_proba(feature_array)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            
            # Get species name
            species = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # Get cluster information
            cluster_info = self.get_cluster_info(features)
            
            return {
                "Species": species,
                "Confidence": round(confidence, 4),
                "Cluster": cluster_info["cluster"],
                "Cluster_Species": cluster_info["species_in_cluster"],
                "Top_3_Predictions": self.get_top_predictions(probabilities, 3)
            }
            
        except Exception as e:
            return {"Species": "Prediction Error", "Confidence": 0.0, "Error": str(e)}
    
    def get_cluster_info(self, features):
        """Get cluster information for a sequence"""
        if self.cluster_model is None or self.pca_model is None:
            return {"cluster": "Unknown", "species_in_cluster": []}
        
        try:
            # Transform features using PCA
            features_pca = self.pca_model.transform(features.reshape(1, -1))
            
            # Predict cluster
            cluster = self.cluster_model.predict(features_pca)[0]
            
            # Get species typically found in this cluster
            # This would need to be pre-computed during training
            cluster_species = self.species_list[:3]  # Simplified for now
            
            return {
                "cluster": int(cluster),
                "species_in_cluster": cluster_species
            }
            
        except Exception as e:
            return {"cluster": "Error", "species_in_cluster": []}
    
    def get_top_predictions(self, probabilities, top_k=3):
        """Get top K predictions with confidence scores"""
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            species = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            top_predictions.append({
                "species": species,
                "confidence": round(confidence, 4)
            })
        
        return top_predictions
    
    def batch_predict(self, fasta_file):
        """Predict species for multiple sequences from FASTA file"""
        results = []
        
        try:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
            print(f"🧬 Processing {len(sequences)} sequences...")
            
            for i, record in enumerate(sequences, 1):
                seq_id = record.id
                sequence = str(record.seq)
                
                print(f"Processing {i}/{len(sequences)}: {seq_id}")
                
                # Validate sequence
                if not self.validate_sequence(sequence):
                    results.append({
                        "Sequence_ID": seq_id,
                        "Species": "Invalid",
                        "Confidence": 0.0,
                        "Error": "Invalid DNA sequence"
                    })
                    continue
                
                # Predict
                prediction = self.predict_species(sequence)
                prediction["Sequence_ID"] = seq_id
                prediction["Sequence_Length"] = len(sequence)
                results.append(prediction)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in batch prediction: {e}")
            return []
    
    def validate_sequence(self, sequence):
        """Validate DNA sequence"""
        if not sequence or len(sequence) < 20:
            return False
        return all(c.upper() in "ATCGN" for c in sequence)
    
    def visualize_clusters(self, output_file="clusters.png"):
        """Visualize species clusters"""
        if self.cluster_model is None:
            print("❌ No cluster model available")
            return
        
        print("📊 Creating cluster visualization...")
        
        X, y = self.load_training_data()
        if X is None:
            return
        
        # PCA transformation
        X_pca = self.pca_model.transform(X)
        cluster_labels = self.cluster_model.predict(X_pca)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('eDNA Species Clustering')
        plt.colorbar(scatter)
        
        # Add species labels for cluster centers
        centers = self.cluster_model.cluster_centers_
        for i, center in enumerate(centers):
            plt.annotate(f'Cluster {i}', center, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Cluster visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced LSTM Predictor for eDNA")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", "-p", help="Predict single sequence")
    parser.add_argument("--batch", "-b", help="Batch predict from FASTA file")
    parser.add_argument("--visualize", action="store_true", help="Create cluster visualization")
    parser.add_argument("--output", "-o", default="predictions.csv", help="Output file")
    
    args = parser.parse_args()
    
    predictor = eDNAPredictor()
    
    if args.train:
        print("🚀 Training enhanced eDNA predictor...")
        if predictor.train_model() and predictor.train_clustering():
            predictor.save_models()
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
            sys.exit(1)
    
    elif args.predict:
        if not predictor.load_models():
            print("❌ Could not load models. Please train first.")
            sys.exit(1)
        
        result = predictor.predict_species(args.predict)
        print("\n🧬 Prediction Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    elif args.batch:
        if not predictor.load_models():
            print("❌ Could not load models. Please train first.")
            sys.exit(1)
        
        results = predictor.batch_predict(args.batch)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"✅ Results saved to {args.output}")
        
        # Print summary
        successful = sum(1 for r in results if r.get("Species") not in ["Invalid", "Prediction Error"])
        print(f"\n📊 Summary: {successful}/{len(results)} successful predictions")
    
    elif args.visualize:
        if not predictor.load_models():
            print("❌ Could not load models. Please train first.")
            sys.exit(1)
        
        predictor.visualize_clusters()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
