#!/usr/bin/env python3
"""
Test script for the enhanced LSTM predictor with clustering
"""

from simple_enhanced_predictor import eDNAPredictor
import pandas as pd

def test_predictor():
    """Test the enhanced predictor with sample sequences"""
    
    predictor = eDNAPredictor()
    
    # Load the trained model
    if not predictor.load_models():
        print("Error: Could not load models. Please train first.")
        return
    
    print("=== Enhanced eDNA Predictor Test ===")
    print(f"Available species: {len(predictor.species_list)}")
    print(f"Species list: {predictor.species_list[:5]}...")  # Show first 5
    
    # Test sequences from your training data
    test_sequences = [
        {
            "name": "Salmo_salar_sample",
            "sequence": "GCTCCACGCCAGCGAGCCGGGCTTCTTACCCATTTAAAGTTTGAGAATAGGTTGAGATCGTTTCGGCCCCAAGACCTCTAATCATTCGCTTTACCGGATAAAACTGCGTGGCGGGGGTGCGTCGGGTCTGCGAGAGCGCCAGCTATCCTGA",
            "expected": "Salmo_salar"
        },
        {
            "name": "Anguilla_anguilla_sample", 
            "sequence": "GTGCCAAGCTCGTCGCCTAAGTCAAATGACTTTAGATCGGCGCCGTAACTATGGCCACCAGCTCCTTTATTACCGTTCTTACGAAGAAGAACCTTGCGGTAAGCCACTGGTATTTCGCCCACATGAGGGACAAGGACACCAAGTGTCTCA",
            "expected": "Anguilla_anguilla"
        },
        {
            "name": "Pseudomonas_sp_sample",
            "sequence": "CTTTTGTATTCCCTGCCATAATCCATGGGCCTGGCACAGAGTAGGTGCTCAATAAAGTGTGTTGGTTGCATCAGCAACACTAGATTTGATCTGCTATGATTTTTCATCTTAATTTGCATGAAAATGACAGAGATGCAGACACTCACCATCG",
            "expected": "Pseudomonas_sp"
        }
    ]
    
    results = []
    
    for test in test_sequences:
        print(f"\n--- Testing {test['name']} ---")
        print(f"Expected: {test['expected']}")
        
        result = predictor.predict_species(test['sequence'])
        
        print(f"Predicted: {result['Species']}")
        print(f"Confidence: {result['Confidence']}")
        print(f"Cluster: {result['Cluster']}")
        print(f"Top 3 predictions:")
        for i, pred in enumerate(result['Top_3_Predictions'], 1):
            print(f"  {i}. {pred['species']} ({pred['confidence']})")
        
        # Check if prediction is correct
        correct = result['Species'] == test['expected']
        print(f"Correct: {'YES' if correct else 'NO'}")
        
        results.append({
            'Test': test['name'],
            'Expected': test['expected'],
            'Predicted': result['Species'],
            'Confidence': result['Confidence'],
            'Cluster': result['Cluster'],
            'Correct': correct
        })
    
    # Summary
    print("\n=== SUMMARY ===")
    df = pd.DataFrame(results)
    print(df)
    
    accuracy = sum(df['Correct']) / len(df)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Show cluster distribution
    print(f"\nCluster distribution:")
    cluster_counts = df['Cluster'].value_counts()
    print(cluster_counts)

def test_batch_prediction():
    """Test batch prediction with CSV data"""
    
    predictor = eDNAPredictor()
    
    if not predictor.load_models():
        print("Error: Could not load models.")
        return
    
    print("\n=== Batch Prediction Test ===")
    
    # Load some sequences from the training data
    df = pd.read_csv('../models/labeled_sequences.csv')
    
    # Take first 10 sequences for testing
    test_df = df.head(10)
    
    results = []
    correct_predictions = 0
    
    for idx, row in test_df.iterrows():
        sequence = row['sequence']
        true_species = row['species']
        
        prediction = predictor.predict_species(sequence)
        predicted_species = prediction['Species']
        confidence = prediction['Confidence']
        cluster = prediction['Cluster']
        
        is_correct = predicted_species == true_species
        if is_correct:
            correct_predictions += 1
        
        results.append({
            'Index': idx,
            'True_Species': true_species,
            'Predicted_Species': predicted_species,
            'Confidence': confidence,
            'Cluster': cluster,
            'Correct': is_correct
        })
        
        print(f"Seq {idx}: {true_species} -> {predicted_species} ({confidence:.3f}) [Cluster {cluster}] {'YES' if is_correct else 'NO'}")
    
    accuracy = correct_predictions / len(results)
    print(f"\nBatch Accuracy: {accuracy:.2%} ({correct_predictions}/{len(results)})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_test_results.csv', index=False)
    print("Results saved to batch_test_results.csv")

if __name__ == "__main__":
    test_predictor()
    test_batch_prediction()
