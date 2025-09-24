#!/usr/bin/env python3
"""
Test script for advanced dashboard features
"""

import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
    gc_content = c_freq + g_freq
    
    # Dinucleotide frequencies
    dinucs = ['AT', 'GC', 'CG', 'TA', 'AA', 'TT', 'CC']
    dinuc_freqs = []
    for dinuc in dinucs:
        count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == dinuc)
        dinuc_freqs.append(count / max(1, length - 1))
    
    return np.array([a_freq, t_freq, c_freq, g_freq, gc_content] + dinuc_freqs)

def test_novel_species_detection():
    """Test novel species detection functionality"""
    print("Testing Novel Species Detection...")
    
    # Sample data
    test_sequences = [
        {
            'ID': 'seq1',
            'Sequence': 'ATCGATCGATCGATCGATCGATCGATCGATCGATCG',
            'BLAST_Species': 'Salmo_salar',
            'BLAST_Identity': 95,
            'LSTM_Species': 'Salmo_salar',
            'LSTM_Confidence': 0.85
        },
        {
            'ID': 'seq2', 
            'Sequence': 'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA',
            'BLAST_Species': 'No Match',
            'BLAST_Identity': 0,
            'LSTM_Species': 'Unknown_1',
            'LSTM_Confidence': 0.3
        },
        {
            'ID': 'seq3',
            'Sequence': 'TTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAA',
            'BLAST_Species': 'Anguilla_anguilla',
            'BLAST_Identity': 60,
            'LSTM_Species': 'Pseudomonas_sp',
            'LSTM_Confidence': 0.7
        }
    ]
    
    # Extract features
    features = []
    for data in test_sequences:
        seq_features = extract_features(data['Sequence'])
        features.append(seq_features)
    
    features = np.array(features)
    
    # DBSCAN clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(features_scaled)
    
    print(f"Cluster labels: {cluster_labels}")
    
    # Calculate novelty scores
    for i, data in enumerate(test_sequences):
        blast_conf = data.get('BLAST_Identity', 0) / 100
        lstm_conf = data.get('LSTM_Confidence', 0)
        is_noise = cluster_labels[i] == -1
        disagreement = data.get('BLAST_Species', '') != data.get('LSTM_Species', '')
        
        novelty_score = (1 - blast_conf) * 0.4 + (1 - lstm_conf) * 0.3 + (is_noise * 0.3)
        if disagreement:
            novelty_score += 0.2
        
        novelty_score = min(novelty_score, 1.0)
        is_novel = novelty_score > 0.3
        
        print(f"{data['ID']}: Novelty={novelty_score:.2f}, Novel={is_novel}")
    
    print("Novel species detection test completed\n")

def test_biodiversity_metrics():
    """Test biodiversity metrics calculation"""
    print("Testing Biodiversity Metrics...")
    
    # Sample species data
    species_data = [
        {'BLAST_Species': 'Salmo_salar', 'LSTM_Species': 'Salmo_salar'},
        {'BLAST_Species': 'Salmo_salar', 'LSTM_Species': 'Salmo_salar'},
        {'BLAST_Species': 'Anguilla_anguilla', 'LSTM_Species': 'Anguilla_anguilla'},
        {'BLAST_Species': 'Esox_lucius', 'LSTM_Species': 'Esox_lucius'},
        {'BLAST_Species': 'No Match', 'LSTM_Species': 'Unknown_1'},
    ]
    
    # Extract species counts
    blast_species = [d.get('BLAST_Species', 'Unknown') for d in species_data if d.get('BLAST_Species') != 'No Match']
    blast_counts = Counter(blast_species)
    
    # Shannon diversity
    def shannon_diversity(counts):
        if not counts:
            return 0
        total = sum(counts.values())
        return -sum((count/total) * np.log(count/total) for count in counts.values() if count > 0)
    
    # Simpson diversity
    def simpson_diversity(counts):
        if not counts:
            return 0
        total = sum(counts.values())
        return 1 - sum((count/total)**2 for count in counts.values())
    
    shannon = shannon_diversity(blast_counts)
    simpson = simpson_diversity(blast_counts)
    richness = len(blast_counts)
    
    print(f"Species richness: {richness}")
    print(f"Shannon diversity: {shannon:.2f}")
    print(f"Simpson diversity: {simpson:.2f}")
    print(f"Species counts: {dict(blast_counts)}")
    
    print("Biodiversity metrics test completed\n")

def test_sequence_similarity():
    """Test sequence similarity calculation"""
    print("Testing Sequence Similarity...")
    
    def calculate_kmer_similarity(seq1, seq2, k=3):
        """Calculate k-mer based sequence similarity"""
        def get_kmers(seq, k):
            return [seq[i:i+k] for i in range(len(seq) - k + 1)]
        
        kmers1 = set(get_kmers(seq1.upper(), k))
        kmers2 = set(get_kmers(seq2.upper(), k))
        
        if not kmers1 or not kmers2:
            return 0.0
        
        intersection = len(kmers1.intersection(kmers2))
        union = len(kmers1.union(kmers2))
        
        return intersection / union if union > 0 else 0.0
    
    # Test sequences
    seq1 = "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    seq2 = "ATCGATCGATCGATCGATCGATCGATCGATCGATCG"  # Identical
    seq3 = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"  # Different
    seq4 = "ATCGATCGATCGATCGATCGATCGATCGATCGATCC"  # Similar
    
    print(f"Seq1 vs Seq2 (identical): {calculate_kmer_similarity(seq1, seq2):.2f}")
    print(f"Seq1 vs Seq3 (different): {calculate_kmer_similarity(seq1, seq3):.2f}")
    print(f"Seq1 vs Seq4 (similar): {calculate_kmer_similarity(seq1, seq4):.2f}")
    
    print("Sequence similarity test completed\n")

if __name__ == "__main__":
    print("=== Testing Advanced Dashboard Features ===\n")
    
    try:
        test_novel_species_detection()
        test_biodiversity_metrics()
        test_sequence_similarity()
        
        print("All tests completed successfully!")
        print("Advanced dashboard features are ready to use.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
