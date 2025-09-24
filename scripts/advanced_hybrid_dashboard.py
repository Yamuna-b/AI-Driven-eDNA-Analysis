#!/usr/bin/env python3
"""
Advanced Hybrid eDNA Species Predictor Dashboard
Includes novel species detection, biodiversity metrics, and phylogenetic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from io import StringIO
import json
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# Import your predictors
from blast_batch_predictor import run_blast
from lstm_predictor import run_lstm

# Configure page
st.set_page_config(
    page_title="Advanced eDNA Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .novel-species {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def validate_sequence(seq):
    """Validate DNA sequence"""
    return all(c.upper() in "ATCGN" for c in seq) and len(seq) >= 20

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

def detect_novel_species(sequences_data, novelty_threshold=0.3):
    """Detect novel species using DBSCAN clustering"""
    if len(sequences_data) < 2:
        return sequences_data, []
    
    # Extract features for clustering
    features = []
    for data in sequences_data:
        seq_features = extract_features(data['Sequence'])
        features.append(seq_features)
    
    features = np.array(features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(features_scaled)
    
    # Calculate novelty scores
    novelty_scores = []
    for i, data in enumerate(sequences_data):
        # Base novelty on BLAST confidence and cluster assignment
        blast_conf = data.get('BLAST_Identity', 0) / 100 if data.get('BLAST_Identity') else 0
        lstm_conf = data.get('LSTM_Confidence', 0)
        
        # Novel if: low BLAST identity, assigned to noise cluster, or disagreement
        is_noise = cluster_labels[i] == -1
        low_blast = blast_conf < 0.8
        disagreement = data.get('BLAST_Species', '') != data.get('LSTM_Species', '')
        
        novelty_score = (1 - blast_conf) * 0.4 + (1 - lstm_conf) * 0.3 + (is_noise * 0.3)
        if disagreement:
            novelty_score += 0.2
        
        novelty_scores.append(min(novelty_score, 1.0))
        data['Novelty_Score'] = novelty_score
        data['Cluster'] = cluster_labels[i]
        data['Is_Novel'] = novelty_score > novelty_threshold
    
    novel_species = [data for data in sequences_data if data['Is_Novel']]
    
    return sequences_data, novel_species

def calculate_biodiversity_metrics(species_data):
    """Calculate comprehensive biodiversity metrics"""
    if not species_data:
        return {}
    
    # Extract species counts
    blast_species = [d.get('BLAST_Species', 'Unknown') for d in species_data if d.get('BLAST_Species') != 'No Match']
    lstm_species = [d.get('LSTM_Species', 'Unknown') for d in species_data]
    
    blast_counts = Counter(blast_species)
    lstm_counts = Counter(lstm_species)
    
    # Basic metrics
    total_sequences = len(species_data)
    blast_richness = len(blast_counts)
    lstm_richness = len(lstm_counts)
    novel_count = sum(1 for d in species_data if d.get('Is_Novel', False))
    
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
    
    # Evenness (Pielou's evenness)
    def evenness(counts):
        if not counts or len(counts) <= 1:
            return 0
        shannon = shannon_diversity(counts)
        return shannon / np.log(len(counts))
    
    blast_shannon = shannon_diversity(blast_counts)
    lstm_shannon = shannon_diversity(lstm_counts)
    blast_simpson = simpson_diversity(blast_counts)
    lstm_simpson = simpson_diversity(lstm_counts)
    blast_evenness = evenness(blast_counts)
    lstm_evenness = evenness(lstm_counts)
    
    return {
        'total_sequences': total_sequences,
        'blast_richness': blast_richness,
        'lstm_richness': lstm_richness,
        'novel_species_count': novel_count,
        'blast_shannon': blast_shannon,
        'lstm_shannon': lstm_shannon,
        'blast_simpson': blast_simpson,
        'lstm_simpson': lstm_simpson,
        'blast_evenness': blast_evenness,
        'lstm_evenness': lstm_evenness,
        'blast_counts': blast_counts,
        'lstm_counts': lstm_counts
    }

def create_phylogenetic_tree(sequences_data):
    """Create phylogenetic tree visualization"""
    if len(sequences_data) < 3:
        return None
    
    # Calculate pairwise distances
    sequences = [d['Sequence'] for d in sequences_data]
    species_names = [f"{d.get('BLAST_Species', 'Unknown')}_{i}" for i, d in enumerate(sequences_data)]
    
    # Calculate similarity matrix
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            sim = calculate_kmer_similarity(sequences[i], sequences[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    
    # Convert to distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # Hierarchical clustering
    condensed_distances = pdist(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    return linkage_matrix, species_names

# Main Dashboard
st.markdown('<h1 class="main-header"> Advanced eDNA Species Predictor</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header(" Analysis Options")

# Single sequence analysis
st.sidebar.subheader("Single Sequence Analysis")
manual_seq = st.sidebar.text_area("Enter DNA Sequence (ATCG only)", height=100)

if manual_seq and validate_sequence(manual_seq):
    st.subheader(" Sequence Analysis")
    
    # Basic stats
    gc_content = round((manual_seq.count('G') + manual_seq.count('C')) / len(manual_seq) * 100, 2)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Length", len(manual_seq))
    with col2:
        st.metric("GC Content", f"{gc_content}%")
    with col3:
        st.metric("AT Content", f"{100-gc_content}%")
    
    # Predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  BLAST Prediction")
        blast_result = run_blast(manual_seq)
        if blast_result:
            st.json(blast_result)
            blast_species = blast_result.get("Species", "No Match")
            blast_identity = blast_result.get("Identity", 0)
        else:
            st.json({"Result": "No Match"})
            blast_species = "No Match"
            blast_identity = 0
    
    with col2:
        st.markdown("### AI Driven Prediction")
        lstm_result = run_lstm(manual_seq)
        st.json(lstm_result)
        lstm_species = lstm_result.get("Species", "Unknown")
        lstm_confidence = lstm_result.get("Confidence", 0)
    
    # Novelty assessment
    single_data = [{
        'ID': 'manual_input',
        'Sequence': manual_seq,
        'BLAST_Species': blast_species,
        'BLAST_Identity': blast_identity,
        'AI_Driven_Species': lstm_species,
        'AI_Driven_Confidence': lstm_confidence
    }]
    
    analyzed_data, novel_species = detect_novel_species(single_data)
    
    if analyzed_data[0].get('Is_Novel', False):
        st.markdown("""
        <div class="novel-species">
        🆕 <strong>Novel Species Detected!</strong><br>
        Novelty Score: {:.2f}<br>
        This sequence shows characteristics of a potentially novel species.
        </div>
        """.format(analyzed_data[0]['Novelty_Score']), unsafe_allow_html=True)

# File upload analysis
st.sidebar.subheader("Batch Analysis")
uploaded_file = st.sidebar.file_uploader("Upload Sequence File", type=["fasta", "fastq", "fa", "fq"])

if uploaded_file:
    # Determine file format based on extension
    file_name = uploaded_file.name.lower()
    if file_name.endswith(('.fastq', '.fq')):
        file_format = "fastq"
    else:
        file_format = "fasta"
    
    # Process uploaded file
    try:
        sequences = list(SeqIO.parse(StringIO(uploaded_file.getvalue().decode("utf-8")), file_format))
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        st.info("Please ensure your file is in valid FASTA or FASTQ format.")
        sequences = []
    
    if sequences:
        st.subheader(f" Batch Analysis: {len(sequences)} sequences")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process sequences
        hybrid_results = []
        for i, rec in enumerate(sequences):
            seq = str(rec.seq)
            if validate_sequence(seq):
                status_text.text(f'Processing sequence {i+1}/{len(sequences)}: {rec.id}')
                
                blast = run_blast(seq)
                lstm = run_lstm(seq)
                
                hybrid_results.append({
                    "ID": rec.id,
                    "Sequence": seq,
                    "BLAST_Species": blast["Species"] if blast else "No Match",
                    "BLAST_Identity": blast["Identity"] if blast else None,
                    "AI_Driven_Species": lstm["Species"],
                    "AI_Driven_Confidence": lstm["Confidence"],
                    "Agreement": blast and blast["Species"] == lstm["Species"]
                })
            
            progress_bar.progress((i + 1) / len(sequences))
        
        progress_bar.empty()
        status_text.empty()
        
        if hybrid_results:
            # Novel species detection
            analyzed_results, novel_species = detect_novel_species(hybrid_results)
            df = pd.DataFrame(analyzed_results)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                " Overview", 
                " Novel Species", 
                " Biodiversity", 
                " Phylogenetic", 
                " Advanced Metrics"
            ])
            
            with tab1:
                st.subheader(" Prediction Overview")
                
                # Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Sequences", len(df))
                with col2:
                    st.metric("Agreement Rate", f"{df['Agreement'].mean():.1%}")
                with col3:
                    st.metric("Novel Species", len(novel_species))
                with col4:
                    avg_novelty = df['Novelty_Score'].mean()
                    st.metric("Avg Novelty Score", f"{avg_novelty:.2f}")
                
                # Results table
                st.dataframe(df[['ID', 'BLAST_Species', 'AI_Driven_Species', 'Agreement', 'Novelty_Score', 'Is_Novel']])
                
                # Agreement visualization
                agreement_counts = df['Agreement'].value_counts()
                if len(agreement_counts) > 0:
                    # Ensure we have both agreement and disagreement labels
                    agreement_labels = []
                    agreement_values = []
                    
                    if True in agreement_counts.index:
                        agreement_labels.append('Agreement')
                        agreement_values.append(agreement_counts[True])
                    
                    if False in agreement_counts.index:
                        agreement_labels.append('Disagreement')
                        agreement_values.append(agreement_counts[False])
                    
                    if len(agreement_labels) > 0:
                        fig_agreement = px.pie(
                            values=agreement_values,
                            names=agreement_labels,
                            title="BLAST vs AI Driven Agreement"
                        )
                        st.plotly_chart(fig_agreement)
                    else:
                        st.info("No agreement data available for visualization.")
            
            with tab2:
                st.subheader(" Novel Species Detection")
                
                if novel_species:
                    st.success(f"Detected {len(novel_species)} potentially novel species!")
                    
                    # Novel species details
                    novel_df = pd.DataFrame(novel_species)
                    st.dataframe(novel_df[['ID', 'BLAST_Species', 'AI_Driven_Species', 'Novelty_Score', 'Cluster']])
                    
                    # Novelty score distribution
                    fig_novelty = px.histogram(
                        df, x='Novelty_Score', nbins=20,
                        title="Novelty Score Distribution",
                        labels={'Novelty_Score': 'Novelty Score', 'count': 'Number of Sequences'}
                    )
                    fig_novelty.add_vline(x=0.3, line_dash="dash", line_color="red", 
                                         annotation_text="Novelty Threshold")
                    st.plotly_chart(fig_novelty)
                    
                    # LSTM-based DBSCAN clustering visualization
                    if len(df) > 2:
                        features = np.array([extract_features(seq) for seq in df['Sequence']])
                        pca = PCA(n_components=2)
                        features_2d = pca.fit_transform(features)
                        
                        # Create clustering visualization focused on LSTM predictions
                        fig_cluster = px.scatter(
                            x=features_2d[:, 0], y=features_2d[:, 1],
                            color=df['Cluster'].astype(str),
                            hover_data=[df['ID'], df['AI_Driven_Species'], df['AI_Driven_Confidence']],
                            title="AI Driven-based DBSCAN Clustering (PCA Projection)",
                            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
                        )
                        fig_cluster.update_traces(marker=dict(size=10, opacity=0.7))
                        st.plotly_chart(fig_cluster)
                        
                        # LSTM species clustering analysis
                        st.subheader("AI Driven Species Cluster Analysis")
                        cluster_analysis = df.groupby('Cluster')['AI_Driven_Species'].apply(list).to_dict()
                        
                        for cluster_id, species_list in cluster_analysis.items():
                            if cluster_id != -1:  # Skip noise points
                                unique_species = list(set(species_list))
                                st.write(f"**Cluster {cluster_id}**: {', '.join(unique_species)}")
                        
                        if -1 in cluster_analysis:
                            noise_species = list(set(cluster_analysis[-1]))
                            st.write(f"**Noise Points (Potential Novel Species)**: {', '.join(noise_species)}")
                
                else:
                    st.info("No novel species detected in this dataset.")
            
            with tab3:
                st.subheader(" Biodiversity Metrics")
                
                # Calculate biodiversity metrics
                bio_metrics = calculate_biodiversity_metrics(analyzed_results)
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Basic Metrics")
                    st.metric("Species Richness (BLAST)", bio_metrics['blast_richness'])
                    st.metric("Species Richness (AI Driven)", bio_metrics['lstm_richness'])
                    st.metric("Novel Species", bio_metrics['novel_species_count'])
                
                with col2:
                    st.markdown("#### Diversity Indices")
                    st.metric("Shannon Diversity (BLAST)", f"{bio_metrics['blast_shannon']:.2f}")
                    st.metric("Simpson Diversity (BLAST)", f"{bio_metrics['blast_simpson']:.2f}")
                    st.metric("Evenness (BLAST)", f"{bio_metrics['blast_evenness']:.2f}")
                
                # Species abundance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if bio_metrics['blast_counts']:
                        fig_blast_abundance = px.bar(
                            x=list(bio_metrics['blast_counts'].keys()),
                            y=list(bio_metrics['blast_counts'].values()),
                            title="BLAST Species Abundance"
                        )
                        fig_blast_abundance.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig_blast_abundance)
                
                with col2:
                    if bio_metrics['lstm_counts']:
                        fig_lstm_abundance = px.bar(
                            x=list(bio_metrics['lstm_counts'].keys()),
                            y=list(bio_metrics['lstm_counts'].values()),
                            title="AI Driven Species Abundance"
                        )
                        fig_lstm_abundance.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig_lstm_abundance)
                
                # Diversity comparison
                diversity_comparison = pd.DataFrame({
                    'Method': ['BLAST', 'AI Driven'],
                    'Shannon': [bio_metrics['blast_shannon'], bio_metrics['lstm_shannon']],
                    'Simpson': [bio_metrics['blast_simpson'], bio_metrics['lstm_simpson']],
                    'Evenness': [bio_metrics['blast_evenness'], bio_metrics['lstm_evenness']]
                })
                
                fig_diversity = px.bar(
                    diversity_comparison.melt(id_vars='Method'),
                    x='variable', y='value', color='Method',
                    title="Diversity Metrics Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig_diversity)
            
            with tab4:
                st.subheader(" Phylogenetic Analysis")
                
                if len(analyzed_results) >= 3:
                    # Sequence similarity heatmap based on LSTM predictions
                    sequences = [d['Sequence'] for d in analyzed_results]
                    species_names = [d.get('AI_Driven_Species', 'Unknown') for d in analyzed_results]
                    
                    n = len(sequences)
                    similarity_matrix = np.zeros((n, n))
                    
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                sim = calculate_kmer_similarity(sequences[i], sequences[j])
                                similarity_matrix[i][j] = sim
                            else:
                                similarity_matrix[i][j] = 1.0
                    
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        similarity_matrix,
                        labels=dict(x="AI Driven Species", y="AI Driven Species", color="Similarity"),
                        x=species_names,
                        y=species_names,
                        title="AI Driven Species Similarity Matrix (K-mer based)"
                    )
                    st.plotly_chart(fig_heatmap)
                    
                    # LSTM-based family relationship network
                    if len(set(species_names)) > 1:
                        G = nx.Graph()
                        
                        # Add nodes
                        for species in set(species_names):
                            G.add_node(species)
                        
                        # Add edges for similar species (similarity > 0.3)
                        for i in range(n):
                            for j in range(i+1, n):
                                if similarity_matrix[i][j] > 0.3:
                                    G.add_edge(species_names[i], species_names[j], 
                                             weight=similarity_matrix[i][j])
                        
                        # Create network layout
                        pos = nx.spring_layout(G)
                        
                        # Extract edges
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        # Extract nodes
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        node_text = list(G.nodes())
                        
                        # Create network plot
                        fig_network = go.Figure()
                        
                        fig_network.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines'
                        ))
                        
                        fig_network.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="middle center",
                            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black'))
                        ))
                        
                        fig_network.update_layout(
                            title="AI Driven Species Relationship Network",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Connections show sequence similarity > 30%",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color='gray', size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        
                        st.plotly_chart(fig_network)
                
                else:
                    st.info("Need at least 3 sequences for phylogenetic analysis.")
            
            with tab5:
                st.subheader(" Advanced Metrics")
                
                # Confidence distribution
                fig_conf = px.histogram(
                    df, x='AI_Driven_Confidence', nbins=20,
                    title="AI Driven Confidence Distribution"
                )
                st.plotly_chart(fig_conf)
                
                # Identity vs Confidence scatter
                valid_data = df[df['BLAST_Identity'].notna()]
                if not valid_data.empty:
                    fig_scatter = px.scatter(
                        valid_data, x='BLAST_Identity', y='AI_Driven_Confidence',
                        color='Agreement', hover_data=['ID', 'AI_Driven_Species'],
                        title="BLAST Identity vs AI Driven Confidence"
                    )
                    st.plotly_chart(fig_scatter)
                
                # Feature correlation heatmap
                features_matrix = np.array([extract_features(seq) for seq in df['Sequence']])
                feature_names = ['A_freq', 'T_freq', 'C_freq', 'G_freq', 'GC_content', 
                               'AT_dinuc', 'GC_dinuc', 'CG_dinuc', 'TA_dinuc', 
                               'AA_dinuc', 'TT_dinuc', 'CC_dinuc']
                
                corr_matrix = np.corrcoef(features_matrix.T)
                
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=feature_names,
                    y=feature_names,
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig_corr)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
 Advanced eDNA Species Predictor | Powered by BLAST + AI Driven + Machine Learning<br>
Features: Novel Species Detection • Biodiversity Analysis • Phylogenetic Relationships
</div>
""", unsafe_allow_html=True)
