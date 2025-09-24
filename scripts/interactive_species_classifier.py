import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO
from Bio import SeqIO
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    with open("../models/edna_model.pkl", "rb") as f:
        data = pickle.load(f)
        return data["model"], data["encoder"]

model, encoder = load_model_and_encoder()

# Feature extraction
def extract_features(seq):
    seq = seq.upper()
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    comp = [seq.count(base) / len(seq) for base in "ATCG"]
    return [gc] + comp

# Validate DNA sequence
def validate_sequence(seq):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    return all(base in "ATCG" for base in seq) and len(seq) >= 50

# Predict known species
def predict_known_species(seq):
    if not validate_sequence(seq):
        return {"status": "Invalid", "confidence": 0, "species": "Invalid Sequence"}
    features = extract_features(seq)
    encoded_pred = model.predict([features])[0]
    confidence = model.predict_proba([features])[0].max() * 100
    species = encoder.inverse_transform([encoded_pred])[0]
    label = "✅ Known" if confidence > 70 else "⚠️ Uncertain" if confidence > 50 else "❌ Unknown"
    return {"status": label, "confidence": round(confidence, 2), "species": species}

# Parse FASTQ
def parse_fastq(uploaded_file):
    string_data = uploaded_file.getvalue().decode("utf-8")
    text_stream = StringIO(string_data)
    sequences = []
    for record in SeqIO.parse(text_stream, "fastq"):
        sequences.append((record.id, str(record.seq)))
    return sequences

# Novel species clustering
def discover_novel_species(sequences):
    valid_sequences = [seq for seq in sequences if validate_sequence(seq)]
    features = [extract_features(seq) for seq in valid_sequences]
    clustering = DBSCAN(eps=0.05, min_samples=5).fit(features)
    labels = clustering.labels_
    novelty_scores = [round(np.mean(pairwise_distances([f], features)), 3) for f in features]
    clusters = pd.DataFrame({
        "Sequence": valid_sequences,
        "Cluster": labels,
        "Novelty Score": novelty_scores,
        "GC Content": [f[0] for f in features]
    })
    return clusters

# Biodiversity metrics
def calculate_biodiversity(df):
    species_counts = df["species"].value_counts()
    total = species_counts.sum()
    proportions = species_counts / total
    shannon = -np.sum(proportions * np.log2(proportions))
    simpson = 1 - np.sum(proportions ** 2)
    evenness = shannon / np.log2(len(species_counts)) if len(species_counts) > 1 else 0
    return {
        "Species Richness": len(species_counts),
        "Shannon Diversity": round(shannon, 3),
        "Simpson Diversity": round(simpson, 3),
        "Evenness": round(evenness, 3)
    }

# Dashboard UI
st.set_page_config(page_title="🧬 Integrated Species Classification", layout="wide")
st.title("🧬 Integrated Species Classification System")

st.markdown("Upload a FASTQ file (up to 500 MB) to run full analysis: known species prediction, novel species discovery, biodiversity metrics, and visualizations.")

fastq_file = st.file_uploader("Upload FASTQ File", type=["fastq"])
if fastq_file:
    sequences = parse_fastq(fastq_file)
    ids = [seq_id for seq_id, _ in sequences]
    raw_seqs = [seq for _, seq in sequences]

    # Known species predictions
    known_results = [predict_known_species(seq) for seq in raw_seqs]
    known_df = pd.DataFrame(known_results)
    known_df["ID"] = ids
    known_df["Sequence"] = raw_seqs

    # Novel species clustering
    novel_df = discover_novel_species(raw_seqs)

    # Biodiversity metrics
    biodiversity = calculate_biodiversity(known_df)

    # Layout
    st.subheader("🔍 Known Species Predictions")
    st.dataframe(known_df)

    st.subheader("🆕 Novel Species Clusters")
    st.dataframe(novel_df)

    with st.expander("📊 Biodiversity Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Species Richness", biodiversity["Species Richness"])
        col2.metric("Shannon Diversity", biodiversity["Shannon Diversity"])
        col3.metric("Simpson Diversity", biodiversity["Simpson Diversity"])
        col4.metric("Evenness", biodiversity["Evenness"])

    st.subheader("📈 Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Species Abundance (Bar Plot)**")
        fig1, ax1 = plt.subplots()
        known_df["species"].value_counts().plot.bar(ax=ax1, color="teal")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Species")
        ax1.set_title("Species Abundance")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Confidence Distribution (Histogram)**")
        fig2, ax2 = plt.subplots()
        sns.histplot(known_df["confidence"], bins=15, kde=True, ax=ax2, color="orange")
        ax2.set_title("Confidence Scores")
        st.pyplot(fig2)

    st.markdown("**GC Content vs Novelty Score (Scatter Plot)**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=novel_df, x="GC Content", y="Novelty Score", hue="Cluster", palette="tab10", ax=ax3)
    ax3.set_title("Cluster View: GC vs Novelty")
    st.pyplot(fig3)

    st.markdown("**Cluster Size Distribution (Box Plot)**")
    fig4, ax4 = plt.subplots()
    cluster_sizes = novel_df["Cluster"].value_counts()
    sns.boxplot(data=cluster_sizes.values, ax=ax4, color="lightblue")
    ax4.set_title("Cluster Size Distribution")
    st.pyplot(fig4)