import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from Bio import SeqIO

# Load model and encoder
@st.cache_resource
def load_edna_model():
    model_path = "../models/edna_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found at models/edna_model.pkl")
        return None, None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        return data["model"], data["encoder"]

model, encoder = load_edna_model()

# DNA validation
def validate_sequence(seq):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    valid_bases = set("ATCG")
    return all(base in valid_bases for base in seq) and len(seq) >= 50

# Feature extraction
def extract_features(seq):
    seq = seq.upper()
    gc_content = (seq.count("G") + seq.count("C")) / len(seq)
    base_comp = [seq.count(base) / len(seq) for base in "ATCG"]
    return [gc_content] + base_comp

# Prediction logic
def predict_species(seq):
    if not validate_sequence(seq):
        return {"status": "Invalid", "confidence": 0, "species": "Invalid Sequence"}

    features = extract_features(seq)
    encoded_pred = model.predict([features])[0]
    confidence = model.predict_proba([features])[0].max() * 100
    species = encoder.inverse_transform([encoded_pred])[0]

    if confidence > 70:
        label = "✅ Known Species"
    elif confidence > 50:
        label = "⚠️ Uncertain"
    else:
        label = "❌ Unknown"

    return {
        "status": label,
        "confidence": round(confidence, 2),
        "species": species
    }

# Sequence statistics
def sequence_stats(seq):
    seq = seq.upper()
    return {
        "Length": len(seq),
        "GC Content (%)": round((seq.count("G") + seq.count("C")) / len(seq) * 100, 2),
        "A": seq.count("A"),
        "T": seq.count("T"),
        "C": seq.count("C"),
        "G": seq.count("G")
    }

# FASTA parser
def parse_fasta(file):
    sequences = []
    for record in SeqIO.parse(file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

# Dashboard UI
st.set_page_config(page_title="eDNA Species Predictor", layout="wide")
st.title("🧬 eDNA Species Predictor Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Species Prediction", "Batch Analysis", "Model Info", "Help & Docs"])

with tab1:
    st.header("🔬 Predict Species from DNA Sequence")
    input_method = st.radio("Choose Input Method", ["Manual Entry", "Upload FASTA", "Use Example"])

    if input_method == "Manual Entry":
        seq_input = st.text_area("Enter DNA Sequence (A,T,C,G only)", height=150)
        if st.button("Predict"):
            result = predict_species(seq_input)
            st.write("**Prediction:**", result["status"])
            st.write("**Species:**", result["species"])
            st.write("**Confidence:**", f"{result['confidence']}%")
            st.write("**Sequence Stats:**", sequence_stats(seq_input))

    elif input_method == "Upload FASTA":
        fasta_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa"])
        if fasta_file and st.button("Run Batch Prediction"):
            sequences = parse_fasta(fasta_file)
            results = []
            for seq_id, seq in sequences:
                res = predict_species(seq)
                res.update({"ID": seq_id, "Length": len(seq), "GC": sequence_stats(seq)["GC Content (%)"]})
                results.append(res)
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Visualizations
            st.subheader("📊 Visualizations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Species Distribution")
                st.pyplot(df["species"].value_counts().plot.pie(autopct='%1.1f%%').figure)
            with col2:
                st.write("Confidence Histogram")
                st.pyplot(sns.histplot(df["confidence"], bins=10).figure)
            with col3:
                st.write("GC Content vs Confidence")
                st.pyplot(sns.scatterplot(data=df, x="GC", y="confidence", hue="status").figure)

    else:
        st.write("Using example sequence:")
        example_seq = "ATCG" * 20
        st.code(example_seq)
        if st.button("Predict Example"):
            result = predict_species(example_seq)
            st.write("**Prediction:**", result["status"])
            st.write("**Species:**", result["species"])
            st.write("**Confidence:**", f"{result['confidence']}%")
            st.write("**Sequence Stats:**", sequence_stats(example_seq))

with tab2:
    st.header("📁 Batch Analysis")
    st.write("Upload a multi-sequence FASTA file to analyze multiple DNA sequences at once.")
    st.write("Results include species predictions, confidence scores, and visualizations.")

with tab3:
    st.header("📘 Model Information")
    st.write("Model loaded from `models/edna_model.pkl`")
    st.write("Uses GC content and base composition for species classification.")
    st.write("Confidence scores are derived from model probabilities.")
    st.write("Prediction labels:")
    st.markdown("- ✅ Known Species: Confidence > 70%")
    st.markdown("- ⚠️ Uncertain: Confidence 50–70%")
    st.markdown("- ❌ Unknown: Confidence < 50%")

with tab4:
    st.header("📖 Help & Documentation")
    st.markdown("""
    **Input Requirements:**
    - Only A, T, C, G bases
    - Minimum length: 50 bases
    - No ambiguous bases (N, R, Y, etc.)

    **Supported Formats:**
    - Plain text
    - FASTA (.fasta, .fa)

    **Troubleshooting:**
    - Ensure model file exists in `models/`
    - Check sequence formatting
    - Install dependencies via `pip install -r requirements.txt`
    """)