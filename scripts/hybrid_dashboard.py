import streamlit as st
import pandas as pd
from Bio import SeqIO
from io import StringIO
from blast_batch_predictor import run_blast
from lstm_predictor import run_lstm

def validate_sequence(seq):
    return all(c in "ATCG" for c in seq.upper()) and len(seq) >= 20

st.set_page_config("Hybrid eDNA Predictor", layout="wide")
st.title("🧬 Hybrid eDNA Species Predictor (BLAST + LSTM)")

manual_seq = st.sidebar.text_area("Enter DNA Sequence", height=100)
if manual_seq and validate_sequence(manual_seq):
    st.subheader("Sequence Stats")
    st.write(f"Length: {len(manual_seq)} | GC%: {round((manual_seq.count('G') + manual_seq.count('C')) / len(manual_seq) * 100, 2)}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 BLAST Prediction")
        blast_result = run_blast(manual_seq)
        st.json(blast_result if blast_result else {"Result": "No Match"})

    with col2:
        st.markdown("### 🧠 LSTM Prediction")
        lstm_result = run_lstm(manual_seq)
        st.json(lstm_result)

uploaded_file = st.sidebar.file_uploader("Upload FASTA File", type=["fasta"])
if uploaded_file:
    sequences = list(SeqIO.parse(StringIO(uploaded_file.getvalue().decode("utf-8")), "fasta"))
    hybrid_results = []
    for rec in sequences:
        seq = str(rec.seq)
        if validate_sequence(seq):
            blast = run_blast(seq)
            lstm = run_lstm(seq)
            hybrid_results.append({
                "ID": rec.id,
                "Sequence": seq,
                "BLAST_Species": blast["Species"] if blast else "No Match",
                "BLAST_Identity": blast["Identity"] if blast else None,
                "LSTM_Species": lstm["Species"],
                "LSTM_Confidence": lstm["Confidence"],
                "Agreement": blast and blast["Species"] == lstm["Species"]
            })
    df = pd.DataFrame(hybrid_results)
    st.subheader("📊 Hybrid Predictions")
    st.dataframe(df)
    st.bar_chart(df["Agreement"].value_counts())