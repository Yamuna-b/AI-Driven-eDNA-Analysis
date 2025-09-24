import streamlit as st
import pandas as pd
import subprocess
import os
from Bio import SeqIO
from io import StringIO

# --- Config ---
BLAST_DB = "../blast_db/blast_db"           # points to the database built from ref.fasta
QUERY_FASTA = "../data/queries.fasta"       # your input sequences
BLAST_BIN = "blastn"
QUERY_FILE = "temp_query.fasta"
OUTPUT_FILE = "temp_output.txt"

# --- Run BLAST ---
def run_blast(sequence):
    with open(QUERY_FILE, "w") as f:
        f.write(f">query\n{sequence}")
    subprocess.run([
        BLAST_BIN,
        "-query", QUERY_FILE,
        "-db", BLAST_DB,
        "-out", OUTPUT_FILE,
        "-outfmt", "6 qseqid sseqid pident length evalue bitscore"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        return None

    df = pd.read_csv(OUTPUT_FILE, sep="\t", header=None)
    df.columns = ["Query", "Species", "Identity", "Length", "E-value", "Bit Score"]
    return df.iloc[0].to_dict()

# --- Sequence Validation ---
def validate_sequence(seq):
    return all(c in "ATCG" for c in seq.upper()) and len(seq) >= 20

def get_stats(seq):
    seq = seq.upper()
    return {
        "Length": len(seq),
        "GC_Content (%)": round((seq.count("G") + seq.count("C")) / len(seq) * 100, 2),
        "A": seq.count("A"),
        "T": seq.count("T"),
        "C": seq.count("C"),
        "G": seq.count("G")
    }

# --- UI ---
st.set_page_config("BLAST eDNA Predictor", layout="wide")
st.title("🧬 BLAST-Driven eDNA Species Predictor")

# --- Manual Entry ---
manual_seq = st.sidebar.text_area("Enter DNA Sequence (ATCG only)", height=100)
if manual_seq:
    if validate_sequence(manual_seq):
        st.subheader("Sequence Statistics")
        st.json(get_stats(manual_seq))

        st.markdown("### 🔍 BLAST Prediction")
        result = run_blast(manual_seq)
        if result:
            st.json(result)
        else:
            st.warning("No BLAST match found.")
    else:
        st.error("Invalid sequence. Use only A,T,C,G and ≥20 bases.")

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload FASTA File", type=["fasta"])
if uploaded_file:
    fasta_text = uploaded_file.getvalue().decode("utf-8")
    sequences = list(SeqIO.parse(StringIO(fasta_text), "fasta"))
    blast_results = []

    for rec in sequences:
        seq = str(rec.seq)
        if validate_sequence(seq):
            result = run_blast(seq)
            blast_results.append({
                "ID": rec.id,
                "Sequence": seq,
                "Species": result["Species"] if result else "No Match",
                "Identity": result["Identity"] if result else None,
                "E-value": result["E-value"] if result else None,
                "Bit Score": result["Bit Score"] if result else None
            })

    df = pd.DataFrame(blast_results)
    st.subheader("📊 BLAST Predictions")
    st.dataframe(df)

# --- Footer ---
st.markdown("---")
st.markdown("Built by Swetha • Powered by BLAST • For competition-grade eDNA analysis")