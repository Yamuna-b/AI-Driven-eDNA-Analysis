import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Step 1: Parse FASTQ and extract sequences
def parse_fastq(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i % 4 == 1:  # sequence line in FASTQ format
                sequences.append(line.strip())
    return sequences

# Step 2: K-mer tokenization (default k=6)
def kmer_tokenize(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def process_paired_fastq(forward_path, reverse_path):
    forward_seqs = parse_fastq(forward_path)
    reverse_seqs = parse_fastq(reverse_path)
    assert len(forward_seqs) == len(reverse_seqs), "Paired FASTQ files differ in length"
    forward_kmers = [kmer_tokenize(seq) for seq in forward_seqs]
    reverse_kmers = [kmer_tokenize(seq) for seq in reverse_seqs]
    return forward_kmers, reverse_kmers

# Step 3: Load DNABERT tokenizer and model
def load_model_and_tokenizer(model_name="zhihan1996/DNA_bert_6"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Step 4: Batch embed k-mer tokenized sequences
def embed_sequences_batch(kmer_sequences, tokenizer, model, device='cpu', batch_size=16):
    embeddings = []
    for i in range(0, len(kmer_sequences), batch_size):
        batch = kmer_sequences[i:i+batch_size]
        texts = [" ".join(kmers) for kmers in batch]

        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu())

    embeddings = torch.vstack(embeddings)
    return embeddings

# Step 5: Save embeddings to numpy file
def save_embeddings(embeddings, filepath):
    np.save(filepath, embeddings.numpy())
    print(f"Saved embeddings to {filepath}")

if __name__ == "__main__":
    # Paths to your FASTQ files
    forward_fastq = "../data/SRR12076396_1.fastq"
    reverse_fastq = "../data/SRR12076396_2.fastq"

    print("Parsing and tokenizing sequences...")
    f_kmers, r_kmers = process_paired_fastq(forward_fastq, reverse_fastq)

    print(f"Forward reads parsed: {len(f_kmers)}")
    print(f"Reverse reads parsed: {len(r_kmers)}")

    print("Loading DNABERT model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    print("Embedding forward reads in batches...")
    forward_embeddings = embed_sequences_batch(f_kmers, tokenizer, model, device=device, batch_size=16)

    print("Embedding reverse reads in batches...")
    reverse_embeddings = embed_sequences_batch(r_kmers, tokenizer, model, device=device, batch_size=16)

    # Create results folder if not existing
    os.makedirs("../results", exist_ok=True)

    save_embeddings(forward_embeddings, "../results/forward_read_embeddings.npy")
    save_embeddings(reverse_embeddings, "../results/reverse_read_embeddings.npy")

    print("Embedding pipeline completed successfully.")

