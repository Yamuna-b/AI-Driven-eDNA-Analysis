def parse_fastq(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i % 4 == 1:  # Every sequence line in FASTQ
                sequences.append(line.strip())
    return sequences

def kmer_tokenize(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def process_paired_fastq(forward_path, reverse_path):
    forward_seqs = parse_fastq(forward_path)
    reverse_seqs = parse_fastq(reverse_path)
    assert len(forward_seqs) == len(reverse_seqs), "Paired files differ in length"
    forward_kmers = [kmer_tokenize(seq) for seq in forward_seqs]
    reverse_kmers = [kmer_tokenize(seq) for seq in reverse_seqs]
    return forward_kmers, reverse_kmers

if __name__ == "__main__":
    f_path = "../data/SRR12076396_1.fastq"
    r_path = "../data/SRR12076396_2.fastq"
    f_kmers, r_kmers = process_paired_fastq(f_path, r_path)

    print(f"Reads parsed: {len(f_kmers)}")
    print("First forward read k-mers:", f_kmers[0])
    print("First reverse read k-mers:", r_kmers[0])
