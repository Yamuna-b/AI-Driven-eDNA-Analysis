def parse_fastq(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        line_number = 0
        for line in file:
            line_number += 1
            # Every 4th line starting from line 2 contains sequence
            if line_number % 4 == 2:
                seq = line.strip()
                sequences.append(seq)
    return sequences

if __name__ == "__main__":
    # Example: Change the filepath to your actual FASTQ file path
    fastq_path = "../data/SRR12076396_1.fastq"  # relative path from 'scripts/' folder

    seqs = parse_fastq(fastq_path)
    print(f"Total sequences parsed: {len(seqs)}")
    print("First 5 sequences:")
    for s in seqs[:5]:
        print(s)