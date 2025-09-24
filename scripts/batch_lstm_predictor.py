from lstm_predictor import extract_features
from Bio import SeqIO

for record in SeqIO.parse("../data/queries.fasta", "fasta"):
    features = extract_features(str(record.seq))
    print(f"{record.id}: {features}")