#!/usr/bin/env python3
"""
Simple BLAST Batch Predictor for eDNA Species Classification
Minimal version with basic dependencies
"""

import subprocess
import os
import sys
import argparse
import csv
import json
from pathlib import Path

# Configuration
BLAST_DB = "../blast_db/blast_db"
TEMP_QUERY = "temp_query.fasta"
TEMP_OUTPUT = "temp_output.txt"
BLAST_BIN = "blastn"

def validate_sequence(seq):
    """Validate DNA sequence - must contain only ATCG and be at least 20 bases long"""
    if not seq or len(seq) < 20:
        return False
    return all(c.upper() in "ATCG" for c in seq)

def get_sequence_stats(seq):
    """Calculate basic statistics for a DNA sequence"""
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return {}
    
    return {
        "Length": length,
        "GC_Content": round((seq.count("G") + seq.count("C")) / length * 100, 2),
        "A_count": seq.count("A"),
        "T_count": seq.count("T"),
        "C_count": seq.count("C"),
        "G_count": seq.count("G")
    }

def parse_fasta_simple(fasta_file):
    """Simple FASTA parser without BioPython"""
    sequences = []
    current_id = None
    current_seq = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append((current_id, current_seq))
                current_id = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line
        
        # Add the last sequence
        if current_id:
            sequences.append((current_id, current_seq))
    
    return sequences

def run_blast(sequence, sequence_id="query"):
    """Run BLAST search for a single sequence"""
    try:
        # Write query sequence to temporary file
        with open(TEMP_QUERY, "w") as f:
            f.write(f">{sequence_id}\n{sequence}")
        
        # Run BLAST command with more sensitive parameters
        result = subprocess.run([
            BLAST_BIN,
            "-query", TEMP_QUERY,
            "-db", BLAST_DB,
            "-out", TEMP_OUTPUT,
            "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
            "-max_target_seqs", "10",
            "-word_size", "7",  # Smaller word size for short sequences
            "-evalue", "10",    # More permissive e-value
            "-dust", "no"      # Turn off low complexity filtering
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"BLAST failed for sequence {sequence_id}: {result.stderr}")
            return None
        
        # Check if output file exists and has content
        if not os.path.exists(TEMP_OUTPUT) or os.path.getsize(TEMP_OUTPUT) == 0:
            print(f"No BLAST matches found for sequence {sequence_id}")
            return None
        
        # Read BLAST results manually
        with open(TEMP_OUTPUT, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split('\t')
                if len(parts) >= 6:
                    return {
                        "Query": parts[0],
                        "Species": parts[1],
                        "Identity": float(parts[2]),
                        "Length": int(parts[3]),
                        "E-value": float(parts[4]),
                        "Bit_Score": float(parts[5])
                    }
        
        return None
        
    except Exception as e:
        print(f"Error running BLAST for sequence {sequence_id}: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        for temp_file in [TEMP_QUERY, TEMP_OUTPUT]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def process_fasta_file(fasta_file_path):
    """Process a FASTA file containing multiple sequences"""
    results = []
    
    try:
        sequences = parse_fasta_simple(fasta_file_path)
        print(f"Processing {len(sequences)} sequences from {fasta_file_path}")
        
        for i, (seq_id, sequence) in enumerate(sequences, 1):
            print(f"Processing sequence {i}/{len(sequences)}: {seq_id}")
            
            # Validate sequence
            if not validate_sequence(sequence):
                print(f"Invalid sequence {seq_id}: must contain only ATCG and be ≥20 bases")
                results.append({
                    "Sequence_ID": seq_id,
                    "Sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
                    "Status": "Invalid",
                    "Species": None,
                    "Identity": None,
                    "E-value": None,
                    "Bit_Score": None,
                    "Length": len(sequence),
                    "GC_Content": None
                })
                continue
            
            # Get sequence statistics
            stats = get_sequence_stats(sequence)
            
            # Run BLAST
            blast_result = run_blast(sequence, seq_id)
            
            # Compile results
            result_entry = {
                "Sequence_ID": seq_id,
                "Sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
                "Status": "Success" if blast_result else "No Match",
                "Species": blast_result["Species"] if blast_result else "No Match",
                "Identity": blast_result["Identity"] if blast_result else None,
                "E-value": blast_result["E-value"] if blast_result else None,
                "Bit_Score": blast_result["Bit_Score"] if blast_result else None,
                "Length": stats["Length"],
                "GC_Content": stats["GC_Content"]
            }
            
            results.append(result_entry)
            
    except Exception as e:
        print(f"Error processing FASTA file {fasta_file_path}: {str(e)}")
        return []
    
    return results

def process_single_sequence(sequence, seq_id="manual_input"):
    """Process a single DNA sequence"""
    print(f"Processing single sequence: {seq_id}")
    
    # Validate sequence
    if not validate_sequence(sequence):
        print("Invalid sequence: must contain only ATCG and be ≥20 bases")
        return None
    
    # Get sequence statistics
    stats = get_sequence_stats(sequence)
    print(f"Sequence stats: {stats}")
    
    # Run BLAST
    blast_result = run_blast(sequence, seq_id)
    
    if blast_result:
        result = {
            "Sequence_ID": seq_id,
            "Sequence": sequence,
            "Status": "Success",
            "Species": blast_result["Species"],
            "Identity": blast_result["Identity"],
            "E-value": blast_result["E-value"],
            "Bit_Score": blast_result["Bit_Score"],
            "Length": stats["Length"],
            "GC_Content": stats["GC_Content"]
        }
        print(f"BLAST prediction: {result['Species']} ({result['Identity']}% identity)")
        return result
    else:
        print("No BLAST match found")
        return {
            "Sequence_ID": seq_id,
            "Sequence": sequence,
            "Status": "No Match",
            "Species": "No Match",
            "Identity": None,
            "E-value": None,
            "Bit_Score": None,
            "Length": stats["Length"],
            "GC_Content": stats["GC_Content"]
        }

def save_results_csv(results, output_file):
    """Save results to CSV file"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving CSV results: {str(e)}")
        return False

def save_results_json(results, output_file):
    """Save results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving JSON results: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and execute batch processing"""
    parser = argparse.ArgumentParser(
        description="Simple BLAST Batch Predictor for eDNA Species Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a FASTA file
  python simple_blast_predictor.py -f input.fasta -o results.csv
  
  # Process a single sequence
  python simple_blast_predictor.py -s "ATCGATCGATCGATCGATCG" -o result.json --format json
        """
    )
    
    parser.add_argument("-f", "--fasta", help="Input FASTA file path")
    parser.add_argument("-s", "--sequence", help="Single DNA sequence to process")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", 
                       help="Output format (default: csv)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.fasta and not args.sequence:
        print("Error: Either --fasta or --sequence must be provided")
        parser.print_help()
        sys.exit(1)
    
    if args.fasta and args.sequence:
        print("Error: Cannot process both --fasta and --sequence simultaneously")
        sys.exit(1)
    
    # Check if BLAST database exists
    if not os.path.exists(BLAST_DB + ".nhr"):
        print(f"Error: BLAST database not found at {BLAST_DB}")
        print("Please ensure the BLAST database is properly installed")
        sys.exit(1)
    
    # Process input
    results = []
    
    if args.fasta:
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA file not found: {args.fasta}")
            sys.exit(1)
        
        print(f"Starting batch processing of {args.fasta}")
        results = process_fasta_file(args.fasta)
        
    elif args.sequence:
        print("Processing single sequence")
        result = process_single_sequence(args.sequence)
        if result:
            results = [result]
    
    # Save results
    if results:
        if args.format.lower() == "csv":
            success = save_results_csv(results, args.output)
        else:
            success = save_results_json(results, args.output)
            
        if success:
            print(f"Processing complete! {len(results)} sequences processed.")
            print(f"Results saved to: {args.output}")
            
            # Print summary
            successful = sum(1 for r in results if r["Status"] == "Success")
            no_match = sum(1 for r in results if r["Status"] == "No Match")
            invalid = sum(1 for r in results if r["Status"] == "Invalid")
            
            print(f"\n=== PROCESSING SUMMARY ===")
            print(f"Total sequences: {len(results)}")
            print(f"Successful matches: {successful}")
            print(f"No matches found: {no_match}")
            print(f"Invalid sequences: {invalid}")
            print(f"Output file: {args.output}")
        else:
            print("Failed to save results")
            sys.exit(1)
    else:
        print("No results to save")
        sys.exit(1)

if __name__ == "__main__":
    main()
