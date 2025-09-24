#!/usr/bin/env python3
"""
BLAST Batch Predictor for eDNA Species Classification
Processes multiple DNA sequences through BLAST and provides species predictions
"""

import subprocess
import os
import pandas as pd
import argparse
import sys
import json
import logging
from Bio import SeqIO
from io import StringIO
from pathlib import Path
import time

# Configuration
BLAST_DB = "../blast_db/blast_db"
TEMP_QUERY = "temp_query.fasta"
TEMP_OUTPUT = "temp_output.txt"
BLAST_BIN = "blastn"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blast_batch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_sequence(seq):
    """
    Validate DNA sequence - must contain only ATCG and be at least 20 bases long
    """
    if not seq or len(seq) < 20:
        return False
    return all(c.upper() in "ATCG" for c in seq)

def get_sequence_stats(seq):
    """
    Calculate basic statistics for a DNA sequence
    """
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return {}
    
    return {
        "Length": length,
        "GC_Content (%)": round((seq.count("G") + seq.count("C")) / length * 100, 2),
        "A_count": seq.count("A"),
        "T_count": seq.count("T"),
        "C_count": seq.count("C"),
        "G_count": seq.count("G")
    }

def run_blast(sequence, sequence_id="query"):
    """
    Run BLAST search for a single sequence
    """
    try:
        # Write query sequence to temporary file
        with open(TEMP_QUERY, "w") as f:
            f.write(f">{sequence_id}\n{sequence}")
        
        # Run BLAST command
        result = subprocess.run([
            BLAST_BIN,
            "-query", TEMP_QUERY,
            "-db", BLAST_DB,
            "-out", TEMP_OUTPUT,
            "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
            "-max_target_seqs", "10",  # Get top 10 matches
            "-word_size", "7",  # Smaller word size for short sequences
            "-evalue", "10",    # More permissive e-value
            "-dust", "no"      # Turn off low complexity filtering
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"BLAST failed for sequence {sequence_id}: {result.stderr}")
            return None
        
        # Check if output file exists and has content
        if not os.path.exists(TEMP_OUTPUT) or os.path.getsize(TEMP_OUTPUT) == 0:
            logger.warning(f"No BLAST matches found for sequence {sequence_id}")
            return None
        
        # Read BLAST results
        df = pd.read_csv(TEMP_OUTPUT, sep="\t", header=None)
        df.columns = ["Query", "Species", "Identity", "Length", "E-value", "Bit Score"]
        
        # Return best match (first row)
        best_match = df.iloc[0].to_dict()
        logger.info(f"BLAST match found for {sequence_id}: {best_match['Species']} ({best_match['Identity']}% identity)")
        
        return best_match
        
    except Exception as e:
        logger.error(f"Error running BLAST for sequence {sequence_id}: {str(e)}")
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
    """
    Process a FASTA file containing multiple sequences
    """
    results = []
    
    try:
        sequences = list(SeqIO.parse(fasta_file_path, "fasta"))
        logger.info(f"Processing {len(sequences)} sequences from {fasta_file_path}")
        
        for i, record in enumerate(sequences, 1):
            seq_id = record.id
            sequence = str(record.seq)
            
            logger.info(f"Processing sequence {i}/{len(sequences)}: {seq_id}")
            
            # Validate sequence
            if not validate_sequence(sequence):
                logger.warning(f"Invalid sequence {seq_id}: must contain only ATCG and be ≥20 bases")
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
                "Bit_Score": blast_result["Bit Score"] if blast_result else None,
                "Length": stats["Length"],
                "GC_Content": stats["GC_Content (%)"]
            }
            
            results.append(result_entry)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error processing FASTA file {fasta_file_path}: {str(e)}")
        return []
    
    return results

def process_single_sequence(sequence, seq_id="manual_input"):
    """
    Process a single DNA sequence
    """
    logger.info(f"Processing single sequence: {seq_id}")
    
    # Validate sequence
    if not validate_sequence(sequence):
        logger.error("Invalid sequence: must contain only ATCG and be ≥20 bases")
        return None
    
    # Get sequence statistics
    stats = get_sequence_stats(sequence)
    logger.info(f"Sequence stats: {stats}")
    
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
            "Bit_Score": blast_result["Bit Score"],
            "Length": stats["Length"],
            "GC_Content": stats["GC_Content (%)"]
        }
        logger.info(f"BLAST prediction: {result['Species']} ({result['Identity']}% identity)")
        return result
    else:
        logger.warning("No BLAST match found")
        return {
            "Sequence_ID": seq_id,
            "Sequence": sequence,
            "Status": "No Match",
            "Species": "No Match",
            "Identity": None,
            "E-value": None,
            "Bit_Score": None,
            "Length": stats["Length"],
            "GC_Content": stats["GC_Content (%)"]
        }

def save_results(results, output_file, format_type="csv"):
    """
    Save results to file in specified format
    """
    try:
        if format_type.lower() == "csv":
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        elif format_type.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error(f"Unsupported format: {format_type}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False

def main():
    """
    Main function to handle command line arguments and execute batch processing
    """
    parser = argparse.ArgumentParser(
        description="BLAST Batch Predictor for eDNA Species Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a FASTA file
  python blast_batch_predictor.py -f input.fasta -o results.csv
  
  # Process a single sequence
  python blast_batch_predictor.py -s "ATCGATCGATCGATCGATCG" -o result.json --format json
  
  # Process FASTA file with custom output format
  python blast_batch_predictor.py -f sequences.fasta -o output.json --format json
        """
    )
    
    parser.add_argument("-f", "--fasta", help="Input FASTA file path")
    parser.add_argument("-s", "--sequence", help="Single DNA sequence to process")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", 
                       help="Output format (default: csv)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.fasta and not args.sequence:
        logger.error("Either --fasta or --sequence must be provided")
        parser.print_help()
        sys.exit(1)
    
    if args.fasta and args.sequence:
        logger.error("Cannot process both --fasta and --sequence simultaneously")
        sys.exit(1)
    
    # Check if BLAST database exists
    if not os.path.exists(BLAST_DB + ".nhr"):  # BLAST database files have extensions
        logger.error(f"BLAST database not found at {BLAST_DB}")
        logger.error("Please ensure the BLAST database is properly installed")
        sys.exit(1)
    
    # Process input
    results = []
    
    if args.fasta:
        if not os.path.exists(args.fasta):
            logger.error(f"FASTA file not found: {args.fasta}")
            sys.exit(1)
        
        logger.info(f"Starting batch processing of {args.fasta}")
        results = process_fasta_file(args.fasta)
        
    elif args.sequence:
        logger.info("Processing single sequence")
        result = process_single_sequence(args.sequence)
        if result:
            results = [result]
    
    # Save results
    if results:
        success = save_results(results, args.output, args.format)
        if success:
            logger.info(f"Processing complete! {len(results)} sequences processed.")
            logger.info(f"Results saved to: {args.output}")
            
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
            logger.error("Failed to save results")
            sys.exit(1)
    else:
        logger.error("No results to save")
        sys.exit(1)

if __name__ == "__main__":
    main()