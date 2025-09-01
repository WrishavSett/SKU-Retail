import os
import glob
import argparse
from config import Config
from embedding_pipeline import EmbeddingPipeline
from transaction_matcher import TransactionMatcher

def main():
    parser = argparse.ArgumentParser(description="Transaction Matching Pipeline")
    parser.add_argument("--file", type=str, help="Path to a single transaction CSV")
    parser.add_argument("--folder", type=str, help="Path to a folder containing transaction CSVs")
    parser.add_argument("--build", action='store_true', help="Rebuild FAISS index and metadata")
    args = parser.parse_args()

    config = Config()

    # # Step 1: Build embeddings & FAISS index
    # pipeline = EmbeddingPipeline(config)
    # pipeline.build_master_index()

    # ------------------ Build FAISS index if needed ------------------
    if args.build or not (os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE)):
        print("|INFO| Building FAISS index and generating metadata...")
        pipeline = EmbeddingPipeline(config)
        pipeline.build_master_index()
    else:
        print("|INFO| FAISS index and metadata found. Skipping build. Use --build to force rebuild.")

    # # Step 2: Query transactions
    # matcher = TransactionMatcher(config)
    # results_df = matcher.query_transactions()

    # ------------------ Determine transaction files ------------------
    transaction_files = []
    if args.file:
        if os.path.exists(args.file):
            transaction_files = [args.file]
        else:
            print(f"Error: File not found -> {args.file}")
            return
    elif args.folder:
        folder_path = args.folder
        transaction_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not transaction_files:
            print(f"No CSV files found in folder: {folder_path}")
            return
    else:
        print("Error: Please provide either --file or --folder")
        return

    # ------------------ Process each transaction file ------------------
    for file_path in transaction_files:
        transaction_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n|INFO| Processing transaction file: {file_path}")

        # Dynamically update config for this transaction
        config.transaction_file_path = file_path
        config.transaction_name = transaction_name
        config.OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_transaction_matches.csv")
        config.FINAL_OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_final_matches.csv")

        # Run transaction matcher
        matcher = TransactionMatcher(config)
        results_df = matcher.query_transactions()

        # # Step 3: Filter & save final results
        # matcher.filter_final_results(results_df)

if __name__ == "__main__":
    main()
