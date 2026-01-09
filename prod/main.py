import os
import glob
import argparse
from config import Config
from embedding_pipeline import EmbeddingPipeline
from transaction_matcher import TransactionMatcher
from llm_reranker import LLMReranker

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Transaction Matching Pipeline with Optional LLM Re-ranking (Google Gemini)")
    parser.add_argument("--file", type=str, help="Path to a single transaction CSV")
    parser.add_argument("--folder", type=str, help="Path to a folder containing transaction CSVs")
    parser.add_argument("--build", action='store_true', help="Rebuild FAISS index and metadata")
    
    # LLM OPTIONS
    parser.add_argument("--rerank", action='store_true', help="Enable LLM re-ranking after initial matching")
    parser.add_argument("--llm-model", type=str, default=None, 
                       help="Google Gemini model for re-ranking (default: from Config)")
    parser.add_argument("--rerank-only", action='store_true', 
                       help="Only run LLM re-ranking on existing CSV files (skip initial matching)")
    
    args = parser.parse_args()

    config = Config()
    
    # Validate API key
    if not config.GEMINI_API_KEY:
        print("|ERROR| GEMINI_API_KEY environment variable not set.")
        print("|INFO| Please set your Google Gemini API key:")
        print("|INFO| export GEMINI_API_KEY='your-api-key-here'")
        return

    # Build FAISS index if needed
    if args.build or not (os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE)):
        print("|INFO| Building FAISS index and generating metadata using Google Gemini embeddings...")
        pipeline = EmbeddingPipeline(config)
        pipeline.build_master_index()
    else:
        print("|INFO| FAISS index and metadata found. Skipping build. Use --build to force rebuild.")

    # Determine transaction files
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

    # Initialize LLM re-ranker if needed
    if args.rerank or args.rerank_only:
        llm_model = args.llm_model if args.llm_model else config.LLM_MODEL
        print(f"|INFO| Initializing Google Gemini LLM re-ranker with model: {llm_model}")
        reranker = LLMReranker(model_name=llm_model, api_key=config.GEMINI_API_KEY)

    # Process each transaction file
    for file_path in transaction_files:
        transaction_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n|INFO| Processing transaction file: {file_path}")

        # Update config for this transaction
        config.transaction_file_path = file_path
        config.transaction_name = transaction_name
        config.OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_transaction_matches.csv")
        config.FINAL_OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_final_matches.csv")

        # Run initial matching (unless rerank-only mode)
        if not args.rerank_only:
            print(f"|INFO| Running initial semantic matching for {transaction_name}")
            matcher = TransactionMatcher(config)
            results_df = matcher.query_transactions()
            
            # # Run filtering step (commented out as in original)
            # matcher.filter_final_results(results_df)
        
        # Run LLM re-ranking if requested
        if args.rerank or args.rerank_only:
            print(f"|INFO| Running Google Gemini LLM re-ranking for {transaction_name}")
            
            # Define input and output paths
            input_csv = config.OUTPUT_CSV
            llm_output_csv = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_llm_reranked.csv")
            
            # Check if input file exists
            if os.path.exists(input_csv):
                reranker.rerank_csv_matches(input_csv, llm_output_csv)
                print(f"|INFO| LLM re-ranking completed. Results saved to {llm_output_csv}")
            else:
                print(f"|WARNING| Input file not found: {input_csv}")
                print(f"|INFO| Run without --rerank-only first to generate initial matches")
        
        print(f"|INFO| Completed processing {transaction_name}")

if __name__ == "__main__":
    main()