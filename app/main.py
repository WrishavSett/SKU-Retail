# import os
# import glob
# import argparse
# from config import Config
# from embedding_pipeline import EmbeddingPipeline
# from transaction_matcher import TransactionMatcher

# def main():
#     parser = argparse.ArgumentParser(description="Transaction Matching Pipeline")
#     parser.add_argument("--file", type=str, help="Path to a single transaction CSV")
#     parser.add_argument("--folder", type=str, help="Path to a folder containing transaction CSVs")
#     parser.add_argument("--build", action='store_true', help="Rebuild FAISS index and metadata")
#     args = parser.parse_args()

#     config = Config()

#     # # Step 1: Build embeddings & FAISS index
#     # pipeline = EmbeddingPipeline(config)
#     # pipeline.build_master_index()

#     # ------------------ Build FAISS index if needed ------------------
#     if args.build or not (os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE)):
#         print("|INFO| Building FAISS index and generating metadata...")
#         pipeline = EmbeddingPipeline(config)
#         pipeline.build_master_index()
#     else:
#         print("|INFO| FAISS index and metadata found. Skipping build. Use --build to force rebuild.")

#     # # Step 2: Query transactions
#     # matcher = TransactionMatcher(config)
#     # results_df = matcher.query_transactions()

#     # ------------------ Determine transaction files ------------------
#     transaction_files = []
#     if args.file:
#         if os.path.exists(args.file):
#             transaction_files = [args.file]
#         else:
#             print(f"Error: File not found -> {args.file}")
#             return
#     elif args.folder:
#         folder_path = args.folder
#         transaction_files = glob.glob(os.path.join(folder_path, "*.csv"))
#         if not transaction_files:
#             print(f"No CSV files found in folder: {folder_path}")
#             return
#     else:
#         print("Error: Please provide either --file or --folder")
#         return

#     # ------------------ Process each transaction file ------------------
#     for file_path in transaction_files:
#         transaction_name = os.path.splitext(os.path.basename(file_path))[0]
#         print(f"\n|INFO| Processing transaction file: {file_path}")

#         # Dynamically update config for this transaction
#         config.transaction_file_path = file_path
#         config.transaction_name = transaction_name
#         config.OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_transaction_matches.csv")
#         config.FINAL_OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_final_matches.csv")

#         # Run transaction matcher
#         matcher = TransactionMatcher(config)
#         results_df = matcher.query_transactions()

#         # # Step 3: Filter & save final results
#         # matcher.filter_final_results(results_df)

# if __name__ == "__main__":
#     main()


# # Updated main.py with LLM integration option
# import os
# import glob
# import argparse
# from config import Config
# from embedding_pipeline import EmbeddingPipeline
# from transaction_matcher import TransactionMatcher
# from llm_reranker import LLMReranker

# import logging

# logging.basicConfig(
#     level=logging.INFO,  # keep your own logs
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logging.getLogger("httpx").setLevel(logging.WARNING)


# def main():
#     parser = argparse.ArgumentParser(description="Transaction Matching Pipeline with Optional LLM Re-ranking")
#     parser.add_argument("--file", type=str, help="Path to a single transaction CSV")
#     parser.add_argument("--folder", type=str, help="Path to a folder containing transaction CSVs")
#     parser.add_argument("--build", action='store_true', help="Rebuild FAISS index and metadata")
    
#     # NEW LLM OPTIONS
#     parser.add_argument("--use-llm", action='store_true', help="Enable LLM re-ranking after initial matching")
#     parser.add_argument("--llm-model", type=str, default="qwen2.5:7b", 
#                        help="LLM model for re-ranking (default: qwen2.5:7b)")
#     parser.add_argument("--llm-only", action='store_true', 
#                        help="Only run LLM re-ranking on existing CSV files (skip initial matching)")
    
#     args = parser.parse_args()

#     config = Config()

#     # Build FAISS index if needed (unchanged)
#     if args.build or not (os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE)):
#         print("|INFO| Building FAISS index and generating metadata...")
#         pipeline = EmbeddingPipeline(config)
#         pipeline.build_master_index()
#     else:
#         print("|INFO| FAISS index and metadata found. Skipping build. Use --build to force rebuild.")

#     # Determine transaction files (unchanged)
#     transaction_files = []
#     if args.file:
#         if os.path.exists(args.file):
#             transaction_files = [args.file]
#         else:
#             print(f"Error: File not found -> {args.file}")
#             return
#     elif args.folder:
#         folder_path = args.folder
#         transaction_files = glob.glob(os.path.join(folder_path, "*.csv"))
#         if not transaction_files:
#             print(f"No CSV files found in folder: {folder_path}")
#             return
#     else:
#         print("Error: Please provide either --file or --folder")
#         return

#     # Initialize LLM re-ranker if needed
#     if args.use_llm or args.llm_only:
#         print(f"|INFO| Initializing LLM re-ranker with model: {args.llm_model}")
#         reranker = LLMReranker(args.llm_model)

#     # Process each transaction file
#     for file_path in transaction_files:
#         transaction_name = os.path.splitext(os.path.basename(file_path))[0]
#         print(f"\n|INFO| Processing transaction file: {file_path}")

#         # Update config for this transaction (unchanged)
#         config.transaction_file_path = file_path
#         config.transaction_name = transaction_name
#         config.OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_transaction_matches.csv")
#         config.FINAL_OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_final_matches.csv")

#         # Run initial matching (unless LLM-only mode)
#         if not args.llm_only:
#             print(f"|INFO| Running initial semantic matching for {transaction_name}")
#             matcher = TransactionMatcher(config)
#             results_df = matcher.query_transactions()
            
#             # # Run filtering step (unchanged)  
#             # matcher.filter_final_results(results_df)
        
#         # Run LLM re-ranking if requested
#         if args.use_llm or args.llm_only:
#             print(f"|INFO| Running LLM re-ranking for {transaction_name}")
            
#             # Define input and output paths
#             input_csv = config.OUTPUT_CSV
#             llm_output_csv = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_llm_reranked.csv")
            
#             # Check if input file exists
#             if os.path.exists(input_csv):
#                 reranker.rerank_csv_matches(input_csv, llm_output_csv)
#                 print(f"|INFO| LLM re-ranking completed. Results saved to {llm_output_csv}")
#             else:
#                 print(f"|WARNING| Input file not found: {input_csv}")
#                 print(f"|INFO| Run without --llm-only first to generate initial matches")
        
#         print(f"|INFO| Completed processing {transaction_name}")

# if __name__ == "__main__":
#     main()

# Updated main.py - MINIMAL CHANGE: Only file verification fix
import os
import glob
import argparse
import time
import pandas as pd
from config import Config
from embedding_pipeline import EmbeddingPipeline
from transaction_matcher import TransactionMatcher
from llm_reranker import LLMReranker

def wait_for_file_completion(file_path: str, max_wait: int = 30) -> bool:
    """
    Wait for file to be created and writing to complete
    Returns True if file is ready, False if timeout
    """
    # Wait for file to exist
    wait_time = 0
    while not os.path.exists(file_path) and wait_time < max_wait:
        time.sleep(1)
        wait_time += 1
    
    if not os.path.exists(file_path):
        print(f"|ERROR| File not created within {max_wait} seconds: {file_path}")
        return False
    
    # Wait for file size to stabilize (writing complete)
    prev_size = 0
    stable_count = 0
    while stable_count < 3 and wait_time < max_wait:  # 3 consecutive stable readings
        current_size = os.path.getsize(file_path)
        if current_size == prev_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
        prev_size = current_size
        time.sleep(0.5)
        wait_time += 0.5
    
    if stable_count < 3:
        print(f"|WARNING| File may still be writing: {file_path}")
        return False
    
    # Final verification - try to read the file
    try:
        test_df = pd.read_csv(file_path, nrows=1)
        print(f"|INFO| File verification successful: {current_size} bytes, {len(test_df.columns)} columns")
        return True
    except Exception as e:
        print(f"|ERROR| File verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Transaction Matching Pipeline with Optional LLM Re-ranking")
    parser.add_argument("--file", type=str, help="Path to a single transaction CSV")
    parser.add_argument("--folder", type=str, help="Path to a folder containing transaction CSVs")
    parser.add_argument("--build", action='store_true', help="Rebuild FAISS index and metadata")
    
    # LLM OPTIONS
    parser.add_argument("--use-llm", action='store_true', help="Enable LLM re-ranking after initial matching")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b", 
                       help="LLM model for re-ranking (default: qwen2.5:7b)")
    parser.add_argument("--llm-only", action='store_true', 
                       help="Only run LLM re-ranking on existing CSV files (skip initial matching)")
    
    args = parser.parse_args()

    config = Config()

    # Build FAISS index if needed (unchanged)
    if args.build or not (os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.METADATA_FILE)):
        print("|INFO| Building FAISS index and generating metadata...")
        pipeline = EmbeddingPipeline(config)
        pipeline.build_master_index()
    else:
        print("|INFO| FAISS index and metadata found. Skipping build. Use --build to force rebuild.")

    # Determine transaction files (unchanged)
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

    # Initialize LLM re-ranker if needed (unchanged)
    if args.use_llm or args.llm_only:
        print(f"|INFO| Initializing LLM re-ranker with model: {args.llm_model}")
        reranker = LLMReranker(args.llm_model)

    # Process each transaction file
    for file_path in transaction_files:
        transaction_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n|INFO| Processing transaction file: {file_path}")

        # Update config for this transaction (unchanged)
        config.transaction_file_path = file_path
        config.transaction_name = transaction_name
        config.OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_transaction_matches.csv")
        config.FINAL_OUTPUT_CSV = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_final_matches.csv")

        # Run initial matching (unless LLM-only mode) - UNCHANGED
        if not args.llm_only:
            print(f"|INFO| Running initial semantic matching for {transaction_name}")
            matcher = TransactionMatcher(config)
            results_df = matcher.query_transactions()
            
            # *** ONLY CHANGE: Add file completion verification ***
            print(f"|INFO| Waiting for file writing to complete...")
            if not wait_for_file_completion(config.OUTPUT_CSV, max_wait=60):
                print(f"|ERROR| Could not verify file completion: {config.OUTPUT_CSV}")
                continue
            
            # Run filtering step (unchanged)  
            matcher.filter_final_results(results_df)
        
        # Run LLM re-ranking if requested (unchanged)
        if args.use_llm or args.llm_only:
            print(f"|INFO| Running LLM re-ranking for {transaction_name}")
            
            # Define input and output paths
            input_csv = config.OUTPUT_CSV
            llm_output_csv = os.path.join(config.OUTPUT_FOLDER, f"{transaction_name}_llm_reranked.csv")
            
            # Check if input file exists
            if os.path.exists(input_csv):
                reranker.rerank_csv_matches(input_csv, llm_output_csv)
                print(f"|INFO| LLM re-ranking completed. Results saved to {llm_output_csv}")
            else:
                print(f"|WARNING| Input file not found: {input_csv}")
                print(f"|INFO| Run without --llm-only first to generate initial matches")
        
        print(f"|INFO| Completed processing {transaction_name}")

if __name__ == "__main__":
    main()