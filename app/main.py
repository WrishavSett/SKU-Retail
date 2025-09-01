from config import Config
from embedding_pipeline import EmbeddingPipeline
from transaction_matcher import TransactionMatcher

def main():
    config = Config()

    # # Step 1: Build embeddings & FAISS index
    # pipeline = EmbeddingPipeline(config)
    # pipeline.build_master_index()

    # Step 2: Query transactions
    matcher = TransactionMatcher(config)
    results_df = matcher.query_transactions()

    # # Step 3: Filter & save final results
    # matcher.filter_final_results(results_df)

if __name__ == "__main__":
    main()
