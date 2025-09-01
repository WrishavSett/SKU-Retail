import pandas as pd
import faiss
import json
import re
import numpy as np
from tqdm import tqdm
from config import Config
from embedding_pipeline import EmbeddingPipeline

class TransactionMatcher:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = EmbeddingPipeline(config)

    def extract_numeric(self, val):
        if pd.isna(val):
            return None
        match = re.search(r"\d+(\.\d+)?", str(val))
        return float(match.group()) if match else None

    def query_transactions(self):
        """Query FAISS index with transaction data and save matches."""
        transaction_full = pd.read_csv(self.config.transaction_file_path)
        transaction_proc = transaction_full[self.config.t_columns]

        index = faiss.read_index(self.config.FAISS_INDEX_FILE)
        with open(self.config.METADATA_FILE, "r") as f:
            metadata = json.load(f)

        itemcodes = list(metadata.keys())
        results_list = []

        for i, row in tqdm(transaction_proc.iterrows(), total=len(transaction_proc), desc="Transaction Queries"):
            query_text = " ".join(str(val) for val in row if pd.notna(val))
            query_emb = self.pipeline.get_embedding(query_text)

            if query_emb is not None:
                # # Ensure embedding matches FAISS index dimension
                # if query_emb.shape[0] != index.d:
                #     print(f"|WARN| Skipping transaction row {i}: embedding dim {query_emb.shape[0]} != index dim {index.d}")
                #     # fallback entry with None
                #     result_entry = {"rank": None, "matched_itemcode": None, "distance": None}
                #     for col in transaction_full.columns:
                #         result_entry[f"t_{col}"] = transaction_full.iloc[i][col]
                #     for k in Config.m_columns:
                #         result_entry[f"m_{k}"] = None
                #     results_list.append(result_entry)
                #     continue  # skip to next row

                # Pad or truncate to match FAISS index dimension
                if query_emb.shape[0] < index.d:
                    padded_emb = np.zeros(index.d, dtype=np.float32)
                    padded_emb[:query_emb.shape[0]] = query_emb
                    query_emb = padded_emb
                    print(f"|INFO| Padded query embedding for row {i} from {query_emb.shape[0]} to {index.d} dimensions.")
                    print(f"       Row data: {transaction_full.iloc[i].to_dict()}")
                elif query_emb.shape[0] > index.d:
                    print(f"|INFO| Truncating query embedding for row {i} from {query_emb.shape[0]} to {index.d} dimensions.")
                    print(f"       Row data: {transaction_full.iloc[i].to_dict()}")
                    query_emb = query_emb[:index.d]

                distances, indices = index.search(query_emb.reshape(1, -1), k=3)
                for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                    itemcode = itemcodes[idx]
                    metadata_item = metadata[itemcode]

                    result_entry = {
                        "rank": rank + 1,
                        "matched_itemcode": itemcode,
                        "distance": dist
                    }
                    for col in transaction_full.columns:
                        result_entry[f"t_{col}"] = transaction_full.iloc[i][col]
                    for k, v in metadata_item.items():
                        result_entry[f"m_{k}"] = v
                    results_list.append(result_entry)
            else:
                # Fallback when no embedding
                result_entry = {"rank": None, "matched_itemcode": None, "distance": None}
                for col in transaction_full.columns:
                    result_entry[f"t_{col}"] = transaction_full.iloc[i][col]
                for k in Config.m_columns:
                    result_entry[f"m_{k}"] = None
                results_list.append(result_entry)

        results_df = pd.DataFrame(results_list)
        t_cols_prefixed = [f"t_{c}" for c in transaction_full.columns]
        m_cols_prefixed = [f"m_{c}" for c in Config.m_columns if c != "itemcode"]
        ordered_cols = t_cols_prefixed + ["rank", "matched_itemcode", "distance"] + m_cols_prefixed
        results_df = results_df[ordered_cols]

        results_df.to_csv(self.config.OUTPUT_CSV, index=False)
        print(f"|INFO| Saved transaction matches to {self.config.OUTPUT_CSV}")

        return results_df

    def filter_final_results(self, results_df):
        """Filter results based on pack size and save final CSV."""
        final_results = []

        grouped = results_df.groupby([col for col in results_df.columns if col.startswith("t_")])

        for _, group in grouped:
            t_packtype_val = group.iloc[0]["t_PACKTYPE"]
            t_num = self.extract_numeric(t_packtype_val)

            chosen_row = None
            if t_num is not None:
                for _, row in group.iterrows():
                    try:
                        if float(row["m_wght"]) == t_num:
                            chosen_row = row
                            break
                    except (ValueError, TypeError):
                        continue
            if chosen_row is None:
                chosen_row = group.sort_values("rank").iloc[0]

            final_results.append(chosen_row)

        final_df = pd.DataFrame(final_results)
        final_df.to_csv(self.config.FINAL_OUTPUT_CSV, index=False)
        print(f"|INFO| Saved final filtered matches to {self.config.FINAL_OUTPUT_CSV}")
