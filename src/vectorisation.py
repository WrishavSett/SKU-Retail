import os
import numpy as np
import pandas as pd
import ollama
import faiss
import json
from tqdm import tqdm
import re

# ----------------- CONFIG -----------------
MODEL_NAME = "nomic-embed-text"

FAISS_INDEX_FILE = "./temp/master_index.faiss"
METADATA_FILE = "./temp/metadata.json"
OUTPUT_CSV = "./temp/transaction_matches.csv"
FINAL_OUTPUT_CSV = "./temp/transaction_final_matches.csv"

# ----------------- LOAD DATA -----------------
master = pd.read_csv('./dataset/master.csv')
transaction = pd.read_csv('./dataset/transaction/jan-22.csv')

m_columns = ['itemcode', 'catcode', 'company', 'mbrand', 'brand', 'sku', 'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']
t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']

master = master[m_columns]
transaction = transaction[t_columns]

# ----------------- FUNCTION: EMBEDDING -----------------
def get_embedding(text):
    """Call local Ollama client to get embedding for a given text."""
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"|ERROR| Error embedding text: {e}")
        return None

# # ----------------- PREPARE MASTER EMBEDDINGS -----------------
# print(f"|INFO| Generating embeddings for {len(master)} master rows...")
# texts = []
# itemcodes = []
# metadata = {}

# for _, row in tqdm(master.iterrows(), total=len(master), desc="Master Embeddings"):
#     itemcode = str(row["itemcode"])
#     itemcodes.append(itemcode)
    
#     # Join all fields except itemcode
#     row_text = " ".join(str(val) for col, val in row.items() if col != "itemcode" and pd.notna(val))
#     texts.append(row_text)
    
#     # Store metadata
#     metadata[itemcode] = row.drop("itemcode").to_dict()

# # Generate embeddings
# embeddings = []
# for text in tqdm(texts, desc="Fetching Embeddings"):
#     emb = get_embedding(text)
#     if emb is not None:
#         embeddings.append(emb)
#     else:
#         embeddings.append(np.zeros(1536, dtype=np.float32))  # fallback, adjust dim if model differs

# # Convert embeddings to numpy array
# embeddings_np = np.vstack(embeddings).astype('float32')

# # ----------------- BUILD FAISS CPU INDEX -----------------
# embedding_dim = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(embeddings_np)

# # Save index and metadata
# faiss.write_index(index, FAISS_INDEX_FILE)
# with open(METADATA_FILE, "w") as f:
#     json.dump(metadata, f, indent=2)

# print(f"|INFO| Saved {len(embeddings)} embeddings to {FAISS_INDEX_FILE} and metadata to {METADATA_FILE}")

# ----------------- QUERY TRANSACTIONS AND SAVE -----------------
print(f"\n|INFO| Querying {len(transaction)} transaction rows...")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

itemcodes = list(metadata.keys())
results_list = []

for _, row in tqdm(transaction.iterrows(), total=len(transaction), desc="Transaction Queries"):
    query_text = " ".join(str(val) for val in row if pd.notna(val))
    query_emb = get_embedding(query_text)
    
    if query_emb is not None:
        query_np = np.array([query_emb]).astype('float32')
        distances, indices = index.search(query_np, k=10)
        
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            itemcode = itemcodes[idx]
            metadata_item = metadata[itemcode]
            
            result_entry = {
                "rank": rank + 1,
                "matched_itemcode": itemcode,
                "distance": dist
            }
            
            # Add all transaction columns with prefix t_
            for col in transaction.columns:
                result_entry[f"t_{col}"] = row[col]
            
            # Add master metadata as separate columns prefixed with m_
            for k, v in metadata_item.items():
                result_entry[f"m_{k}"] = v
            
            results_list.append(result_entry)
    else:
        result_entry = {
            "rank": None,
            "matched_itemcode": None,
            "distance": None
        }
        
        # Add all transaction columns with prefix t_
        for col in transaction.columns:
            result_entry[f"t_{col}"] = row[col]
        
        # Add empty master metadata columns
        for k in master.columns:
            if k != "itemcode":  # skip itemcode, it's already in matched_itemcode
                result_entry[f"m_{k}"] = None
        
        results_list.append(result_entry)

# Save results to CSV with transaction columns first
results_df = pd.DataFrame(results_list)

# Build ordered column list: transaction first, then match info, then master metadata
t_cols_prefixed = [f"t_{c}" for c in t_columns]
m_cols_prefixed = [f"m_{c}" for c in master.columns if c != "itemcode"]
ordered_cols = t_cols_prefixed + ["rank", "matched_itemcode", "distance"] + m_cols_prefixed
results_df = results_df[ordered_cols]

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"|INFO| Saved transaction matches to {OUTPUT_CSV}")

# ----------------- HELPER: Extract numeric from PACKTYPE -----------------
def extract_numeric(val):
    """Extract the first numeric value from a string, else return None."""
    if pd.isna(val):
        return None
    match = re.search(r"\d+(\.\d+)?", str(val))
    return float(match.group()) if match else None

# ----------------- PROCESS FILTERED OUTPUT -----------------
final_results = []

# Group results by transaction row (we assume all suggestions have the same t_ columns per transaction)
grouped = results_df.groupby([col for col in results_df.columns if col.startswith("t_")])

for _, group in grouped:
    # Get numeric part from t_PACKTYPE (all rows in group have the same transaction values)
    t_packtype_val = group.iloc[0]["t_PACKTYPE"]
    t_num = extract_numeric(t_packtype_val)

    chosen_row = None

    if t_num is not None:
        # Try to find a match in m_wght
        for _, row in group.iterrows():
            try:
                m_wght_val = float(row["m_wght"])
                if m_wght_val == t_num:
                    chosen_row = row
                    break
            except (ValueError, TypeError):
                continue

    # If no match found, take first suggestion (rank=1)
    if chosen_row is None:
        chosen_row = group.sort_values("rank").iloc[0]

    final_results.append(chosen_row)

# Build final dataframe
final_df = pd.DataFrame(final_results)

# Save new CSV
final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
print(f"Saved final filtered matches to {FINAL_OUTPUT_CSV}")