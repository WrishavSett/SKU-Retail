import os
import numpy as np
import pandas as pd
import ollama
import faiss
import json
from tqdm import tqdm

# ----------------- CONFIG -----------------
MODEL_NAME = "nomic-embed-text"

FAISS_INDEX_FILE = "./temp/master_index.faiss"
METADATA_FILE = "./temp/metadata.json"
OUTPUT_CSV = "./temp/transaction_matches.csv"

# ----------------- LOAD DATA -----------------
master = pd.read_csv('./master.csv')
transaction = pd.read_csv('./jan-22.csv')

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
        print(f"Error embedding text: {e}")
        return None

# ----------------- PREPARE MASTER EMBEDDINGS -----------------
print(f"Generating embeddings for {len(master)} master rows...")
texts = []
itemcodes = []
metadata = {}

for _, row in tqdm(master.iterrows(), total=len(master), desc="Master Embeddings"):
    itemcode = str(row["itemcode"])
    itemcodes.append(itemcode)
    
    # Join all fields except itemcode
    row_text = " ".join(str(val) for col, val in row.items() if col != "itemcode" and pd.notna(val))
    texts.append(row_text)
    
    # Store metadata
    metadata[itemcode] = row.drop("itemcode").to_dict()

# Generate embeddings
embeddings = []
for text in tqdm(texts, desc="Fetching Embeddings"):
    emb = get_embedding(text)
    if emb is not None:
        embeddings.append(emb)
    else:
        embeddings.append(np.zeros(1536, dtype=np.float32))  # fallback, adjust dim if model differs

# Convert embeddings to numpy array
embeddings_np = np.vstack(embeddings).astype('float32')

# # ----------------- BUILD FAISS INDEX -----------------
# embedding_dim = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(embeddings_np)

# # Save index and metadata
# faiss.write_index(index, FAISS_INDEX_FILE)
# with open(METADATA_FILE, "w") as f:
#     json.dump(metadata, f, indent=2)

# print(f"Saved {len(embeddings)} embeddings to {FAISS_INDEX_FILE} and metadata to {METADATA_FILE}")

# # ----------------- QUERY TRANSACTIONS -----------------
# print(f"\nQuerying {len(transaction)} transaction rows...")
# index = faiss.read_index(FAISS_INDEX_FILE)
# with open(METADATA_FILE, "r") as f:
#     metadata = json.load(f)

# itemcodes = list(metadata.keys())

# for _, row in tqdm(transaction.iterrows(), total=len(transaction), desc="Transaction Queries"):
#     query_text = " ".join(str(val) for val in row if pd.notna(val))
#     query_emb = get_embedding(query_text)
    
#     if query_emb is not None:
#         query_np = np.array([query_emb]).astype('float32')
#         distances, indices = index.search(query_np, k=3)  # top 3 matches
        
#         print("\nðŸ”¹ Transaction Query:")
#         print(query_text)
#         print("Top Matches:")
#         for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
#             itemcode = itemcodes[idx]
#             print(f" {rank+1}. Itemcode: {itemcode}, Distance: {dist:.4f}")
#             print(" Metadata:", metadata[itemcode])
#     else:
#         print(f"Failed to embed transaction row: {query_text}")


# ----------------- BUILD FAISS GPU INDEX -----------------
embedding_dim = embeddings_np.shape[1]
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(embedding_dim)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(embeddings_np)

index_cpu = faiss.index_gpu_to_cpu(index)
faiss.write_index(index_cpu, FAISS_INDEX_FILE)
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved {len(embeddings)} embeddings to {FAISS_INDEX_FILE} and metadata to {METADATA_FILE}")

# ----------------- QUERY TRANSACTIONS AND SAVE -----------------
print(f"\nQuerying {len(transaction)} transaction rows...")

index_cpu = faiss.read_index(FAISS_INDEX_FILE)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

itemcodes = list(metadata.keys())

results_list = []

for _, row in tqdm(transaction.iterrows(), total=10, desc="Transaction Queries"):
    query_text = " ".join(str(val) for val in row if pd.notna(val))
    query_emb = get_embedding(query_text)
    
    if query_emb is not None:
        query_np = np.array([query_emb]).astype('float32')
        distances, indices = index.search(query_np, k=3)
        
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            itemcode = itemcodes[idx]
            metadata_item = metadata[itemcode]
            # Flatten metadata dictionary into key=value strings for CSV
            metadata_str = "; ".join([f"{k}={v}" for k, v in metadata_item.items()])
            results_list.append({
                "transaction_text": query_text,
                "rank": rank + 1,
                "matched_itemcode": itemcode,
                "distance": dist,
                "metadata": metadata_str
            })
    else:
        results_list.append({
            "transaction_text": query_text,
            "rank": None,
            "matched_itemcode": None,
            "distance": None,
            "metadata": None
        })

# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved transaction matches to {OUTPUT_CSV}")