import numpy as np
import pandas as pd
import faiss
import ollama
import json
from tqdm import tqdm
from config import Config

class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config

    def get_embedding(self, text: str):
        """Call local Ollama client to get embedding for a given text."""
        try:
            response = ollama.embeddings(model=self.config.MODEL_NAME, prompt=text)
            return np.array(response["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"|ERROR| Error embedding text: {e}")
            return None

    def build_master_index(self):
        """Generate embeddings, build FAISS index, and save metadata."""
        master = pd.read_csv(self.config.master_file_path)[self.config.m_columns]

        texts, itemcodes, metadata, embeddings = [], [], {}, []
        print(f"|INFO| Generating embeddings for {len(master)} master rows...")

        for _, row in tqdm(master.iterrows(), total=len(master), desc="Master Embeddings"):
            itemcode = str(row["itemcode"])
            itemcodes.append(itemcode)

            row_text = " ".join(str(val) for col, val in row.items() if col != "itemcode" and pd.notna(val))
            texts.append(row_text)

            metadata[itemcode] = row.drop("itemcode").to_dict()

        for text in tqdm(texts, desc="Fetching Embeddings"):
            emb = self.get_embedding(text)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(1536, dtype=np.float32))  # adjust if dimension differs

        embeddings_np = np.vstack(embeddings).astype('float32')

        # Build FAISS index
        embedding_dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)

        faiss.write_index(index, self.config.FAISS_INDEX_FILE)
        with open(self.config.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"|INFO| Saved {len(embeddings)} embeddings to {self.config.FAISS_INDEX_FILE} "
              f"and metadata to {self.config.METADATA_FILE}")
