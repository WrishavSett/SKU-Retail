import numpy as np
import pandas as pd
import faiss
import json
from tqdm import tqdm
from config import Config
from google import genai

class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize Google Gemini client
        if not self.config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
        
        self.client = genai.Client(api_key=self.config.GEMINI_API_KEY)

    def get_embedding(self, text: str):
        """Call Google Gemini API to get embedding for a given text."""
        try:
            response = self.client.models.embed_content(
                model=self.config.EMBEDDING_MODEL,
                contents=text
            )
            # Extract embedding from response
            embedding = response.embeddings[0].values
            return np.array(embedding, dtype=np.float32)
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

        embedding_dim = None
        pending_fallbacks = 0   # count of rows failed before first success
        
        for text in tqdm(texts, desc="Fetching Embeddings"):
            emb = self.get_embedding(text)
            if emb is not None:
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
                    # Fix all previously failed rows with zero vectors
                    embeddings.extend([np.zeros(embedding_dim, dtype=np.float32) for _ in range(pending_fallbacks)])
                embeddings.append(emb)
            else:
                if embedding_dim is None:
                    pending_fallbacks += 1
                else:
                    embeddings.append(np.zeros(embedding_dim, dtype=np.float32))

        if embedding_dim is None:
            raise ValueError("|ERROR| No embeddings could be generated from the master data.")

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