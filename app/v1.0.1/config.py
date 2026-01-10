import os

class Config:
    # Google Gemini API Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBvfBBTystPNLsuh838Rebl6V5OZqWHllM")
    EMBEDDING_MODEL = "text-embedding-004"  # Google's embedding model
    LLM_MODEL = "gemini-2.5-flash-lite"  # Google's LLM for re-ranking
    
    # Search parameters
    TOP_K_MATCHES = 3
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 50

    # File paths
    master_file_path = './dataset/master.csv'
    # transaction_file_path = './dataset/transaction/dec-24.csv'
    # transaction_name = os.path.basename(transaction_file_path).split('.')[0]

    OUTPUT_FOLDER = "./temp"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    FAISS_INDEX_FILE = "./store/master_index.faiss"
    METADATA_FILE = "./store/metadata.json"
    # OUTPUT_CSV = f"./temp/{transaction_name}_transaction_matches.csv"
    # FINAL_OUTPUT_CSV = f"./temp/{transaction_name}_final_matches.csv"

    # Columns
    m_columns = ['itemcode', 'catcode', 'category', 'subcat', 'ssubcat', 'company', 'mbrand', 'brand', 'sku', 
                 'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']
    t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']