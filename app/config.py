import os

class Config:
    # Model
    MODEL_NAME = "nomic-embed-text"

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
    m_columns = ['itemcode', 'catcode', 'company', 'mbrand', 'brand', 'sku', 
                 'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp'] # category, subcat, ssubcat
    t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']
