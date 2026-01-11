import csv
import json
from typing import List, Dict, Any
import requests


# ==========================
# CONFIGURATION
# ==========================

MASTER_CHUNK_SIZE = 10
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

GENERATION_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_predict": 50
}

m_columns = ['itemcode', 'catcode', 'category', 'subcat', 'ssubcat', 'company', 'mbrand', 'brand', 'sku',
             'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']
t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']

# ==========================
# DATA LOADING
# ==========================

def load_csv(filepath: str, columns: List[str]) -> List[Dict[str, Any]]:
    print(f"|INFO| Loading CSV file: {filepath}")
    set_columns = set(columns)
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            clean_row = {
                k.strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
                if k.strip() in set_columns
            }
            data.append(clean_row)
    print(f"|INFO| Loaded {len(data)} rows from {filepath}")
    return data


# ==========================
# PROMPT ENGINEERING
# ==========================

def build_prompt(transaction: Dict[str, Any],
                 master_rows: Dict[str, Dict[str, Any]]) -> str:
    return f"""
You are a data-matching engine.

TASK:
- Compare the given TRANSACTION row against ALL MASTER rows.
- Identify the SINGLE closest MASTER row.
- Use ALL fields from both transaction and master rows.

RULES:
- Return ONLY valid JSON.
- No explanations.
- No extra keys.
- Score must be between 0 and 1.
- Choose exactly ONE row.

OUTPUT FORMAT:
{{
  "row_num": "<row_num>",
  "score": <float>
}}

TRANSACTION ROW:
{json.dumps(transaction, indent=2)}

MASTER ROWS:
{json.dumps(master_rows, indent=2)}
""".strip()


# ==========================
# LLM CALL (OLLAMA)
# ==========================

def query_llm(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": GENERATION_OPTIONS
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw = response.json().get("response", "")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("|WARNING| LLM returned invalid JSON. Raw output:")
        print(raw)
        raise


# ==========================
# CHUNKING LOGIC
# ==========================

def chunk_master(master_rows: List[Dict[str, Any]],
                 chunk_size: int) -> List[Dict[str, Dict[str, Any]]]:
    chunks = []
    total = len(master_rows)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = {
            str(i): master_rows[i]
            for i in range(start, end)
        }
        chunks.append(chunk)

    return chunks


# ==========================
# HIERARCHICAL REDUCTION
# ==========================

def hierarchical_match(transaction: Dict[str, Any],
                        master_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    current_candidates = master_rows
    iteration = 1

    while len(current_candidates) > 1:
        print(f"|INFO| Iteration {iteration} | Candidates: {len(current_candidates)}")

        chunks = chunk_master(current_candidates, MASTER_CHUNK_SIZE)
        winners = []

        for idx, chunk in enumerate(chunks):
            print(f"|INFO| Processing chunk {idx + 1}/{len(chunks)}")

            prompt = build_prompt(transaction, chunk)
            result = query_llm(prompt)

            row_num = int(result["row_num"])
            score = float(result["score"])

            winners.append({
                "row_num": row_num,
                "score": score,
                "data": current_candidates[row_num]
            })

        current_candidates = [w["data"] for w in winners]
        iteration += 1

    print("|OUTPUT| Final best match found")
    return {
        "row_num": 0,
        "score": 1.0,
        "data": current_candidates[0]
    }


# ==========================
# MAIN PIPELINE
# ==========================

def main():
    master_data = load_csv("./redundant/master.csv", m_columns)
    transaction_data = load_csv("./redundant/transaction.csv", t_columns)

    results = []

    for idx, transaction in enumerate(transaction_data):
        print(f"\n|INFO| Processing transaction {idx + 1}/{len(transaction_data)}")
        best_match = hierarchical_match(transaction, master_data)
        results.append({
            "transaction_index": idx,
            "best_match": best_match
        })

    print("\n|OUTPUT| Matching complete for all transactions")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
