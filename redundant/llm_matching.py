import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
from google import genai


def load_data(master_path: str, transaction_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load master and transaction data from CSV files.
    
    Args:
        master_path: Path to the master CSV file
        transaction_path: Path to the transaction CSV file
        
    Returns:
        Tuple of (master_df, transaction_df)
    """
    print(f"|INFO| Loading master file from: {master_path}")
    master_df = pd.read_csv(master_path)
    print(f"|INFO| Master file loaded: {len(master_df)} rows")
    
    print(f"|INFO| Loading transaction file from: {transaction_path}")
    transaction_df = pd.read_csv(transaction_path)
    print(f"|INFO| Transaction file loaded: {len(transaction_df)} rows")
    
    return master_df, transaction_df


def prepare_master_rows_json(master_df: pd.DataFrame, start_idx: int, batch_size: int) -> Dict:
    """
    Prepare a batch of master rows in JSON format with row_num as keys.
    
    Args:
        master_df: Master dataframe
        start_idx: Starting index for the batch
        batch_size: Number of rows to include in the batch
        
    Returns:
        Dictionary with row_num as keys and row data as values
    """
    end_idx = min(start_idx + batch_size, len(master_df))
    batch_data = {}
    
    for i in range(start_idx, end_idx):
        row_data = master_df.iloc[i].to_dict()
        batch_data[str(i)] = row_data
    
    return batch_data


def create_matching_prompt(transaction_row: Dict, master_rows: Dict) -> str:
    """
    Create a prompt for the LLM to find the closest match.
    
    Args:
        transaction_row: Dictionary containing transaction row data
        master_rows: Dictionary of master rows with row_num as keys
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert at matching product transaction records to master product database entries.

TRANSACTION RECORD TO MATCH:
{json.dumps(transaction_row, indent=2)}

MASTER DATABASE ENTRIES (candidate matches):
{json.dumps(master_rows, indent=2)}

TASK:
Analyze the transaction record and find the ONE closest matching product from the master database entries provided above.

MATCHING CRITERIA (in order of importance):
1. Product brand and name similarity
2. Pack size and pack type (e.g., PET, CAN, HL)
3. Category alignment
4. Manufacturer/company name
5. MRP (price) similarity if available
6. Flavor or variant matching

SCORING GUIDANCE:
- 1.0: Perfect match (all key attributes align)
- 0.8-0.9: Very strong match (brand, size, type match; minor differences)
- 0.6-0.7: Good match (brand and category match; some attribute differences)
- 0.4-0.5: Moderate match (category match; significant differences in specifics)
- 0.2-0.3: Weak match (some similarity but major differences)
- 0.0-0.1: Poor match (minimal similarity)

RESPONSE FORMAT:
You must respond with ONLY a valid JSON object in this exact format (no markdown, no code blocks, no additional text):
{{"row_num": "X", "score": 0.XX}}

Where:
- row_num: The row number (as a string) of the best matching master entry
- score: A number between 0 and 1 representing match confidence

Example response:
{{"row_num": "42", "score": 0.85}}

Provide your response now:"""
    
    return prompt


def query_llm(prompt: str, client: genai.Client, model: str = "gemma-3-12b-it") -> Dict:
    """
    Query the Google Gemini API with the matching prompt.
    
    Args:
        prompt: The formatted prompt string
        client: Google GenAI client instance
        model: Model name to use (default: gemma-3-12b-it)
        
    Returns:
        Dictionary containing row_num and score
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            # Remove first line with ```json or ```
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:])
            # Remove closing ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        return result
        
    except Exception as e:
        print(f"|WARNING| Error querying LLM: {str(e)}")
        print(f"|WARNING| Response text: {response_text if 'response_text' in locals() else 'N/A'}")
        return {"row_num": "-1", "score": 0.0}


def find_best_matches_iteration(
    transaction_row: Dict,
    master_df: pd.DataFrame,
    candidate_indices: List[int],
    batch_size: int,
    client: genai.Client,
    model: str = "gemma-3-12b-it"
) -> List[Tuple[int, float]]:
    """
    Run one iteration of matching: batch through candidates and find best matches.
    
    Args:
        transaction_row: Transaction row to match
        master_df: Master dataframe
        candidate_indices: List of master row indices to consider
        batch_size: Number of rows per batch
        client: Google GenAI client instance
        model: Model name to use
        
    Returns:
        List of tuples (row_index, score) sorted by score descending
    """
    matches = []
    total_batches = (len(candidate_indices) + batch_size - 1) // batch_size
    
    print(f"|INFO| Processing {len(candidate_indices)} candidates in {total_batches} batches of {batch_size}")
    
    for batch_num in range(total_batches):
        start_pos = batch_num * batch_size
        end_pos = min(start_pos + batch_size, len(candidate_indices))
        
        batch_indices = candidate_indices[start_pos:end_pos]
        
        # Create a mapping of local batch positions to actual master indices
        batch_master_rows = {}
        for local_idx, actual_idx in enumerate(batch_indices):
            row_data = master_df.iloc[actual_idx].to_dict()
            batch_master_rows[str(local_idx)] = row_data
        
        # Query LLM
        prompt = create_matching_prompt(transaction_row, batch_master_rows)
        result = query_llm(prompt, client, model)
        
        # Map back to actual master index
        local_row_num = int(result["row_num"])
        if local_row_num >= 0 and local_row_num < len(batch_indices):
            actual_row_idx = batch_indices[local_row_num]
            score = float(result["score"])
            matches.append((actual_row_idx, score))
            
            print(f"|INFO| Batch {batch_num + 1}/{total_batches}: Best match row {actual_row_idx} with score {score:.3f}")
        else:
            print(f"|WARNING| Batch {batch_num + 1}/{total_batches}: Invalid row_num returned")
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches


def match_single_transaction(
    transaction_row: Dict,
    master_df: pd.DataFrame,
    batch_size: int,
    client: genai.Client,
    model: str = "gemma-3-12b-it"
) -> Tuple[int, float]:
    """
    Match a single transaction row to the best master record through iterative refinement.
    
    Args:
        transaction_row: Transaction row to match
        master_df: Master dataframe
        batch_size: Number of rows per batch in first iteration
        client: Google GenAI client instance
        model: Model name to use
        
    Returns:
        Tuple of (best_match_index, best_match_score)
    """
    print(f"|INFO| Starting matching process for transaction")
    
    # Iteration 1: Process all master rows in batches
    print(f"|INFO| === ITERATION 1: Processing all {len(master_df)} master rows ===")
    all_indices = list(range(len(master_df)))
    iteration_1_matches = find_best_matches_iteration(
        transaction_row, master_df, all_indices, batch_size, client, model
    )
    
    # Take top N matches (where N is determined by batch size)
    top_n = min(batch_size, len(iteration_1_matches))
    top_candidates = [match[0] for match in iteration_1_matches[:top_n]]
    print(f"|OUTPUT| Iteration 1 complete: Top {len(top_candidates)} candidates selected")
    
    # Iteration 2: Narrow down to top 2
    print(f"|INFO| === ITERATION 2: Narrowing to top 2 from {len(top_candidates)} candidates ===")
    iteration_2_matches = find_best_matches_iteration(
        transaction_row, master_df, top_candidates, max(2, len(top_candidates) // 2), client, model
    )
    
    top_2_candidates = [match[0] for match in iteration_2_matches[:2]]
    print(f"|OUTPUT| Iteration 2 complete: Top 2 candidates selected")
    
    # Iteration 3: Final match between top 2
    print(f"|INFO| === ITERATION 3: Final selection from top 2 candidates ===")
    final_matches = find_best_matches_iteration(
        transaction_row, master_df, top_2_candidates, 2, client, model
    )
    
    if final_matches:
        best_match_idx, best_match_score = final_matches[0]
        print(f"|OUTPUT| Final best match: Row {best_match_idx} with score {best_match_score:.3f}")
        return best_match_idx, best_match_score
    else:
        print(f"|WARNING| No valid match found")
        return -1, 0.0


def process_all_transactions(
    master_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    batch_size: int,
    client: genai.Client,
    model: str = "gemma-3-12b-it",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Process all transactions and find best matches from master data.
    
    Args:
        master_df: Master dataframe
        transaction_df: Transaction dataframe
        batch_size: Number of rows per batch in first iteration
        client: Google GenAI client instance
        model: Model name to use
        output_path: Optional path to save results
        
    Returns:
        DataFrame with matching results
    """
    results = []
    total_transactions = len(transaction_df)
    
    print(f"|INFO| Starting processing of {total_transactions} transactions")
    print(f"|INFO| Batch size for iteration 1: {batch_size}")
    print(f"|INFO| Using model: {model}")
    
    for idx, transaction_row in transaction_df.iterrows():
        print(f"\n|INFO| {'='*80}")
        print(f"|INFO| Processing transaction {idx + 1}/{total_transactions}")
        print(f"|INFO| {'='*80}")
        
        transaction_dict = transaction_row.to_dict()
        
        best_match_idx, best_match_score = match_single_transaction(
            transaction_dict, master_df, batch_size, client, model
        )
        
        # Prepare result record
        result = {
            'transaction_idx': idx,
            'transaction_itemcode': transaction_row.get('ITEMCODE'),
            'transaction_brand': transaction_row.get('BRAND'),
            'transaction_itemdesc': transaction_row.get('ITEMDESC'),
            'matched_master_idx': best_match_idx,
            'matched_itemcode': master_df.iloc[best_match_idx]['itemcode'] if best_match_idx >= 0 else None,
            'matched_brand': master_df.iloc[best_match_idx]['brand'] if best_match_idx >= 0 else None,
            'matched_sku': master_df.iloc[best_match_idx]['sku'] if best_match_idx >= 0 else None,
            'match_score': best_match_score
        }
        
        results.append(result)
        
        print(f"|OUTPUT| Transaction {idx + 1} completed: Match score = {best_match_score:.3f}\n")
    
    results_df = pd.DataFrame(results)
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"|INFO| Results saved to: {output_path}")
    
    return results_df


def main():
    """
    Main function to orchestrate the product matching process.
    """
    print("|INFO| Starting Product Matching System")
    print("|INFO| " + "="*80)
    
    # Configuration
    MASTER_FILE = "./dataset/master.csv"  # Update with your file path
    TRANSACTION_FILE = "./dataset/transaction.csv"  # Update with your file path
    BATCH_SIZE = 100  # Number of master rows per batch in first iteration
    MODEL_NAME = "gemma-3-12b-it"  # Can also use "gemini-1.5-flash", "gemini-1.5-pro", etc.
    OUTPUT_FILE = f"./temp/transaction_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Get API key from environment variable
    # api_key = os.environ.get("GOOGLE_API_KEY")
    api_key = "AIzaSyCZSxPpmHec8OonCFPjp0gc6NmZfd133PI"
    if not api_key:
        print("|WARNING| GOOGLE_API_KEY not found in environment variables")
        print("|INFO| Please set your API key: export GOOGLE_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize Google GenAI client
        print(f"|INFO| Initializing Google GenAI client with model: {MODEL_NAME}")
        client = genai.Client(api_key=api_key)
        
        # Load data
        master_df, transaction_df = load_data(MASTER_FILE, TRANSACTION_FILE)
        
        # Process all transactions
        results_df = process_all_transactions(
            master_df, 
            transaction_df, 
            BATCH_SIZE, 
            client,
            MODEL_NAME,
            OUTPUT_FILE
        )
        
        # Summary statistics
        print("\n|INFO| " + "="*80)
        print("|OUTPUT| MATCHING PROCESS COMPLETE")
        print("|INFO| " + "="*80)
        print(f"|OUTPUT| Total transactions processed: {len(results_df)}")
        print(f"|OUTPUT| Average match score: {results_df['match_score'].mean():.3f}")
        print(f"|OUTPUT| Matches with score > 0.8: {(results_df['match_score'] > 0.8).sum()}")
        print(f"|OUTPUT| Matches with score > 0.6: {(results_df['match_score'] > 0.6).sum()}")
        print(f"|OUTPUT| Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"|WARNING| Fatal error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()