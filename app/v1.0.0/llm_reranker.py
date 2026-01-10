import pandas as pd
import ollama
import json
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMReranker:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        Standalone LLM Re-ranker for transaction matches
        
        Recommended models for your hardware (11GB VRAM / 16GB RAM):
        1. qwen2.5:7b     - Best for structured output (RECOMMENDED)
        2. llama3.1:8b    - Best overall reasoning  
        3. mistral:7b     - Fastest processing
        4. gemma2:9b      - Best logical reasoning
        5. phi3:medium    - Most memory efficient
        """
        self.model_name = model_name
        self.client = ollama
        
    def create_prompt_template(self) -> str:
        """Create the LLM prompt template"""
        return """You are an expert product matching system. Your task is to find the best match between a transaction record and suggested master records.

TRANSACTION RECORD:
Date: {t_date}
Store Code: {t_storecode}  
Item Code: {t_itemcode}
New Code: {t_new_codes}
Category: {t_category}
Manufacturer: {t_manufacture}
Brand: {t_brand}
Description: {t_itemdesc}
Pack Size: {t_packsize}
Pack Type: {t_packtype}
MRP: {t_mrp}

SUGGESTED MATCHES:
{context_matches}

MATCHING CRITERIA (Priority Order):
1. EXACT manufacturer/company match (HIGHEST PRIORITY)
2. EXACT brand match  
3. EXACT pack size and pack type match
4. Category alignment
5. MRP/price similarity
6. Semantic similarity score (lower distance = better)

INSTRUCTIONS:
- Compare transaction record with each suggested match carefully
- Prioritize exact matches in manufacturer, brand, and pack specifications
- Consider semantic similarity scores (lower distance values are better matches)
- Choose the rank number of the BEST overall match
- If NO match seems reasonable, return "0"

RESPOND WITH JSON ONLY:
{{"match_rank": "X"}}

Where X is the rank number (1, 2, 3, etc.) or "0" for no good match."""

    def format_context_matches(self, matches_group: pd.DataFrame) -> str:
        """Format context matches for the LLM prompt"""
        context_lines = []
        
        for _, row in matches_group.iterrows():
            rank = row.get('rank', 'N/A')
            itemcode = row.get('matched_itemcode', 'N/A')
            distance = row.get('distance', 'N/A')
            
            # Format distance for readability
            try:
                distance_str = f"{float(distance):.2f}" if distance != 'N/A' else 'N/A'
            except:
                distance_str = str(distance)
            
            # Master record details
            company = row.get('m_company', 'N/A')
            brand = row.get('m_brand', 'N/A')
            category = row.get('m_category', 'N/A')
            subcat = row.get('m_subcat', 'N/A')
            packtype = row.get('m_packtype', 'N/A')
            base_pack = row.get('m_base_pack', 'N/A')
            mrp = row.get('m_mrp', 'N/A')
            sku = row.get('m_sku', 'N/A')
            
            context_line = f"""RANK {rank}:
  Item Code: {itemcode} | Similarity Score: {distance_str}
  Manufacturer: {company}
  Brand: {brand}
  Category: {category} → {subcat}
  Pack Type: {packtype}
  Pack Size: {base_pack}
  SKU: {sku}
  MRP: {mrp}"""
            
            context_lines.append(context_line)
        
        return "\n\n".join(context_lines)
    
    def extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract match_rank from LLM response"""
        try:
            # Look for JSON pattern
            json_match = re.search(r'\{[^}]*"match_rank"\s*:\s*"([^"]*)"[^}]*\}', response_text)
            if json_match:
                return json_match.group(1)
            
            # Look for just the rank value
            rank_match = re.search(r'"match_rank"\s*:\s*"([^"]*)"', response_text)
            if rank_match:
                return rank_match.group(1)
                
            # Try to parse as full JSON
            json_obj = json.loads(response_text.strip())
            return str(json_obj.get('match_rank', '0'))
            
        except:
            # Last resort: look for any number
            number_match = re.search(r'["\']?(\d+)["\']?', response_text)
            if number_match:
                return number_match.group(1)
            
            logger.warning(f"Could not extract rank from response: {response_text[:100]}...")
            return None
    
    def query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM and return selected rank"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,      # Low temperature for consistency
                    "top_p": 0.9,
                    "num_predict": 50,       # Short response expected
                    "stop": ["\n\n", "EXPLANATION", "REASONING"]  # Stop early
                }
            )
            
            response_text = response['response'].strip()
            return self.extract_json_from_response(response_text)
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return None
    
    def rerank_csv_matches(self, csv_file_path: str, output_file_path: str) -> pd.DataFrame:
        """
        Re-rank matches from existing CSV output
        
        Args:
            csv_file_path: Path to your transaction_matches.csv file
            output_file_path: Path where to save the re-ranked results
            
        Returns:
            DataFrame with LLM re-ranked results
        """
        # Load existing matches
        logger.info(f"Loading matches from {csv_file_path}")
        matches_df = pd.read_csv(csv_file_path)
        
        if matches_df.empty:
            logger.warning("No matches found in CSV file")
            return matches_df
        
        logger.info(f"Processing {len(matches_df)} match records...")
        
        # # Group by transaction - using the key transaction identifiers
        # # Adjust these columns based on what uniquely identifies a transaction
        # group_cols = []
        # for col in ['t_ITEMCODE', 't_NEW_CODES', 't_STORECODE', 't_DATE']:
        #     if col in matches_df.columns:
        #         group_cols.append(col)
        
        # if not group_cols:
        #     logger.error("Cannot find transaction grouping columns")
        #     return matches_df
        
        # logger.info(f"Grouping transactions by: {group_cols}")
        # grouped = matches_df.groupby(group_cols)

        # ✅ FIX: group by all transaction-prefixed columns to avoid dropping rows
        group_cols = [col for col in matches_df.columns if col.startswith("t_")]

        if not group_cols:
            logger.error("Cannot find transaction grouping columns")
            return matches_df
        
        logger.info(f"Grouping transactions by: {group_cols}")
        grouped = matches_df.groupby(group_cols, dropna=False)
        
        reranked_results = []
        
        for group_key, group_df in tqdm(grouped, desc="LLM Re-ranking"):
            # Get transaction details from first row
            t_row = group_df.iloc[0]
            
            # Create the LLM prompt
            prompt = self.create_prompt_template().format(
                t_date=t_row.get('t_DATE', 'N/A'),
                t_storecode=t_row.get('t_STORECODE', 'N/A'), 
                t_itemcode=t_row.get('t_ITEMCODE', 'N/A'),
                t_new_codes=t_row.get('t_NEW_CODES', 'N/A'),
                t_category=t_row.get('t_CATEGORY', 'N/A'),
                t_manufacture=t_row.get('t_MANUFACTURE', 'N/A'),
                t_brand=t_row.get('t_BRAND', 'N/A'),
                t_itemdesc=t_row.get('t_ITEMDESC', 'N/A'),
                t_packsize=t_row.get('t_PACKSIZE', 'N/A'),
                t_packtype=t_row.get('t_PACKTYPE', 'N/A'),
                t_mrp=t_row.get('t_MRP', 'N/A'),
                context_matches=self.format_context_matches(group_df)
            )
            
            # Query LLM
            selected_rank = self.query_llm(prompt)
            
            if selected_rank:
                try:
                    rank_num = int(selected_rank)
                    
                    if rank_num == 0:
                        # No match - take first row but mark as rejected
                        best_row = group_df.iloc[0].copy()
                        best_row['llm_rank'] = 0
                        best_row['llm_selected'] = False
                        best_row['llm_confidence'] = 'rejected'
                        reranked_results.append(best_row)
                        
                    else:
                        # Find the selected rank
                        selected_matches = group_df[group_df['rank'] == rank_num]
                        
                        if not selected_matches.empty:
                            best_row = selected_matches.iloc[0].copy()
                            best_row['llm_rank'] = rank_num
                            best_row['llm_selected'] = True
                            best_row['llm_confidence'] = 'high'
                            reranked_results.append(best_row)
                        else:
                            # Fallback to best available rank
                            best_row = group_df.iloc[0].copy()
                            best_row['llm_rank'] = -1
                            best_row['llm_selected'] = False
                            best_row['llm_confidence'] = 'fallback'
                            reranked_results.append(best_row)
                            
                except ValueError:
                    # Fallback for parsing errors
                    best_row = group_df.iloc[0].copy()
                    best_row['llm_rank'] = -1
                    best_row['llm_selected'] = False
                    best_row['llm_confidence'] = 'parse_error'
                    reranked_results.append(best_row)
            else:
                # LLM query failed - use fallback
                best_row = group_df.iloc[0].copy()
                best_row['llm_rank'] = -1
                best_row['llm_selected'] = False
                best_row['llm_confidence'] = 'llm_failed'
                reranked_results.append(best_row)
        
        # Convert to DataFrame
        reranked_df = pd.DataFrame(reranked_results)
        
        # Save results
        reranked_df.to_csv(output_file_path, index=False)
        logger.info(f"Saved LLM re-ranked results to {output_file_path}")
        
        # Print statistics
        self._print_reranking_stats(reranked_df)
        
        return reranked_df
    
    def _print_reranking_stats(self, df: pd.DataFrame):
        """Print re-ranking statistics"""
        if 'llm_selected' in df.columns:
            total = len(df)
            selected = df['llm_selected'].sum()
            rejected = (df['llm_rank'] == 0).sum()
            failed = (df['llm_confidence'] == 'llm_failed').sum()
            
            logger.info("\n" + "="*50)
            logger.info("LLM RE-RANKING SUMMARY")
            logger.info("="*50)
            logger.info(f"Total transactions processed: {total}")
            logger.info(f"Successfully re-ranked: {selected} ({selected/total*100:.1f}%)")
            logger.info(f"Rejected (no good match): {rejected} ({rejected/total*100:.1f}%)")  
            logger.info(f"LLM query failures: {failed} ({failed/total*100:.1f}%)")
            logger.info("="*50)