# Imports
import re
import os
import numpy as np
import pandas as pd
import json
from google import genai
from google.genai import types

# Configuration
gemini_api_key = "AIzaSyDJ7SIlL-NPVIVinMVNjkCb2mKMRv6g0Xg"
client = genai.Client(api_key=gemini_api_key)

model_name = "gemini-3-flash-preview"
gen_config = types.GenerateContentConfig(
    temperature=0.1,
    top_p=0.9,
    max_output_tokens=50,
    response_mime_type="application/json"
)
chunk_size = 10
max_iterations = 10

master_file_path = "./redundant/data_master.csv"
m_columns = ['itemcode', 'catcode', 'category', 'subcat', 'ssubcat', 'company', 'mbrand', 'brand', 'sku',
             'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']

transaction_file_path = "./redundant/data_transaction.csv"
t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']

# Data loader
master_file = pd.read_csv(master_file_path, usecols=m_columns)
master_dict = master_file.to_dict(orient='index')
master_file.head()

transaction_file = pd.read_csv(transaction_file_path, usecols=t_columns)
transaction_dict = transaction_file.to_dict(orient='index')
transaction_file.head()

# Format data for prompt generation
def format_master(master_row):
    return f"""
Item Code: {master_row['itemcode']}
Category Code: {master_row['catcode']}
Category: {master_row['category']}
Subcategory: {master_row['subcat']}
Sub-Subcategory: {master_row['ssubcat']}
Company: {master_row['company']}
Main Brand: {master_row['mbrand']}
Brand: {master_row['brand']}
Pack Type: {master_row['packtype']}
Pack Size: {master_row['base_pack']}
Flavor: {master_row['flavor']}
Color: {master_row['color']}
Unit of Measure: {master_row['uom']}
MRP: {master_row['mrp']}
    """

def format_transaction(transaction_row):
    return f"""
Category Code: {transaction_row['CATEGORY']}
Company: {transaction_row['MANUFACTURE']}
Brand: {transaction_row['BRAND']}
Item Description: {transaction_row['ITEMDESC']}
MRP: {transaction_row['MRP']}
Pack Size: {re.match(r'(\d+)\s*(.*)', transaction_row['PACKSIZE']).group(1)}
Unit of Measure: {re.match(r'(\d+)\s*(.*)', transaction_row['PACKSIZE']).group(2)}
Pack Type: {transaction_row['PACKTYPE']}
    """

# Format context data for LLM prompt
def format_context_matches(master_dictionary) -> str:
    """Format context matches for the LLM prompt"""
    context_lines = []
    
    for idx, master_row in master_dictionary.items():
        formatted_row = format_master(master_row)
        context_lines.append(f"Context item {idx}:{formatted_row}")
    return "\n".join(context_lines)

# Format query data for LLM prompt
def format_query_match(transaction_dictionary) -> str:
    """Format context matches for the LLM prompt"""
    context_lines = []
    
    for idx, master_row in transaction_dictionary.items():
        formatted_row = format_transaction(master_row)
        context_lines.append(f"Transaction item {idx}:{formatted_row}")
    return "\n".join(context_lines)

# Call LLM
def call_llm(prompt):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        # config=gen_config
    )

    print("Gemini response:", response)
    return response

# Generate prompt
def generate_prompt(context_rows, transaction_row):
  prompt = f"""
You are an expert product matching AI.
You have been given a transaction item and a set of context items from the master catalog.
Determine which context item best matches the transaction item.

Here are the context items from the master catalog:
\n{format_context_matches(context_rows)}

Here is the transaction item to match:
\n{format_query_match(transaction_row)}

Matching criteria (Priority Order):
1. Category code alignment
2. EXACT company match
3. EXACT brand match
4. EXACT pack size and pack type match
5. MRP/price similarity

Instructions:
1. Compare transaction item with each context item carefully
2. Prioritize exact matches in category code, company, brand, pack size, and pack type
3. Consider MRP similarity as a secondary factor

Based on the attributes provided, identify the best matching context item for the transaction item.
Respond strictly with a JSON object in the following format:
{{
  "context_item": "<The context item number, for example, 0 if 'Context item 0'>",
  "score": "<The confidence score normalized between 0 and 1, with 1 being a perfect match>",
}}
  """
  return prompt

# Tournament style elimination
def reduce_once(
    candidates: dict,
    transaction_row: dict,
    chunk_size: int
) -> dict:
    """
    candidates: {absolute_id: master_row}
    returns: reduced {absolute_id: master_row}
    """

    candidate_ids = list(candidates.keys())
    reduced = {}

    start = 0
    while start < len(candidate_ids):
        end = min(start + chunk_size, len(candidate_ids))
        chunk_ids = candidate_ids[start:end]

        context_rows = {
            i: candidates[cid]
            for i, cid in enumerate(chunk_ids)
        }

        prompt = generate_prompt(context_rows, transaction_row)
        response = call_llm(prompt)

        try:
            parsed = json.loads(response)
            chunk_idx = int(parsed["context_item"])
            winner_id = chunk_ids[chunk_idx]

            reduced[winner_id] = candidates[winner_id]

        except (json.JSONDecodeError, KeyError, ValueError, IndexError):
            pass

        start += chunk_size

    return reduced

def tournament_match(
    master_dict: dict,
    transaction_row: dict,
    chunk_size: int = chunk_size,
    max_iterations: int = max_iterations
):
    candidates = master_dict
    iteration = 1

    while len(candidates) > 1 and iteration <= max_iterations:
        print(
            f"Iteration {iteration}: "
            f"{len(candidates)} candidates → "
            f"{(len(candidates) + chunk_size - 1) // chunk_size} chunks"
        )

        candidates = reduce_once(
            candidates=candidates,
            transaction_row=transaction_row,
            chunk_size=chunk_size
        )

        iteration += 1

    return candidates

def main():
    final_candidate = tournament_match(
        master_dict=master_dict,
        transaction_row={0: transaction_dict[0]},
        chunk_size=chunk_size,
        max_iterations=max_iterations
    )

    print("Final match:", final_candidate)

if __name__ == "__main__":
    main()