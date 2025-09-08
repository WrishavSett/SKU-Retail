# SKU Matching System

A sophisticated product matching pipeline that uses semantic embeddings, rule-based filtering, and LLM re-ranking to match transaction records with master product data.

## 🏗️ Architecture Overview

The system implements a 3-stage matching pipeline:

1. **Semantic Search**: FAISS-based similarity matching using embeddings
2. **Rule-based Filtering**: Domain-specific logic (pack size matching)
3. **LLM Re-ranking**: Intelligent final selection with business priorities

```
Transaction Data → Embedding Pipeline → FAISS Index → Transaction Matcher → LLM Reranker → Final Results
```

## 📁 Project Structure

```
SKU(R)-ORG/
├── app/
│   ├── config.py              # Configuration settings
│   ├── embedding_pipeline.py  # FAISS index building
│   ├── transaction_matcher.py # Semantic matching logic
│   ├── llm_reranker.py       # LLM-based re-ranking
│   └── main.py               # Main execution pipeline
├── dataset/
│   ├── master.csv            # Master product database
│   ├── transaction.csv       # Sample transaction data
│   └── transaction/          # Monthly transaction files
│       ├── jan-22.csv
│       ├── feb-22.csv
│       └── ...
├── store/
│   ├── master_index.faiss    # FAISS vector index
│   └── metadata.json         # Product metadata
└── temp/                     # Output files
    ├── *_transaction_matches.csv
    └── *_llm_reranked.csv
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install pandas numpy faiss-cpu ollama tqdm

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# Pull required models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### Basic Usage

```bash
# Build FAISS index and process transactions
python main.py --build --folder ./dataset/transaction --use-llm

# Process a single file
python main.py --file ./dataset/transaction/dec-24.csv

# Re-rank existing results only
python main.py --llm-only --folder ./dataset/transaction
```

## 📋 Detailed Usage

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file` | Process single transaction CSV | `--file ./dataset/transaction/dec-24.csv` |
| `--folder` | Process all CSVs in folder | `--folder ./dataset/transaction` |
| `--build` | Force rebuild FAISS index | `--build` |
| `--use-llm` | Enable LLM re-ranking | `--use-llm` |
| `--llm-model` | Specify LLM model | `--llm-model mistral:7b` |
| `--llm-only` | Only run LLM re-ranking | `--llm-only` |

### Execution Modes

#### 1. Full Pipeline (Recommended)
```bash
python main.py --build --folder ./dataset/transaction --use-llm
```
- Builds/rebuilds FAISS index
- Processes all transaction files
- Applies LLM re-ranking for best accuracy

#### 2. Semantic Matching Only
```bash
python main.py --folder ./dataset/transaction
```
- Fast processing without LLM
- Good for large batches or testing

#### 3. LLM Re-ranking Only
```bash
python main.py --llm-only --folder ./dataset/transaction --llm-model qwen2.5:7b
```
- Re-ranks existing results
- Useful for experimenting with different models

## 🔧 Configuration

### Data Schema

#### Master Data Columns (`master.csv`)
```python
m_columns = [
    'itemcode',    # Unique item identifier
    'catcode',     # Category code
    'category',    # Product category
    'subcat',      # Subcategory
    'ssubcat',     # Sub-subcategory
    'company',     # Manufacturer
    'mbrand',      # Master brand
    'brand',       # Product brand
    'sku',         # Stock keeping unit
    'packtype',    # Package type
    'base_pack',   # Base package size
    'flavor',      # Product flavor
    'color',       # Product color
    'wght',        # Weight
    'uom',         # Unit of measure
    'mrp'          # Maximum retail price
]
```

#### Transaction Data Columns
```python
t_columns = [
    'CATEGORY',    # Product category
    'MANUFACTURE', # Manufacturer name
    'BRAND',       # Brand name
    'ITEMDESC',    # Item description
    'MRP',         # Maximum retail price
    'PACKSIZE',    # Package size
    'PACKTYPE'     # Package type
]
```

### Model Configuration

#### Recommended LLM Models (11GB VRAM / 16GB RAM)
1. **qwen2.5:7b** - Best for structured output (RECOMMENDED)
2. **llama3.1:8b** - Best overall reasoning
3. **mistral:7b** - Fastest processing
4. **gemma2:9b** - Best logical reasoning
5. **phi3:medium** - Most memory efficient

## 📊 Output Files

### Transaction Matches (`*_transaction_matches.csv`)
Contains top-3 semantic matches for each transaction:
- All original transaction columns (prefixed with `t_`)
- Matched master data (prefixed with `m_`)
- `rank`: Match ranking (1-3)
- `matched_itemcode`: Matched item identifier
- `distance`: Semantic similarity score (lower = better)

### LLM Re-ranked (`*_llm_reranked.csv`)
Final results after LLM re-ranking:
- All transaction match data
- `llm_rank`: LLM selected rank
- `llm_selected`: Boolean selection flag
- `llm_confidence`: Confidence level (`high`, `rejected`, `fallback`, etc.)

## 🤖 How It Works

### 1. Embedding Pipeline
- Generates semantic embeddings using `nomic-embed-text`
- Builds FAISS index for fast similarity search
- Stores product metadata separately

### 2. Transaction Matching
- Converts transaction records to embeddings
- Retrieves top-3 similar products from FAISS
- Applies pack size filtering logic
- Handles embedding dimension mismatches

### 3. LLM Re-ranking
- Evaluates matches using business criteria:
  1. **Manufacturer/company match** (highest priority)
  2. **Brand alignment**
  3. **Pack specifications**
  4. **Price similarity**
  5. **Semantic similarity score**
- Can reject poor matches (returns rank "0")
- Provides confidence scoring

## 📈 Performance Metrics

The system tracks and reports:
- **Processing speed**: Transactions per minute
- **Match quality**: LLM selection vs rejection rates
- **Error handling**: Failed embeddings, LLM timeouts
- **Confidence distribution**: High/medium/low confidence matches

Example output:
```
==================================================
LLM RE-RANKING SUMMARY
==================================================
Total transactions processed: 1,250
Successfully re-ranked: 1,125 (90.0%)
Rejected (no good match): 85 (6.8%)
LLM query failures: 40 (3.2%)
==================================================
```

## 🛠️ Development

### Adding New Models

1. **Embedding Models**: Update `MODEL_NAME` in `config.py`
2. **LLM Models**: Use `--llm-model` parameter or modify default in `llm_reranker.py`

### Customizing Matching Logic

1. **Semantic Criteria**: Modify embedding generation in `embedding_pipeline.py`
2. **Filtering Rules**: Update `filter_final_results()` in `transaction_matcher.py`
3. **LLM Criteria**: Adjust prompt template in `llm_reranker.py`

### Extending Data Schema

1. Add columns to `m_columns` or `t_columns` in `config.py`
2. Update embedding text generation logic
3. Modify LLM prompt template to include new fields

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Connection Errors
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
```

#### 2. FAISS Index Issues
```bash
# Force rebuild index
python main.py --build --file ./dataset/transaction/sample.csv
```

#### 3. Memory Issues
- Use smaller LLM models (`phi3:medium`, `mistral:7b`)
- Process files individually instead of batches
- Reduce FAISS index size by filtering master data

#### 4. Low Match Quality
- Increase embedding model quality
- Adjust LLM matching criteria
- Fine-tune pack size filtering logic
- Add data preprocessing/cleaning

## 📝 Logging

The system provides comprehensive logging:
- **INFO**: Progress updates and statistics
- **WARNING**: Non-critical issues (dimension mismatches, parsing errors)
- **ERROR**: Critical failures requiring attention

Logs include:
- Processing timestamps
- Match quality metrics
- Error details and fallback actions
- Performance statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Open an issue with:
   - Error messages
   - Sample data (anonymized)
   - System specifications
   - Steps to reproduce

---

**Note**: This system is designed for product matching scenarios where semantic similarity combined with business rules and LLM reasoning provides optimal results. Performance may vary based on data quality, model selection, and hardware specifications.