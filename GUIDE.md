# ðŸš€ Multi-Hop QA with GPT - Pure Python Setup

## ðŸ“‹ What You Get

- **100% Pure Python** - No bash scripts, no notebooks
- **One Command** - Downloads data, processes it, runs evaluation
- **Complete Pipeline** - From raw data to results

## Required Files

1. `ultimate_runner.py` - Main script (does everything)
2. `dataset_processor.py` - Dataset handling
3. `config.py` - Configuration
4. `multihop_qa_gpt.py` - Evaluation logic
5. `requirements.txt` - Dependencies

## ðŸ”§ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Windows
set OPENAI_API_KEY=your-key-here

# Linux/Mac
export OPENAI_API_KEY=your-key-here

# Or create .env file
echo 'OPENAI_API_KEY=your-key-here' > .env
```

### 3. Run Everything

```bash
python ultimate_runner.py
```

## What Happens

1. **Environment Check** - Verifies setup
2. **Dataset Download** - Gets 2WikiMultiHopQA from HuggingFace
3. **Data Processing** - Creates both JSON files:
   - `candidate_2wiki.json` (original format)
   - `data/enhanced_2wiki_dataset.json` (with reasoning chains)
4. **Cost Estimation** - Shows expected cost
5. **Evaluation** - Runs all three strategies:
   - Direct prompting
   - Chain-of-Thought
   - Question decomposition
6. **Results** - Shows performance and saves detailed results

## Output Files

- `candidate_2wiki.json` - Original sampled dataset
- `data/enhanced_2wiki_dataset.json` - Enhanced with reasoning chains
- `results/evaluation_results_*.json` - Detailed evaluation results
- `logs/` - Execution logs
