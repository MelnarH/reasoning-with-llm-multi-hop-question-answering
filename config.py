"""
Configuration for Multi-Hop QA with GPT
FIXED: All paths and variables that the system expects
"""

import os
from pathlib import Path

# Model Configuration
# Changed from MODEL_NAME to match multihop_qa_gpt.py
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 150
TEMPERATURE = 0.0
MAX_BUDGET_USD = 30.0  # Added to match multihop_qa_gpt.py's expected variable

# Cost tracking configuration
COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {
        "input": 0.0015,
        "output": 0.002
    }
}

# File Paths
# Added to match expected variable
DATASET_PATH = "data/enhanced_2wiki_dataset.json"
ENHANCED_DATASET_PATH = DATASET_PATH  # Keep for backwards compatibility
CANDIDATE_DATASET_PATH = "candidate_2wiki.json"

# Batch processing
BATCH_SIZE = 10  # Added for batch processing in multihop_qa_gpt.py

# Results Configuration
RESULTS_PATH = "results/"  # Added trailing slash to match expected format
LOGS_PATH = "logs/"  # Already correct

# Create directories if they don't exist
Path(RESULTS_PATH).mkdir(exist_ok=True)
Path(LOGS_PATH).mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

print("✅ Configuration loaded successfully")
print(f"📁 Enhanced dataset path: {ENHANCED_DATASET_PATH}")
print(f"📁 Logs path: {LOGS_PATH}")
print(f"💰 Budget limit: ${MAX_BUDGET_USD}")

INPUT_COST_PER_1K = COST_PER_1K_TOKENS[OPENAI_MODEL]["input"]
OUTPUT_COST_PER_1K = COST_PER_1K_TOKENS[OPENAI_MODEL]["output"]
