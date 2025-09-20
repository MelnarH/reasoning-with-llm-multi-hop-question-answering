# Multi-Hop Question Answering with GPT

A comprehensive implementation comparing different prompting strategies for multi-hop question answering using OpenAI's GPT models on the 2WikiMultiHopQA dataset.

## Project Overview

This project implements and compares three different prompting strategies for multi-hop question answering:

1. **Direct Prompting**: Straightforward question-answer format
2. **Chain-of-Thought (CoT)**: Step-by-step reasoning approach
3. **Question Decomposition**: Breaking complex questions into simpler sub-questions

## Dataset

- **Source**: [2WikiMultiHopQA](https://huggingface.co/datasets/framolfese/2WikiMultiHopQA)
- **Size**: 80 selected examples
- **Enhancement**: Added `reasoning_chain` field to track expected reasoning paths

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd multihop-qa-gpt

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup OpenAI API

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Project Structure

```
project/
├── candidate_2wiki.json      # Candidate dataset file
├── config.py                 # Configuration settings
├── dataset_processor.py      # Dataset processing utilities
├── GUIDE.md                 # Project guide and documentation
├── multihop_qa_gpt.py       # Main implementation
├── results_analyzer.py       # Results analysis and visualization
├── ultimate_runner.py        # Main execution script
├── requirements.txt          # Python dependencies
├── data/
│   └── enhanced_2wiki_dataset.json  # Enhanced dataset with reasoning chains
├── logs/
│   └── multihop_qa.log      # Execution logs
└── results/
    ├── detailed_analysis.csv    # Detailed analysis results
    ├── evaluation_results_*.json # Evaluation outputs
    └── visualizations/         # Performance visualization charts
        ├── accuracy_vs_cost.png
        ├── cost_analysis.png
        ├── performance_by_type.png
        └── strategy_comparison.png
```

## Configuration

Key settings in `config.py`:

- **Model**: `gpt-3.5-turbo` (changeable to `gpt-4`)
- **Budget**: $30 USD limit with cost tracking
- **Sample Size**: 80 examples

## Prompting Strategies

### 1. Direct Prompting

```
Answer the following question using the provided context. Give a clear, concise answer.

Context:
[context]

Question: [question]

Instructions: Read the context carefully and provide the specific answer requested. End your response with "Final Answer: [your answer]".

Answer:
```

### 2. Chain-of-Thought

With reasoning chain:

```
Answer the following question using the provided context. Think step by step following this reasoning:

Context:
[context]

Question: [question]

Reasoning steps to follow:
[reasoning_chain split into numbered steps]

Think through each step systematically and end your response with "Final Answer: [your answer]".

Let's think step by step:
```

Without reasoning chain:

```
Answer the following question using the provided context. Think step by step.

Context:
[context]

Question: [question]

Think through this systematically:
1. Identify what information is needed
2. Find that information in the context
3. Connect the information to answer the question

End your response with "Final Answer: [your answer]".

Let's think step by step:
```

### 3. Question Decomposition

```
Answer the following question by breaking it down into sub-questions.

Context:
[context]

Question: [question]

Instructions:
1. Break down the main question into smaller sub-questions
2. Answer each sub-question using the context
3. Combine the answers to get the final result
4. End your response with "Final Answer: [your answer]"

Step-by-step decomposition:
```

## Cost Management

- **Budget Tracking**: Real-time cost monitoring
- **Usage Limits**: Automatic stopping at budget threshold
- **Token Estimation**: Pre-request cost estimation

## Expected Outputs

1. **Performance Metrics**: Accuracy for each strategy
2. **Cost Analysis**: Token usage and API costs
3. **Visualizations**: Strategy comparison charts
4. **Detailed Results**: Per-example analysis with reasoning chains

## Key Features

- **Cost-Aware**: Built-in budget tracking and limits
- **Reasoning Chains**: Enhanced dataset with expected reasoning paths
- **Multiple Strategies**: Direct, CoT, and Decomposition approaches
- **Comprehensive Analysis**: Detailed performance metrics and visualizations
- **Error Handling**: Robust error handling and logging

## Dataset Format

Each example includes:

```json
{
  "id": "unique_identifier",
  "question": "Multi-hop question",
  "answer": "Expected answer",
  "type": "inference|comparison|bridge|intersection",
  "context": {
    "title": ["Title1", "Title2"],
    "sentences": [["Sentence 1", "Sentence 2"], ["Sentence 3"]]
  },
  "reasoning_chain": "Entity -> relation -> Value | Entity2 -> relation2 -> Value2"
}
```

## 9 Model Compatibility

- **Primary**: GPT-3.5-turbo (cost-effective)
- **Optional**: GPT-4 (higher accuracy, higher cost)
- **Budget**: $30 limit for complete evaluation

## Expected Results

Based on multi-hop QA literature, expected performance hierarchy:

1. **Chain-of-Thought**: Highest accuracy (explicit reasoning)
2. **Question Decomposition**: Moderate accuracy (structured approach)
3. **Direct Prompting**: Lower accuracy (no reasoning guidance)

## Troubleshooting

- **API Errors**: Check OpenAI API key and rate limits
- **Budget Issues**: Monitor cost tracking logs

## Resources

- [2WikiMultiHopQA Dataset](https://huggingface.co/datasets/framolfese/2WikiMultiHopQA)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## Evaluation Metrics

- **Accuracy**: Exact match between predicted and true answers
- **Cost Efficiency**: Accuracy per dollar spent
- **Reasoning Quality**: Analysis of generated reasoning chains
- **Performance by Type**: Breakdown by question type (inference, comparison, etc.)
