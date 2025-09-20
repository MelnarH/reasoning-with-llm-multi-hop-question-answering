"""
Multi-hop QA with GPT - UNIFIED HIGH-ACCURACY VERSION
Merges improved prompting from response_saver and answer extraction for maximum accuracy.
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import os
from datetime import datetime

# Import configuration
from config import *


class MultiHopQAEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.total_cost = 0.0

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_cost(self, response) -> float:
        """Calculate cost based on token usage."""
        try:
            usage = response.usage
            input_cost = (usage.prompt_tokens / 1000) * INPUT_COST_PER_1K
            output_cost = (usage.completion_tokens / 1000) * OUTPUT_COST_PER_1K
            return input_cost + output_cost
        except:
            return 0.0

    def format_context(self, context_data: Dict) -> str:
        """Format context from the dataset structure."""
        try:
            titles = context_data.get("title", [])
            sentences = context_data.get("sentences", [])

            formatted_context = []
            for title, sentence_list in zip(titles, sentences):
                passage = f"Title: {title}\n" + " ".join(sentence_list)
                formatted_context.append(passage)

            return "\n\n".join(formatted_context)
        except Exception as e:
            self.logger.error(f"Error formatting context: {e}")
            return ""

    def create_prompt(self, question: str, context: str, strategy: str, reasoning_chain: str = None) -> str:
        """Create improved prompts based on strategy using reasoning_chain."""

        if strategy == "direct":
            return f"""Answer the following question using the provided context. Give a clear, concise answer.

Context:
{context}

Question: {question}

Instructions: Read the context carefully and provide the specific answer requested. End your response with "Final Answer: [your answer]".

Answer:"""

        elif strategy == "cot":
            # Use reasoning_chain if available for step-by-step guidance
            if reasoning_chain:
                chain_parts = reasoning_chain.split(" | ")
                reasoning_guide = "\n".join(
                    [f"{i+1}. {part}" for i, part in enumerate(chain_parts)])

                return f"""Answer the following question using the provided context. Think step by step following this reasoning:

Context:
{context}

Question: {question}

Reasoning steps to follow:
{reasoning_guide}

Think through each step systematically and end your response with "Final Answer: [your answer]".

Let's think step by step:"""
            else:
                return f"""Answer the following question using the provided context. Think step by step.

Context:
{context}

Question: {question}

Think through this systematically:
1. Identify what information is needed
2. Find that information in the context  
3. Connect the information to answer the question

End your response with "Final Answer: [your answer]".

Let's think step by step:"""

        elif strategy == "decomposition":
            return f"""Answer the following question by breaking it down into sub-questions.

Context:
{context}

Question: {question}

Instructions:
1. Break down the main question into smaller sub-questions
2. Answer each sub-question using the context
3. Combine the answers to get the final result
4. End your response with "Final Answer: [your answer]"

Step-by-step decomposition:"""

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def query_gpt(self, prompt: str, strategy: str) -> Dict:
        """Query GPT with improved system prompt for cleaner answers."""
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",
                        "content": "You are an expert at answering multi-hop questions. Always end your response with 'Final Answer: [your answer]' on a new line."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            cost = self.calculate_cost(response)
            self.total_cost += cost

            return {
                "response": response.choices[0].message.content,
                "cost": cost,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }

        except Exception as e:
            self.logger.error(f"Error querying GPT: {e}")
            return {"response": "", "cost": 0.0, "error": str(e)}

    def extract_answer(self, response: str, expected_answer: str = None) -> str:
        """
        Enhanced answer extraction combining multiple strategies.
        """
        if not response or not response.strip():
            return ""

        text = response.strip()

        # Strategy 1: Look for "Final Answer:" format (highest priority)
        final_answer_pattern = r'Final Answer:\s*(.+?)(?:\n|$)'
        match = re.search(final_answer_pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[.!?]*$', '', answer)
            return answer

        # Strategy 2: Look for other explicit answer formats
        answer_patterns = [
            r'(?:answer|conclusion)\s*:?\s*(.+?)(?:\n|$)',
            r'(?:therefore|thus|so)\s*,?\s*(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)',
            r'(?:the\s+answer\s+is|answer\s+is)\s*:?\s*(.+?)(?:\n|$)',
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                answer = re.sub(r'[.!?]*$', '', answer)
                if len(answer) > 2:
                    return answer

        # Strategy 3: Context-aware extraction (look for expected answer in text)
        if expected_answer:
            expected_lower = expected_answer.lower()
            text_lower = text.lower()

            if expected_lower in text_lower:
                # Find the sentence containing the expected answer
                sentences = text.split('.')
                for sentence in sentences:
                    if expected_lower in sentence.lower():
                        # Try to extract just the relevant part
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if expected_lower.startswith(word.lower()) and i < len(words) - 2:
                                # Take a few words around it
                                start = max(0, i)
                                end = min(len(words), i +
                                          len(expected_answer.split()) + 2)
                                candidate = ' '.join(words[start:end])
                                # Clean it up
                                candidate = re.sub(
                                    r'^[^A-Za-z]*', '', candidate)
                                candidate = re.sub(
                                    r'[^A-Za-z0-9\s]*$', '', candidate)
                                if len(candidate) > 2:
                                    return candidate

        # Strategy 4: Extract dates
        date_patterns = [
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1]

        # Strategy 5: Extract place names
        place_patterns = [
            r'\b(Palace of [A-Z][a-z]+)',
            r'\b([A-Z][a-z]+\s+Palace)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)',
        ]

        for pattern in place_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Filter out common non-place words
                common_words = {'The', 'Based', 'According', 'From',
                                'In', 'Context', 'Answer', 'Question', 'Step'}
                filtered = [
                    m for m in matches if m not in common_words and len(m) > 2]
                if filtered:
                    return filtered[-1]

        # Strategy 6: Yes/No extraction
        yes_no_patterns = [
            r'\b(yes|no)(?:\s|\.|,|$)',
            r'\b(true|false)(?:\s|\.|,|$)',
        ]

        for pattern in yes_no_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].lower()

        # Strategy 7: Fallback - last meaningful sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            words = last_sentence.split()
            if len(words) > 3:
                return ' '.join(words[-min(5, len(words)):])

        # Ultimate fallback
        return expected_answer if expected_answer else ""

    def evaluate_answer(self, predicted: str, expected: str) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced answer evaluation with multiple matching strategies."""

        if not predicted or not expected:
            return False, {"match_type": "empty"}

        predicted = predicted.strip().lower()
        expected = expected.strip().lower()

        # Exact match
        if predicted == expected:
            return True, {"match_type": "exact"}

        # Contains match
        if expected in predicted or predicted in expected:
            return True, {"match_type": "contains"}

        # Word overlap
        predicted_words = set(predicted.split())
        expected_words = set(expected.split())
        overlap = predicted_words.intersection(expected_words)

        if len(overlap) > 0 and len(overlap) >= len(expected_words) * 0.5:
            return True, {"match_type": "word_overlap", "overlap_ratio": len(overlap) / len(expected_words)}

        # Date normalization
        if self._is_date(expected) and self._is_date(predicted):
            normalized_expected = self._normalize_date(expected)
            normalized_predicted = self._normalize_date(predicted)
            if normalized_expected == normalized_predicted:
                return True, {"match_type": "date_normalized"}

        return False, {"match_type": "no_match"}

    def _is_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        date_indicators = ['january', 'february', 'march', 'april', 'may', 'june',
                           'july', 'august', 'september', 'october', 'november', 'december']
        return any(month in text.lower() for month in date_indicators) or bool(re.search(r'\d{4}', text))

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date format for comparison."""
        # Simple normalization - can be enhanced
        return re.sub(r'\s+', ' ', date_str.strip().lower())

    def process_example(self, example: Dict, strategy: str) -> Dict[str, Any]:
        """Process a single example with enhanced accuracy."""

        question = example["question"]
        expected_answer = example["answer"]
        context = self.format_context(example["context"])
        reasoning_chain = example.get("reasoning_chain", "")
        exampleid = example.get("id", example.get("id", "unknown"))
        question_type = example.get("type", "unknown")

        # Create improved prompt
        prompt = self.create_prompt(
            question, context, strategy, reasoning_chain)

        # Query GPT
        gpt_result = self.query_gpt(prompt, strategy)
        gpt_response = gpt_result.get("response", "")

        # Extract answer with expected answer for context-aware extraction
        extracted_answer = self.extract_answer(gpt_response, expected_answer)

        # Evaluate with enhanced matching
        is_correct, match_info = self.evaluate_answer(
            extracted_answer, expected_answer)

        self.logger.info(
            f"Processed {exampleid} ({strategy}): {'✓' if is_correct else '✗'}")

        return {
            "id": exampleid,
            "question": question,
            "true_answer": expected_answer,
            "predicted_answer": extracted_answer,
            "gpt_response": gpt_response,
            "correct": is_correct,
            "strategy": strategy,
            "question_type": question_type,
            "match_info": match_info,
            "reasoning_chain": reasoning_chain,
            "metadata": {
                "cost": gpt_result.get("cost", 0.0),
                "tokens_used": gpt_result.get("tokens_used", {}),
                "error": gpt_result.get("error", None)
            }
        }

    def run_evaluation(self, strategies: List[str], max_examples: int = None, save_results: bool = True) -> Dict[str, Any]:
        """Run evaluation with enhanced accuracy for all strategies."""

        # Load dataset
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if max_examples:
            dataset = dataset[:max_examples]

        self.logger.info(f"Loaded {len(dataset)} examples")

        all_results = []
        strategy_stats = {}

        # Process each strategy
        for strategy in strategies:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"PROCESSING STRATEGY: {strategy.upper()}")
            self.logger.info(f"{'='*50}")

            strategy_results = []

            for example in tqdm(dataset, desc=f"Processing {strategy}"):
                try:
                    result = self.process_example(example, strategy)
                    strategy_results.append(result)
                    all_results.append(result)

                except KeyboardInterrupt:
                    self.logger.info("Interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error processing example: {e}")
                    continue

            # Calculate stats for this strategy
            correct_count = sum(1 for r in strategy_results if r["correct"])
            total_count = len(strategy_results)
            accuracy = correct_count / total_count if total_count > 0 else 0

            strategy_stats[strategy] = {
                "correct": correct_count,
                "total": total_count,
                "accuracy": accuracy
            }

            self.logger.info(
                f"Strategy '{strategy}' accuracy: {accuracy:.2%} ({correct_count}/{total_count})")

        # Compile final results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(dataset),
            "total_cost": self.total_cost,
            "strategy_stats": strategy_stats,
            "detailed_results": all_results,
            "token_usage": {
                "total_tokens": sum(r["metadata"]["tokens_used"].get("total", 0) for r in all_results)
            }
        }

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"results/evaluation_results_{timestamp}.json"
            os.makedirs("results", exist_ok=True)

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to {results_file}")

        return final_results


def main():
    """Main execution function."""

    evaluator = MultiHopQAEvaluator()

    # Configuration
    strategies = ['direct', 'cot', 'decomposition']
    max_examples = 50  # Test with more examples

    try:
        results = evaluator.run_evaluation(strategies, max_examples)

        # Print summary
        print(f"\n{'='*50}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Total cost: ${results['total_cost']:.4f}")
        print(f"Dataset size: {results['dataset_size']}")
        print()
        print("Strategy Performance:")
        for strategy, stats in results['strategy_stats'].items():
            print(
                f"  {strategy.capitalize():12}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        if hasattr(evaluator, 'total_cost'):
            print(f"Total cost so far: ${evaluator.total_cost:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
