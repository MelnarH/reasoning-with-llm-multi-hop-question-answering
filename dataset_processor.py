"""
Complete Dataset Processor for 2WikiMultiHopQA
"""

import json
import random
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset


def download_and_process_dataset(
    sample_size: int = 80,
    output_dir: str = "data",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Download 2WikiMultiHopQA dataset and process it completely
    This replaces the notebook processing logic
    """

    print("🔄 Starting dataset download and processing...")
    random.seed(seed)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    try:
        # Download dataset from HuggingFace
        print("📥 Downloading 2WikiMultiHopQA from HuggingFace...")
        ds = load_dataset("framolfese/2WikiMultihopQA")
        print(f"✅ Dataset loaded: {len(ds['train'])} training examples")

        # Sample the dataset (replicates notebook logic)
        print(f"🎲 Sampling {sample_size} examples...")
        train_data = list(ds["train"])
        sampled_data = random.sample(
            train_data, min(sample_size, len(train_data)))

        # Save candidate_2wiki.json (original format)
        candidate_path = "candidate_2wiki.json"
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {candidate_path} ({len(sampled_data)} esxamples)")

        # Process and enhance the dataset
        print("🔧 Enhancing dataset with reasoning chains...")
        enhanced_data = []
        reasoning_stats = {"with_reasoning": 0, "empty_reasoning": 0}

        for example in sampled_data:
            # Create reasoning chain from evidences
            reasoning_steps = []
            evidences = example.get("evidences", [])

            for evidence in evidences:
                if len(evidence) >= 3:
                    entity, relation, value = evidence[0], evidence[1], evidence[2]
                    reasoning_steps.append(f"{entity} → {relation} → {value}")
                elif len(evidence) == 2:
                    entity, value = evidence[0], evidence[1]
                    reasoning_steps.append(f"{entity} → {value}")

            # Add reasoning_chain field
            enhanced_example = example.copy()
            reasoning_chain = " | ".join(reasoning_steps)
            enhanced_example["reasoning_chain"] = reasoning_chain
            enhanced_data.append(enhanced_example)

            # Update stats
            if reasoning_chain:
                reasoning_stats["with_reasoning"] += 1
            else:
                reasoning_stats["empty_reasoning"] += 1

        # Save enhanced dataset
        enhanced_path = Path(output_dir) / "enhanced_2wiki_dataset.json"
        with open(enhanced_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {enhanced_path} ({len(enhanced_data)} examples)")

        # Analysis summary
        print(f"📊 Reasoning chain analysis:")
        print(f"   • With reasoning: {reasoning_stats['with_reasoning']}")
        print(f"   • Empty reasoning: {reasoning_stats['empty_reasoning']}")

        # Show sample
        if enhanced_data and enhanced_data[0]["reasoning_chain"]:
            sample_chain = enhanced_data[0]["reasoning_chain"]
            if len(sample_chain) > 100:
                sample_chain = sample_chain[:100] + "..."
            print(f"🔍 Sample reasoning: {sample_chain}")

        return {
            "success": True,
            "candidate_path": candidate_path,
            "enhanced_path": str(enhanced_path),
            "sample_size": len(enhanced_data),
            "reasoning_stats": reasoning_stats
        }

    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def analyze_existing_dataset(dataset_path: str) -> Dict[str, Any]:
    """Analyze an existing dataset file"""

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic statistics
        total_examples = len(data)
        question_types = {}
        reasoning_lengths = []

        for example in data:
            # Question type distribution
            qtype = example.get("type", "unknown")
            question_types[qtype] = question_types.get(qtype, 0) + 1

            # Reasoning chain analysis
            reasoning_chain = example.get("reasoning_chain", "")
            reasoning_steps = len(reasoning_chain.split(
                " | ")) if reasoning_chain else 0
            reasoning_lengths.append(reasoning_steps)

        analysis = {
            "total_examples": total_examples,
            "question_types": question_types,
            "avg_reasoning_steps": sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0,
            "reasoning_length_distribution": {
                "min": min(reasoning_lengths) if reasoning_lengths else 0,
                "max": max(reasoning_lengths) if reasoning_lengths else 0,
                "avg": sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
            }
        }

        return analysis

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    result = download_and_process_dataset(sample_size=80)
    if result["success"]:
        print("✅ Dataset processing completed successfully!")
        print(f"📁 Files created:")
        print(f"   • {result['candidate_path']}")
        print(f"   • {result['enhanced_path']}")
    else:
        print(f"❌ Dataset processing failed: {result['error']}")
