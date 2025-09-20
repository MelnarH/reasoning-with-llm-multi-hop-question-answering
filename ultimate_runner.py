"""
Ultimate Multi-hop QA Runner with Integrated Visualization - FIXED
Runs the unified multihop QA evaluation and automatically generates visualizations.
"""

import os
import re
import traceback
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from dataset_processor import download_and_process_dataset
from multihop_qa_gpt import MultiHopQAEvaluator

# Keep only the needed matplotlib configurations
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.color': 'silver',
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
})


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\|?*]', '_', filename)
    # Remove emojis and special characters that might cause issues
    sanitized = re.sub(r'[^\w\s\-_\.]', '', sanitized)
    return sanitized


class VisualizationGenerator:
    """Generate comprehensive visualizations for multihop QA results."""

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.output_dir = os.path.abspath("results/visualizations/")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Visualization output directory: {self.output_dir}")

    def print_detailed_summary(self):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*70)
        print("MULTI-HOP QA EVALUATION RESULTS")
        print("="*70)

        print(f"Dataset Size: {self.results['dataset_size']}")
        print(f"Total Cost: ${self.results['total_cost']:.4f}")
        print(f"Total Tokens: {self.results['token_usage']['total_tokens']:,}")
        print(f"Timestamp: {self.results['timestamp']}")

        print("\nSTRATEGY PERFORMANCE:")
        print("-"*50)

        best_strategy = ""
        best_accuracy = 0

        for strategy, stats in self.results['strategy_stats'].items():
            accuracy = stats['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_strategy = strategy

            indicator = "[BEST]" if strategy == best_strategy else "      "
            print(
                f"{indicator} {strategy.capitalize():15}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")

        print(
            f"\nBest Strategy: {best_strategy.upper()} with {best_accuracy:.1%} accuracy!")

    def analyze_by_question_type(self) -> Dict[str, Dict]:
        """Analyze performance by question type."""
        type_stats = {}

        for result in self.results['detailed_results']:
            qtype = result.get('question_type', 'unknown')
            strategy = result['strategy']

            if qtype not in type_stats:
                type_stats[qtype] = {}
            if strategy not in type_stats[qtype]:
                type_stats[qtype][strategy] = {"correct": 0, "total": 0}

            type_stats[qtype][strategy]["total"] += 1
            if result['correct']:
                type_stats[qtype][strategy]["correct"] += 1

        # Calculate accuracies
        for qtype in type_stats:
            for strategy in type_stats[qtype]:
                stats = type_stats[qtype][strategy]
                stats['accuracy'] = stats['correct'] / \
                    stats['total'] if stats['total'] > 0 else 0

        return type_stats

    def create_strategy_comparison_chart(self):
        """Create strategy performance comparison chart."""
        strategies = list(self.results['strategy_stats'].keys())
        accuracies = [self.results['strategy_stats'][s]['accuracy']
                      for s in strategies]

        plt.figure(figsize=(12, 8))

        # Create bar chart with custom colors
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
        bars = plt.bar(strategies, accuracies, color=colors[:len(strategies)])

        plt.title('Strategy Performance Comparison',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                     f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()

        # Save and show
        filepath = os.path.join(self.output_dir, "strategy_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # Verify file was created
        if os.path.exists(filepath):
            print(f"   ✅ Strategy comparison saved to: {filepath}")
        else:
            print(f"   ❌ Failed to save strategy comparison to: {filepath}")
        plt.show()

    def create_question_type_analysis(self):
        """Create performance by question type analysis."""
        type_stats = self.analyze_by_question_type()

        if not type_stats:
            print("No question type data available for visualization")
            return

        # Prepare data for visualization
        df_data = []
        for qtype, strategies in type_stats.items():
            for strategy, stats in strategies.items():
                df_data.append({
                    'Question Type': qtype,
                    'Strategy': strategy.capitalize(),
                    'Accuracy': stats['accuracy'],
                    'Correct': stats['correct'],
                    'Total': stats['total']
                })

        df = pd.DataFrame(df_data)

        plt.figure(figsize=(14, 8))

        # Create grouped bar chart
        sns.barplot(data=df, x='Question Type', y='Accuracy', hue='Strategy',
                    palette=['#2E86AB', '#A23B72', '#F18F01'])

        plt.title('Performance by Question Type',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Question Type', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.legend(title='Strategy', title_fontsize=12, fontsize=11)
        plt.tight_layout()

        # Save and show
        filepath = os.path.join(self.output_dir, "performance_by_type.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # Verify file was created
        if os.path.exists(filepath):
            print(f"   ✅ Performance by type saved to: {filepath}")
        else:
            print(f"   ❌ Failed to save performance by type to: {filepath}")
        plt.show()

    def create_cost_analysis(self):
        """Create cost analysis visualization."""
        strategies = list(self.results['strategy_stats'].keys())

        # Calculate cost per strategy (estimate based on proportion)
        total_cost = self.results['total_cost']
        total_examples = sum(
            self.results['strategy_stats'][s]['total'] for s in strategies)

        strategy_costs = []
        for strategy in strategies:
            strategy_total = self.results['strategy_stats'][strategy]['total']
            strategy_cost = (strategy_total / total_examples) * \
                total_cost if total_examples > 0 else 0
            strategy_costs.append(strategy_cost)

        plt.figure(figsize=(10, 6))

        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = plt.bar(strategies, strategy_costs,
                       color=colors[:len(strategies)])

        plt.title('Cost Analysis by Strategy',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Cost ($)', fontsize=14)

        # Add value labels on bars
        for bar, cost in zip(bars, strategy_costs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + height*0.02,
                     f'${cost:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save and show
        filepath = os.path.join(self.output_dir, "cost_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # Verify file was created
        if os.path.exists(filepath):
            print(f"   ✅ Cost analysis saved to: {filepath}")
        else:
            print(f"   ❌ Failed to save cost analysis to: {filepath}")
        plt.show()

    def create_accuracy_vs_cost_scatter(self):
        """Create accuracy vs cost efficiency scatter plot."""
        strategies = list(self.results['strategy_stats'].keys())
        accuracies = [self.results['strategy_stats'][s]['accuracy']
                      for s in strategies]

        # Calculate cost per strategy
        total_cost = self.results['total_cost']
        total_examples = sum(
            self.results['strategy_stats'][s]['total'] for s in strategies)

        costs = []
        for strategy in strategies:
            strategy_total = self.results['strategy_stats'][strategy]['total']
            strategy_cost = (strategy_total / total_examples) * \
                total_cost if total_examples > 0 else 0
            costs.append(strategy_cost)

        plt.figure(figsize=(10, 8))

        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for i, (strategy, acc, cost) in enumerate(zip(strategies, accuracies, costs)):
            plt.scatter(cost, acc, s=200, color=colors[i % len(colors)],
                        alpha=0.7, edgecolors='black', linewidth=2, label=strategy.capitalize())

            # Add strategy labels
            plt.annotate(strategy.capitalize(), (cost, acc),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=11, fontweight='bold')

        plt.title('Accuracy vs Cost Efficiency',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Cost ($)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # Save and show
        filepath = os.path.join(self.output_dir, "accuracy_vs_cost.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # Verify file was created
        if os.path.exists(filepath):
            print(f"   ✅ Accuracy vs cost saved to: {filepath}")
        else:
            print(f"   ❌ Failed to save accuracy vs cost to: {filepath}")
        plt.show()

    def export_detailed_analysis(self):
        """Export detailed analysis to CSV."""
        detailed_data = []

        for result in self.results['detailed_results']:
            detailed_data.append({
                'ID': result['id'],
                'Question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                'True Answer': result['true_answer'],
                'Predicted Answer': result['predicted_answer'],
                'Correct': result['correct'],
                'Strategy': result['strategy'],
                'Question Type': result.get('question_type', 'unknown'),
                'Match Type': result.get('match_info', {}).get('match_type', 'unknown'),
                'Cost': result['metadata'].get('cost', 0),
                'Tokens Used': result['metadata'].get('tokens_used', {}).get('total', 0)
            })

        df = pd.DataFrame(detailed_data)

        # Save to results folder
        csv_filename = "detailed_analysis.csv"
        csv_filepath = os.path.join("results/", csv_filename)
        df.to_csv(csv_filepath, index=False)

        if os.path.exists(csv_filepath):
            print(f"   ✅ Detailed analysis exported to: {csv_filepath}")
        else:
            print(
                f"   ❌ Failed to export detailed analysis to: {csv_filepath}")

        return df

    def generate_all_visualizations(self):
        """Generate all visualizations and analyses."""
        print("\n🎨 GENERATING VISUALIZATIONS...")
        print("-" * 50)

        # Verify output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"📁 Created output directory: {self.output_dir}")

        try:
            # 1. Print summary
            self.print_detailed_summary()

            # 2. Generate visualizations
            print("\nCreating strategy comparison chart...")
            self.create_strategy_comparison_chart()

            print("\nCreating question type analysis...")
            self.create_question_type_analysis()

            print("\nCreating cost analysis...")
            self.create_cost_analysis()

            print("\nCreating accuracy vs cost analysis...")
            self.create_accuracy_vs_cost_scatter()

            # 3. Export detailed data
            print("\nExporting detailed analysis...")
            df = self.export_detailed_analysis()

            # List all created files
            created_files = [f for f in os.listdir(
                self.output_dir) if f.endswith('.png')]
            print(f"\n✅ Generated {len(created_files)} visualization files:")
            for file in created_files:
                full_path = os.path.join(self.output_dir, file)
                print(f"   📈 {full_path}")

            print(f"\n📁 All visualizations saved to: {self.output_dir}")

            return df

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            traceback.print_exc()


def ensure_dataset_exists():
    """
    Ensure the dataset file exists. If not, create it.
    """
    if os.path.exists(DATASET_PATH):
        print(f"✅ Dataset found at {DATASET_PATH}")
        return True

    print(f"⚠️  Dataset not found at {DATASET_PATH}")

    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(DATASET_PATH)
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {data_dir}")

    try:

        print("🔄 Creating dataset automatically...")

        result = download_and_process_dataset(
            sample_size=80,
            output_dir=data_dir or ".",
            seed=42
        )

        if os.path.exists(DATASET_PATH):
            print(f"✅ Dataset created successfully at {DATASET_PATH}")
            return True
        else:
            print(
                f"❌ Dataset creation failed - file not found at {DATASET_PATH}")
            return False

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        print("   Please run dataset_processor.py manually first")
        return False


def main():
    """Main execution function with integrated visualization."""

    print("STARTING ULTIMATE MULTI-HOP QA EVALUATION")
    print("="*60)

    # First, ensure dataset exists
    if not ensure_dataset_exists():
        print("❌ Cannot proceed without dataset. Please create the dataset manually.")
        return

    # Initialize evaluator
    evaluator = MultiHopQAEvaluator()

    # Configuration
    strategies = ['direct', 'cot', 'decomposition']
    max_examples = 80

    print(f"Configuration:")
    print(f"   Strategies: {strategies}")
    print(f"   Max examples: {max_examples}")
    print(f"   Model: {OPENAI_MODEL}")

    try:
        # Run evaluation
        print("\nRunning evaluation...")
        results = evaluator.run_evaluation(
            strategies, max_examples, save_results=True)

        # Generate visualizations
        print("\nGenerating visualizations...")
        viz_generator = VisualizationGenerator(results)
        detailed_df = viz_generator.generate_all_visualizations()

        # Final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"Total Cost: ${results['total_cost']:.4f}")
        print(f"Results saved and visualized")
        print(f"Check the results/ directory for detailed outputs")

        return results, detailed_df

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        if hasattr(evaluator, 'total_cost'):
            print(f"Total cost so far: ${evaluator.total_cost:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
