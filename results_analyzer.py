"""
Analysis and visualization of evaluation results
"""

import seaborn as sns
from typing import Dict, Any, List
import pandas as pd
import json
import matplotlib.pyplot as plt
import cycler

# Enable grid and update its appearance
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'grid.color': 'silver'})
plt.rcParams.update({'grid.linestyle': '--'})

# Set figure resolution
plt.rcParams.update({'figure.dpi': 150})

# Hide the top and right spines
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

# Increase font sizes
plt.rcParams.update({'font.size': 12})  # General font size
plt.rcParams.update({'axes.titlesize': 14})  # Title font size
plt.rcParams.update({'axes.labelsize': 12})  # Axis label font size


class ResultsAnalyzer:
    """Analyze and visualize evaluation results"""

    def __init__(self, results_path: str):
        with open(results_path, 'r') as f:
            self.results = json.load(f)

    def print_summary(self):
        """Print evaluation summary"""
        print("=" * 50)
        print("MULTI-HOP QA EVALUATION RESULTS")
        print("=" * 50)

        print(f"Dataset Size: {self.results['dataset_size']}")
        print(f"Total Cost: ${self.results['total_cost']:.2f}")
        print(f"Total Tokens: {sum(self.results['token_usage'].values()):,}")

        print("\nStrategy Performance:")
        print("-" * 30)

        for strategy, stats in self.results['strategy_stats'].items():
            print(
                f"{strategy.capitalize():15}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    def analyze_by_question_type(self) -> Dict[str, Dict]:
        """Analyze performance by question type"""

        type_stats = {}

        for result in self.results['detailed_results']:
            qtype = result['question_type']
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

    def create_visualizations(self, output_dir: str = "results/"):
        """Create performance visualizations"""

        # Overall performance comparison
        strategies = list(self.results['strategy_stats'].keys())
        accuracies = [self.results['strategy_stats'][s]['accuracy']
                      for s in strategies]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, accuracies, color=[
                       '#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Strategy Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{acc:.2%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{output_dir}strategy_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Performance by question type
        type_stats = self.analyze_by_question_type()

        if type_stats:
            df_data = []
            for qtype, strategies in type_stats.items():
                for strategy, stats in strategies.items():
                    df_data.append({
                        'Question Type': qtype,
                        'Strategy': strategy,
                        'Accuracy': stats['accuracy']
                    })

            df = pd.DataFrame(df_data)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Question Type',
                        y='Accuracy', hue='Strategy')
            plt.title('Performance by Question Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}performance_by_type.png',
                        dpi=300, bbox_inches='tight')
            plt.show()

    def export_detailed_analysis(self, output_path: str):
        """Export detailed analysis to CSV"""

        detailed_data = []
        for result in self.results['detailed_results']:
            detailed_data.append({
                'ID': result['id'],
                'Question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                'True Answer': result['true_answer'],
                'Predicted Answer': result['predicted_answer'],
                'Correct': result['correct'],
                'Strategy': result['strategy'],
                'Question Type': result['question_type'],
                'Processing Time': result['metadata'].get('processing_time', 0),
                'Tokens Used': result['metadata'].get('tokens_used', 0)
            })

        df = pd.DataFrame(detailed_data)
        df.to_csv(output_path, index=False)
        print(f"✓ Detailed analysis exported to {output_path}")


def compare_multiple_runs(results_paths: List[str]):
    """Compare results from multiple evaluation runs"""

    all_results = []
    for path in results_paths:
        with open(path, 'r') as f:
            results = json.load(f)
            all_results.append(results)

    # Create comparison DataFrame
    comparison_data = []
    for i, results in enumerate(all_results):
        for strategy, stats in results['strategy_stats'].items():
            comparison_data.append({
                'Run': f'Run {i+1}',
                'Strategy': strategy,
                'Accuracy': stats['accuracy'],
                'Cost': results['total_cost']
            })

    df = pd.DataFrame(comparison_data)

    # Visualize comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    sns.barplot(data=df, x='Strategy', y='Accuracy', hue='Run')
    plt.title('Accuracy Comparison Across Runs')

    plt.subplot(2, 1, 2)
    run_costs = df.groupby('Run')['Cost'].first()
    plt.bar(run_costs.index, run_costs.values)
    plt.title('Cost per Run')
    plt.ylabel('Cost ($)')

    plt.tight_layout()
    plt.show()

    return df
