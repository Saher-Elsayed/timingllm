#!/usr/bin/env python3
"""
TimingLLM Utility Functions
===========================

Helper functions for data processing, visualization, and analysis.

Author: Saher Elsayed
Institution: University of Pennsylvania
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


def load_dataset(filepath: str = "data/timing_dataset.json") -> List[Dict]:
    """
    Load timing dataset from JSON file.
    
    Args:
        filepath: Path to the dataset JSON file
        
    Returns:
        List of design dictionaries
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: Dict, filepath: str = "results/experiment_results.json") -> None:
    """
    Save experimental results to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def extract_all_paths(dataset: List[Dict]) -> List[Dict]:
    """
    Extract all timing paths from dataset.
    
    Args:
        dataset: List of design dictionaries
        
    Returns:
        Flat list of all timing paths
    """
    all_paths = []
    for design in dataset:
        for path in design['report']['paths']:
            path['design_category'] = design['category']
            path['design_name'] = design['report']['design_name']
        all_paths.extend(design['report']['paths'])
    return all_paths


def split_dataset(paths: List[Dict], 
                  train_ratio: float = 0.8, 
                  seed: int = 42) -> tuple:
    """
    Split paths into train and test sets.
    
    Args:
        paths: List of timing paths
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, test_paths)
    """
    np.random.seed(seed)
    paths_copy = paths.copy()
    np.random.shuffle(paths_copy)
    
    split_idx = int(train_ratio * len(paths_copy))
    return paths_copy[:split_idx], paths_copy[split_idx:]


def compute_slack_improvement(before: float, after: float) -> float:
    """
    Compute percentage improvement in slack.
    
    Args:
        before: Slack before fix (typically negative)
        after: Slack after fix
        
    Returns:
        Percentage improvement
    """
    if before >= 0:
        return 0.0
    return (after - before) / abs(before) * 100


def generate_latex_table(metrics: Dict, caption: str = "Results") -> str:
    """
    Generate LaTeX table from metrics dictionary.
    
    Args:
        metrics: Dictionary with metrics
        caption: Table caption
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Violation Type} & \\textbf{Prec.} & \\textbf{Recall} & \\textbf{F1} \\\\",
        "\\midrule",
    ]
    
    for label, m in metrics.items():
        if isinstance(m, dict) and 'precision' in m:
            name = label.replace('_', ' ').title()
            lines.append(f"{name} & {m['precision']:.2f} & {m['recall']:.2f} & {m['f1']:.2f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def plot_classification_results(per_class_metrics: Dict, 
                                output_path: str = "results/classification_plot.png") -> None:
    """
    Create bar chart of classification results.
    
    Args:
        per_class_metrics: Dictionary with per-class metrics
        output_path: Path to save the figure
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    labels = list(per_class_metrics.keys())
    short_labels = [l.replace('_', '\n') for l in labels]
    precision = [per_class_metrics[l]['precision'] for l in labels]
    recall = [per_class_metrics[l]['recall'] for l in labels]
    f1 = [per_class_metrics[l]['f1'] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(x - width, precision, width, label='Precision', color='#4285F4')
    ax.bar(x, recall, width, label='Recall', color='#34A853')
    ax.bar(x + width, f1, width, label='F1 Score', color='#FBBC04')
    
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance by Violation Type')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_slack_improvement(slack_results: Dict,
                           output_path: str = "results/slack_improvement.png") -> None:
    """
    Create bar chart of slack improvement by category.
    
    Args:
        slack_results: Dictionary with slack improvement by category
        output_path: Path to save the figure
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    categories = [k for k in slack_results.keys() if k != 'overall']
    cat_labels = [c.replace('_', '\n').title() for c in categories]
    wns_red = [slack_results[c]['wns_reduction_pct'] for c in categories]
    tns_red = [slack_results[c]['tns_reduction_pct'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(x - width/2, wns_red, width, label='WNS Reduction', color='#4285F4')
    ax.bar(x + width/2, tns_red, width, label='TNS Reduction', color='#EA4335')
    
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Timing Slack Improvement by Design Category')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend()
    ax.set_ylim(0, 80)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_summary(results: Dict) -> None:
    """
    Print formatted summary of experimental results.
    
    Args:
        results: Dictionary containing all experimental results
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 60)
    
    if 'classification' in results:
        cls = results['classification']
        print("\nClassification Performance:")
        print(f"  Overall F1:    {cls.get('overall', {}).get('f1', 0):.3f}")
        print(f"  Overall Prec:  {cls.get('overall', {}).get('precision', 0):.3f}")
        print(f"  Overall Rec:   {cls.get('overall', {}).get('recall', 0):.3f}")
    
    if 'slack_improvement' in results:
        slack = results['slack_improvement']
        print("\nSlack Improvement:")
        for cat, vals in slack.items():
            if isinstance(vals, dict) and 'wns_reduction_pct' in vals:
                print(f"  {cat:20s}: WNS {vals['wns_reduction_pct']:5.1f}%, TNS {vals['tns_reduction_pct']:5.1f}%")
    
    if 'ablation' in results:
        abl = results['ablation']
        print("\nAblation Study:")
        print(f"  Full Model F1:     {abl.get('full_model', {}).get('f1', 0):.3f}")
        print(f"  Without RAG F1:    {abl.get('ablated_model', {}).get('f1', 0):.3f}")
        print(f"  Degradation:       {abl.get('degradation_pct', 0):.1f}%")
    
    print("\n" + "=" * 60)
