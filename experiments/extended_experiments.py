#!/usr/bin/env python3
"""
TimingLLM Extended Experiments
==============================

Additional experiments for comprehensive evaluation:
1. Scalability Analysis - Performance vs design complexity
2. Cross-Domain Transfer - Train on one domain, test on another

Author: Saher Elsayed
Institution: University of Pennsylvania
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time

# Add source directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_generator import generate_dataset, generate_design_timing_report, ViolationType
from timingllm_agent import TimingLLMAgent, evaluate_agent

plt.style.use('seaborn-v0_8-whitegrid')


class ExtendedExperiments:
    """Extended experimental evaluation for TimingLLM."""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_all(self):
        """Run all extended experiments."""
        print("=" * 60)
        print("TimingLLM Extended Experiments")
        print("=" * 60)
        
        # Experiment 1: Scalability Analysis
        print("\n[1/2] Running Scalability Analysis...")
        self.scalability_results = self._run_scalability_experiment()
        
        # Experiment 2: Cross-Domain Transfer
        print("\n[2/2] Running Cross-Domain Transfer Learning...")
        self.transfer_results = self._run_cross_domain_transfer()
        
        # Generate figures
        print("\n[*] Generating figures...")
        self._generate_extended_figures()
        
        # Save results
        self._save_results()
        
        print("\n" + "=" * 60)
        print("Extended experiments completed!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return self.results
    
    def _run_scalability_experiment(self):
        """
        Experiment 1: Scalability Analysis
        
        Evaluate how TimingLLM performs as design complexity increases.
        Metrics: F1 score, diagnosis time, memory usage proxy
        """
        print("\n  Evaluating performance across design sizes...")
        
        # Define design complexity levels
        complexity_levels = [
            {"name": "Small", "violations": 25, "logic_levels_max": 8},
            {"name": "Medium", "violations": 50, "logic_levels_max": 12},
            {"name": "Large", "violations": 100, "logic_levels_max": 16},
            {"name": "Very Large", "violations": 200, "logic_levels_max": 20},
            {"name": "Ultra Large", "violations": 400, "logic_levels_max": 25},
        ]
        
        results = []
        
        for level in complexity_levels:
            print(f"    Testing {level['name']} designs ({level['violations']} violations)...")
            
            # Generate dataset for this complexity level
            designs = [
                {"name": f"design_{level['name'].lower()}_net", "category": "networking", 
                 "freq": 350.0, "violations": level['violations']},
                {"name": f"design_{level['name'].lower()}_dsp", "category": "signal_processing", 
                 "freq": 400.0, "violations": level['violations']},
                {"name": f"design_{level['name'].lower()}_ai", "category": "ai_accelerator", 
                 "freq": 375.0, "violations": level['violations']},
            ]
            
            all_paths = []
            for design_info in designs:
                report = generate_design_timing_report(
                    design_info['name'],
                    design_info['category'],
                    design_info['freq'],
                    design_info['violations']
                )
                # Convert dataclass to dict
                from dataclasses import asdict
                paths = [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in report.paths]
                all_paths.extend(paths)
            
            # Split data
            np.random.seed(42)
            np.random.shuffle(all_paths)
            split_idx = int(0.8 * len(all_paths))
            train_paths = all_paths[:split_idx]
            test_paths = all_paths[split_idx:]
            
            # Train and evaluate
            agent = TimingLLMAgent()
            
            # Measure training time
            start_time = time.time()
            agent.train(train_paths)
            train_time = time.time() - start_time
            
            # Measure inference time
            start_time = time.time()
            eval_results = evaluate_agent(agent, test_paths)
            inference_time = time.time() - start_time
            
            avg_inference_per_path = inference_time / len(test_paths) * 1000  # ms
            
            results.append({
                'complexity': level['name'],
                'num_violations': level['violations'] * 3,  # 3 designs
                'f1_score': eval_results['overall']['f1'],
                'precision': eval_results['overall']['precision'],
                'recall': eval_results['overall']['recall'],
                'train_time_sec': round(train_time, 3),
                'inference_time_ms_per_path': round(avg_inference_per_path, 2),
            })
        
        # Print results table
        print("\n  Scalability Results:")
        print(f"  {'Complexity':<12} {'Violations':>12} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Infer(ms)':>12}")
        print(f"  {'-'*64}")
        for r in results:
            print(f"  {r['complexity']:<12} {r['num_violations']:>12} {r['f1_score']:>8.3f} "
                  f"{r['precision']:>8.3f} {r['recall']:>8.3f} {r['inference_time_ms_per_path']:>12.2f}")
        
        self.results['scalability'] = results
        return results
    
    def _run_cross_domain_transfer(self):
        """
        Experiment 2: Cross-Domain Transfer Learning
        
        Train on one application domain, test on another.
        Evaluates generalization capability of the model.
        """
        print("\n  Evaluating cross-domain transfer learning...")
        
        # Generate domain-specific datasets
        domains = {
            'networking': [
                {"name": "eth_mac", "freq": 350.0, "violations": 60},
                {"name": "pcie_bridge", "freq": 400.0, "violations": 70},
                {"name": "switch_fabric", "freq": 300.0, "violations": 50},
            ],
            'signal_processing': [
                {"name": "fft_pipe", "freq": 400.0, "violations": 65},
                {"name": "beamformer", "freq": 350.0, "violations": 75},
                {"name": "filter_bank", "freq": 380.0, "violations": 55},
            ],
            'ai_accelerator': [
                {"name": "cnn_engine", "freq": 350.0, "violations": 70},
                {"name": "transformer", "freq": 300.0, "violations": 80},
                {"name": "gemm_array", "freq": 400.0, "violations": 60},
            ],
        }
        
        # Generate paths for each domain
        domain_paths = {}
        for domain, designs in domains.items():
            paths = []
            for design_info in designs:
                report = generate_design_timing_report(
                    design_info['name'],
                    domain,
                    design_info['freq'],
                    design_info['violations']
                )
                from dataclasses import asdict
                paths.extend([asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in report.paths])
            domain_paths[domain] = paths
        
        # Cross-domain evaluation matrix
        transfer_matrix = {}
        domain_names = list(domains.keys())
        
        for train_domain in domain_names:
            transfer_matrix[train_domain] = {}
            
            # Train on this domain
            train_paths = domain_paths[train_domain]
            agent = TimingLLMAgent()
            agent.train(train_paths)
            
            for test_domain in domain_names:
                # Test on each domain
                test_paths = domain_paths[test_domain]
                eval_results = evaluate_agent(agent, test_paths)
                
                transfer_matrix[train_domain][test_domain] = {
                    'f1': eval_results['overall']['f1'],
                    'precision': eval_results['overall']['precision'],
                    'recall': eval_results['overall']['recall'],
                }
        
        # Print transfer matrix
        print("\n  Cross-Domain Transfer Matrix (F1 Scores):")
        print(f"\n  {'Train \\ Test':<20}", end="")
        for d in domain_names:
            print(f"{d[:12]:>14}", end="")
        print()
        print(f"  {'-'*62}")
        
        for train_d in domain_names:
            print(f"  {train_d:<20}", end="")
            for test_d in domain_names:
                f1 = transfer_matrix[train_d][test_d]['f1']
                marker = "*" if train_d == test_d else " "
                print(f"{f1:>13.3f}{marker}", end="")
            print()
        
        print("\n  * = in-domain (train and test on same domain)")
        
        # Calculate transfer degradation
        print("\n  Transfer Degradation Analysis:")
        for train_d in domain_names:
            in_domain_f1 = transfer_matrix[train_d][train_d]['f1']
            for test_d in domain_names:
                if train_d != test_d:
                    cross_f1 = transfer_matrix[train_d][test_d]['f1']
                    degradation = (in_domain_f1 - cross_f1) / in_domain_f1 * 100
                    print(f"    {train_d} → {test_d}: {degradation:+.1f}% degradation")
        
        self.results['cross_domain_transfer'] = transfer_matrix
        return transfer_matrix
    
    def _generate_extended_figures(self):
        """Generate figures for extended experiments."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Figure 1: Scalability Analysis
        if 'scalability' in self.results:
            data = self.results['scalability']
            violations = [d['num_violations'] for d in data]
            f1_scores = [d['f1_score'] for d in data]
            inference_times = [d['inference_time_ms_per_path'] for d in data]
            
            ax1 = axes[0]
            ax1_twin = ax1.twinx()
            
            line1 = ax1.plot(violations, f1_scores, 'b-o', linewidth=2, markersize=8, label='F1 Score')
            ax1.set_xlabel('Number of Violations', fontsize=11)
            ax1.set_ylabel('F1 Score', color='blue', fontsize=11)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0.7, 0.95)
            ax1.axhline(y=0.8, color='blue', linestyle='--', alpha=0.3)
            
            line2 = ax1_twin.plot(violations, inference_times, 'r-s', linewidth=2, markersize=8, label='Inference Time')
            ax1_twin.set_ylabel('Inference Time (ms/path)', color='red', fontsize=11)
            ax1_twin.tick_params(axis='y', labelcolor='red')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')
            ax1.set_title('Scalability: Performance vs Design Complexity', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Figure 2: Cross-Domain Transfer Heatmap
        if 'cross_domain_transfer' in self.results:
            matrix = self.results['cross_domain_transfer']
            domains = list(matrix.keys())
            
            # Create F1 matrix
            f1_matrix = np.zeros((len(domains), len(domains)))
            for i, train_d in enumerate(domains):
                for j, test_d in enumerate(domains):
                    f1_matrix[i, j] = matrix[train_d][test_d]['f1']
            
            # Short labels
            short_labels = ['Network', 'Signal Proc', 'AI Accel']
            
            im = axes[1].imshow(f1_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
            axes[1].set_xticks(range(len(domains)))
            axes[1].set_yticks(range(len(domains)))
            axes[1].set_xticklabels(short_labels, fontsize=10)
            axes[1].set_yticklabels(short_labels, fontsize=10)
            axes[1].set_xlabel('Test Domain', fontsize=11)
            axes[1].set_ylabel('Train Domain', fontsize=11)
            axes[1].set_title('Cross-Domain Transfer (F1 Score)', fontsize=12, fontweight='bold')
            
            # Add text annotations
            for i in range(len(domains)):
                for j in range(len(domains)):
                    text = axes[1].text(j, i, f'{f1_matrix[i, j]:.2f}',
                                       ha='center', va='center', color='black', fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=axes[1], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'extended_experiments.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'extended_experiments.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  Saved extended figures to {self.output_dir}")
    
    def _save_results(self):
        """Save all extended experiment results."""
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        results_serializable = convert_to_serializable(self.results)
        
        with open(self.output_dir / 'extended_experiments.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Generate LaTeX tables
        self._generate_latex_tables()
    
    def _generate_latex_tables(self):
        """Generate LaTeX tables for the paper."""
        
        print("\n" + "=" * 60)
        print("LaTeX Tables for Extended Experiments")
        print("=" * 60)
        
        # Table 1: Scalability Results
        if 'scalability' in self.results:
            print("\n% Table: Scalability Analysis")
            print("\\begin{table}[t]")
            print("\\centering")
            print("\\caption{Scalability Analysis: Performance vs Design Complexity}")
            print("\\label{tab:scalability}")
            print("\\small")
            print("\\begin{tabular}{lcccc}")
            print("\\toprule")
            print("\\textbf{Complexity} & \\textbf{Violations} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Time (ms)} \\\\")
            print("\\midrule")
            
            for r in self.results['scalability']:
                print(f"{r['complexity']} & {r['num_violations']} & {r['f1_score']:.3f} & "
                      f"{r['precision']:.3f} & {r['inference_time_ms_per_path']:.2f} \\\\")
            
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")
        
        # Table 2: Cross-Domain Transfer
        if 'cross_domain_transfer' in self.results:
            print("\n% Table: Cross-Domain Transfer Learning")
            print("\\begin{table}[t]")
            print("\\centering")
            print("\\caption{Cross-Domain Transfer: F1 Scores (Train $\\rightarrow$ Test)}")
            print("\\label{tab:transfer}")
            print("\\small")
            print("\\begin{tabular}{lccc}")
            print("\\toprule")
            print("\\textbf{Train Domain} & \\textbf{Network} & \\textbf{Signal} & \\textbf{AI} \\\\")
            print("\\midrule")
            
            matrix = self.results['cross_domain_transfer']
            name_map = {'networking': 'Networking', 'signal_processing': 'Signal Proc.', 'ai_accelerator': 'AI Accel.'}
            
            for train_d in matrix.keys():
                row = [name_map[train_d]]
                for test_d in matrix.keys():
                    f1 = matrix[train_d][test_d]['f1']
                    if train_d == test_d:
                        row.append(f"\\textbf{{{f1:.2f}}}")
                    else:
                        row.append(f"{f1:.2f}")
                print(" & ".join(row) + " \\\\")
            
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")


if __name__ == "__main__":
    experiments = ExtendedExperiments()
    results = experiments.run_all()
