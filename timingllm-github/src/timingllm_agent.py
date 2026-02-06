#!/usr/bin/env python3
"""
TimingLLM Core Framework
========================

RAG-Augmented LLM for FPGA Timing Closure.
Implements timing violation diagnosis and automated fix recommendation.

Author: Saher Elsayed
Institution: University of Pennsylvania
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


@dataclass
class DiagnosisResult:
    """Result of timing violation diagnosis."""
    path_id: str
    predicted_type: str
    confidence: float
    reasoning: str
    recommended_fix: str
    fix_parameters: Dict


class TimingKnowledgeBase:
    """
    Simulated RAG Knowledge Base for FPGA Timing.
    
    Contains:
    - Device specifications and timing models
    - SDC constraint patterns
    - Historical successful fixes
    """
    
    def __init__(self):
        self.device_specs = self._load_device_specs()
        self.sdc_patterns = self._load_sdc_patterns()
        self.historical_fixes = self._load_historical_fixes()
    
    def _load_device_specs(self) -> Dict:
        """Load generic FPGA device timing specifications."""
        return {
            "high_performance": {
                "logic_delay_ns": 0.180,
                "dsp_delay_ns": 1.200,
                "bram_delay_ns": 1.450,
                "routing_delay_per_hop_ns": 0.120,
                "io_delay_ns": 0.850,
                "pll_jitter_ps": 45,
                "max_fanout_recommended": 16,
            },
            "ultra_high_performance": {
                "logic_delay_ns": 0.150,
                "dsp_delay_ns": 1.000,
                "bram_delay_ns": 1.250,
                "routing_delay_per_hop_ns": 0.100,
                "io_delay_ns": 0.700,
                "pll_jitter_ps": 35,
                "max_fanout_recommended": 20,
            }
        }
    
    def _load_sdc_patterns(self) -> Dict:
        """Load SDC constraint patterns for common scenarios."""
        return {
            "clock_domain_crossing": {
                "false_path": "set_false_path -from [get_clocks {src}] -to [get_clocks {dst}]",
                "async_groups": "set_clock_groups -asynchronous -group {src} -group {dst}",
                "max_delay": "set_max_delay {delay} -from [get_clocks {src}] -to [get_clocks {dst}]",
                "indicators": ["different source clocks", "async interface", "fifo crossing"],
            },
            "missing_constraint": {
                "input_delay": "set_input_delay -clock {clk} -max {delay} [get_ports {port}]",
                "output_delay": "set_output_delay -clock {clk} -max {delay} [get_ports {port}]",
                "generated_clock": "create_generated_clock -source {src} -divide_by {div} {dst}",
                "indicators": ["unconstrained path", "no clock relationship", "port timing"],
            },
            "multicycle_path": {
                "setup": "set_multicycle_path {n} -setup -from {src} -to {dst}",
                "hold": "set_multicycle_path {n-1} -hold -from {src} -to {dst}",
                "indicators": ["enable signal", "slow data rate", "accumulator"],
            },
            "architectural_bottleneck": {
                "pipeline": "# Insert pipeline register after {stage}",
                "retiming": "# Enable retiming: set_dont_touch false -on {module}",
                "restructure": "# Restructure: Move {operation} to separate cycle",
                "indicators": ["high logic levels", "dsp chain", "wide datapath"],
            },
            "hold_violation": {
                "min_delay": "set_min_delay {delay} -from {src} -to {dst}",
                "uncertainty": "set_clock_uncertainty -hold {val} [get_clocks {clk}]",
                "indicators": ["fast path", "short routing", "clock insertion"],
            }
        }
    
    def _load_historical_fixes(self) -> List[Dict]:
        """Load historical successful fixes."""
        return [
            {"pattern": "CDC with async clocks", "fix": "set_clock_groups -asynchronous", "success_rate": 0.92},
            {"pattern": "Missing IO constraint", "fix": "set_input/output_delay", "success_rate": 0.88},
            {"pattern": "DSP chain bottleneck", "fix": "Pipeline insertion", "success_rate": 0.85},
            {"pattern": "FIFO crossing", "fix": "set_false_path for gray code", "success_rate": 0.94},
            {"pattern": "Multicycle arithmetic", "fix": "set_multicycle_path", "success_rate": 0.90},
        ]
    
    def retrieve_relevant_knowledge(self, path_info: Dict) -> Dict:
        """
        RAG retrieval: Get relevant knowledge for a timing path.
        
        Args:
            path_info: Dictionary containing timing path information
            
        Returns:
            Dictionary with retrieved device specs, patterns, and historical fixes
        """
        retrieved = {
            "device_specs": self.device_specs.get("high_performance", {}),
            "applicable_patterns": [],
            "similar_historical_fixes": []
        }
        
        # Check for CDC
        if path_info.get('source_clock') != path_info.get('dest_clock'):
            retrieved["applicable_patterns"].append(self.sdc_patterns["clock_domain_crossing"])
            retrieved["similar_historical_fixes"].append(self.historical_fixes[0])
        
        # Check for high logic levels (architectural bottleneck)
        if path_info.get('logic_levels', 0) > 10:
            retrieved["applicable_patterns"].append(self.sdc_patterns["architectural_bottleneck"])
            retrieved["similar_historical_fixes"].append(self.historical_fixes[2])
        
        # Check for potential multicycle
        if path_info.get('is_multicycle') or 'acc' in str(path_info.get('source_register', '')).lower():
            retrieved["applicable_patterns"].append(self.sdc_patterns["multicycle_path"])
            retrieved["similar_historical_fixes"].append(self.historical_fixes[4])
        
        return retrieved


class TimingViolationClassifier:
    """
    ML-based timing violation classifier.
    
    Uses path features to predict violation root cause with confidence scores.
    """
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.label_map = {
            "clock_domain_crossing": 0,
            "missing_constraint": 1,
            "architectural_bottleneck": 2,
            "hold_violation": 3,
            "multicycle_path": 4
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.is_trained = False
    
    def extract_features(self, path: Dict) -> np.ndarray:
        """
        Extract ML features from timing path.
        
        Args:
            path: Dictionary containing timing path information
            
        Returns:
            Feature vector as numpy array
        """
        features = [
            # Clock domain features
            1.0 if path.get('source_clock') != path.get('dest_clock') else 0.0,
            
            # Path complexity features
            path.get('logic_levels', 5) / 20.0,
            path.get('data_delay_ns', 2.0) / 10.0,
            abs(path.get('slack_ns', 0)) / 5.0,
            
            # Resource features
            1.0 if 'DSP Block' in str(path.get('resource_types', [])) else 0.0,
            1.0 if 'Block RAM' in str(path.get('resource_types', [])) else 0.0,
            
            # Timing characteristics
            path.get('clock_skew_ns', 0) / 1.0,
            path.get('setup_time_ns', 0.1) / 0.5,
            
            # Path type indicators
            1.0 if path.get('is_multicycle') else 0.0,
            1.0 if path.get('is_false_path') else 0.0,
        ]
        return np.array(features)
    
    def train(self, paths: List[Dict]) -> None:
        """
        Train the classifier on labeled paths.
        
        Args:
            paths: List of timing path dictionaries with 'violation_type' labels
        """
        X = np.array([self.extract_features(p) for p in paths])
        y = np.array([self.label_map.get(p.get('violation_type'), 0) for p in paths])
        self.classifier.fit(X, y)
        self.is_trained = True
    
    def predict(self, path: Dict) -> Tuple[str, float]:
        """
        Predict violation type with confidence.
        
        Args:
            path: Timing path dictionary
            
        Returns:
            Tuple of (predicted_type, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        features = self.extract_features(path).reshape(1, -1)
        proba = self.classifier.predict_proba(features)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        
        return self.reverse_label_map[pred_idx], confidence


class TimingLLMAgent:
    """
    Main TimingLLM Agent.
    
    Combines RAG knowledge retrieval with ML classification for
    automated timing violation diagnosis and fix recommendation.
    """
    
    def __init__(self):
        self.knowledge_base = TimingKnowledgeBase()
        self.classifier = TimingViolationClassifier()
    
    def train(self, training_paths: List[Dict]) -> None:
        """
        Train the agent on labeled timing violations.
        
        Args:
            training_paths: List of timing path dictionaries with labels
        """
        self.classifier.train(training_paths)
    
    def diagnose(self, path: Dict) -> DiagnosisResult:
        """
        Diagnose a timing violation using RAG + ML.
        
        Args:
            path: Timing path dictionary
            
        Returns:
            DiagnosisResult with prediction, confidence, and recommended fix
        """
        # Step 1: Retrieve relevant knowledge
        knowledge = self.knowledge_base.retrieve_relevant_knowledge(path)
        
        # Step 2: ML-based classification
        pred_type, confidence = self.classifier.predict(path)
        
        # Step 3: Generate reasoning
        reasoning = self._generate_reasoning(path, pred_type, knowledge)
        
        # Step 4: Generate fix recommendation
        fix, params = self._generate_fix(path, pred_type, knowledge)
        
        return DiagnosisResult(
            path_id=path.get('path_id', 'unknown'),
            predicted_type=pred_type,
            confidence=confidence,
            reasoning=reasoning,
            recommended_fix=fix,
            fix_parameters=params
        )
    
    def _generate_reasoning(self, path: Dict, pred_type: str, knowledge: Dict) -> str:
        """Generate human-readable reasoning for diagnosis."""
        
        reasons = []
        
        if pred_type == "clock_domain_crossing":
            reasons.append(f"Path crosses from {path.get('source_clock')} to {path.get('dest_clock')}")
            reasons.append("Asynchronous clock relationship detected")
            reasons.append("Recommend async clock group constraint or false path")
            
        elif pred_type == "architectural_bottleneck":
            reasons.append(f"High logic depth: {path.get('logic_levels')} levels")
            if 'DSP Block' in str(path.get('resource_types', [])):
                reasons.append("DSP block in critical path increases delay")
            reasons.append("Consider pipelining or retiming to reduce combinatorial depth")
            
        elif pred_type == "missing_constraint":
            reasons.append("Path appears unconstrained or under-constrained")
            reasons.append("Check for missing I/O delay or generated clock definitions")
            
        elif pred_type == "multicycle_path":
            reasons.append("Path characteristics suggest multicycle operation")
            reasons.append("Data rate slower than clock rate - multicycle constraint appropriate")
            
        elif pred_type == "hold_violation":
            reasons.append(f"Short data path with slack {path.get('slack_ns'):.3f}ns")
            reasons.append("Hold time requirement not met - path too fast")
        
        return " | ".join(reasons)
    
    def _generate_fix(self, path: Dict, pred_type: str, knowledge: Dict) -> Tuple[str, Dict]:
        """Generate SDC fix recommendation."""
        
        params = {
            "src_clk": path.get('source_clock', 'clk'),
            "dst_clk": path.get('dest_clock', 'clk'),
            "src_reg": path.get('source_register', '*'),
            "dst_reg": path.get('dest_register', '*'),
        }
        
        patterns = self.knowledge_base.sdc_patterns
        
        if pred_type == "clock_domain_crossing":
            fix = patterns["clock_domain_crossing"]["async_groups"].format(
                src=params["src_clk"], dst=params["dst_clk"]
            )
        elif pred_type == "missing_constraint":
            fix = patterns["missing_constraint"]["input_delay"].format(
                clk=params["src_clk"], delay="2.0", port="*"
            )
        elif pred_type == "multicycle_path":
            fix = patterns["multicycle_path"]["setup"].format(
                n=2, src=params["src_reg"], dst=params["dst_reg"]
            )
        elif pred_type == "architectural_bottleneck":
            fix = "# Pipeline insertion recommended after logic level 5-6"
        elif pred_type == "hold_violation":
            fix = patterns["hold_violation"]["min_delay"].format(
                delay="0.1", src=params["src_reg"], dst=params["dst_reg"]
            )
        else:
            fix = "# Manual analysis required"
        
        return fix, params
    
    def batch_diagnose(self, paths: List[Dict]) -> List[DiagnosisResult]:
        """
        Diagnose multiple timing violations.
        
        Args:
            paths: List of timing path dictionaries
            
        Returns:
            List of DiagnosisResult objects
        """
        return [self.diagnose(path) for path in paths]


def evaluate_agent(agent: TimingLLMAgent, test_paths: List[Dict]) -> Dict:
    """
    Evaluate the TimingLLM agent on test data.
    
    Args:
        agent: Trained TimingLLMAgent
        test_paths: List of timing path dictionaries with ground truth labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_true = []
    y_pred = []
    confidences = []
    
    for path in test_paths:
        result = agent.diagnose(path)
        y_true.append(path.get('violation_type'))
        y_pred.append(result.predicted_type)
        confidences.append(result.confidence)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    labels = list(set(y_true))
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            "precision": round(p_per_class[i], 3),
            "recall": round(r_per_class[i], 3),
            "f1": round(f1_per_class[i], 3)
        }
    
    return {
        "overall": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "avg_confidence": round(np.mean(confidences), 3)
        },
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels
    }


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("TimingLLM Agent Demo")
    print("=" * 60)
    
    # Load dataset
    try:
        with open('data/timing_dataset.json', 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("\nDataset not found. Generating...")
        from dataset_generator import generate_dataset
        dataset = generate_dataset(12)
        with open('data/timing_dataset.json', 'w') as f:
            json.dump(dataset, f, indent=2)
    
    # Extract all paths
    all_paths = []
    for design in dataset:
        all_paths.extend(design['report']['paths'])
    
    print(f"\nTotal paths: {len(all_paths)}")
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(all_paths)
    split_idx = int(0.8 * len(all_paths))
    train_paths = all_paths[:split_idx]
    test_paths = all_paths[split_idx:]
    
    print(f"Training: {len(train_paths)}, Testing: {len(test_paths)}")
    
    # Create and train agent
    agent = TimingLLMAgent()
    agent.train(train_paths)
    
    # Evaluate
    results = evaluate_agent(agent, test_paths)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Precision:  {results['overall']['precision']:.3f}")
    print(f"  Recall:     {results['overall']['recall']:.3f}")
    print(f"  F1 Score:   {results['overall']['f1']:.3f}")
    print(f"  Confidence: {results['overall']['avg_confidence']:.3f}")
    
    print("\nPer-Class Metrics:")
    for label, metrics in results['per_class'].items():
        print(f"  {label:30s}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # Save results
    with open('results/classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/classification_results.json")
