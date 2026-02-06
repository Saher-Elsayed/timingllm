#!/usr/bin/env python3
"""
FPGA Timing Violation Dataset Generator
=======================================

Generates synthetic but realistic FPGA timing violations for TimingLLM experiments.
Simulates timing reports with characteristics based on modern FPGA architectures.

Author: Saher Elsayed
Institution: University of Pennsylvania
"""

import numpy as np
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


class ViolationType(Enum):
    """Types of timing violations that can occur in FPGA designs."""
    CLOCK_DOMAIN_CROSSING = "clock_domain_crossing"
    MISSING_CONSTRAINT = "missing_constraint"
    ARCHITECTURAL_BOTTLENECK = "architectural_bottleneck"
    HOLD_VIOLATION = "hold_violation"
    MULTICYCLE_PATH = "multicycle_path"


class ResourceType(Enum):
    """FPGA resource types that can appear in timing paths."""
    ALM = "Adaptive Logic Module"
    DSP = "DSP Block"
    BRAM = "Block RAM"
    LUTRAM = "Distributed RAM"
    IO = "I/O Element"
    PLL = "PLL"
    HPS = "Hard Processor System"


@dataclass
class TimingPath:
    """Represents a single timing path in the design."""
    path_id: str
    source_register: str
    dest_register: str
    source_clock: str
    dest_clock: str
    logic_levels: int
    data_delay_ns: float
    clock_skew_ns: float
    setup_time_ns: float
    hold_time_ns: float
    required_time_ns: float
    arrival_time_ns: float
    slack_ns: float
    resource_types: List[str]
    violation_type: str
    is_false_path: bool
    is_multicycle: bool
    multicycle_value: int
    recommended_fix: str
    fix_category: str


@dataclass
class TimingReport:
    """Complete timing report for a design."""
    design_name: str
    device_family: str
    device_part: str
    target_frequency_mhz: float
    achieved_frequency_mhz: float
    total_paths: int
    failing_paths: int
    wns_ns: float  # Worst Negative Slack
    tns_ns: float  # Total Negative Slack
    paths: List[TimingPath]


# Generic FPGA device characteristics (vendor-neutral)
FPGA_DEVICES = [
    {"part": "FPGA-HP-500K", "family": "High-Performance", "logic_elements": 500000, "dsps": 3000, "bram_kb": 20000},
    {"part": "FPGA-HP-700K", "family": "High-Performance", "logic_elements": 700000, "dsps": 4500, "bram_kb": 30000},
    {"part": "FPGA-UHP-1M", "family": "Ultra High-Performance", "logic_elements": 1000000, "dsps": 6000, "bram_kb": 50000},
]

# Clock domain patterns
CLOCK_DOMAINS = [
    {"name": "clk_100", "freq_mhz": 100.0, "source": "PLL"},
    {"name": "clk_200", "freq_mhz": 200.0, "source": "PLL"},
    {"name": "clk_250", "freq_mhz": 250.0, "source": "PLL"},
    {"name": "clk_300", "freq_mhz": 300.0, "source": "PLL"},
    {"name": "clk_400", "freq_mhz": 400.0, "source": "PLL"},
    {"name": "clk_ddr", "freq_mhz": 533.0, "source": "Memory Controller"},
    {"name": "clk_pcie", "freq_mhz": 250.0, "source": "PCIe Hard IP"},
    {"name": "clk_eth", "freq_mhz": 156.25, "source": "Ethernet Hard IP"},
]

# Common register naming patterns
REGISTER_PREFIXES = [
    "core_inst", "datapath", "ctrl_fsm", "fifo_ctrl", "mem_ctrl",
    "dma_engine", "pcie_bridge", "eth_mac", "dsp_pipe", "axi_interconnect"
]

# SDC fix patterns for each violation type
SDC_FIX_PATTERNS = {
    ViolationType.CLOCK_DOMAIN_CROSSING: [
        'set_false_path -from [get_clocks {src_clk}] -to [get_clocks {dst_clk}]',
        'set_max_delay -from [get_registers {src_reg}] -to [get_registers {dst_reg}] {delay}',
        'set_clock_groups -asynchronous -group [get_clocks {src_clk}] -group [get_clocks {dst_clk}]'
    ],
    ViolationType.MISSING_CONSTRAINT: [
        'create_generated_clock -source [get_pins {src_pin}] -divide_by {div} [get_pins {dst_pin}]',
        'set_input_delay -clock [get_clocks {clk}] -max {delay} [get_ports {port}]',
        'set_output_delay -clock [get_clocks {clk}] -max {delay} [get_ports {port}]'
    ],
    ViolationType.MULTICYCLE_PATH: [
        'set_multicycle_path -setup -from [get_registers {src_reg}] -to [get_registers {dst_reg}] {cycles}',
        'set_multicycle_path -hold -from [get_registers {src_reg}] -to [get_registers {dst_reg}] {cycles_minus_1}'
    ],
    ViolationType.ARCHITECTURAL_BOTTLENECK: [
        '# Consider pipelining: Insert register stage after {resource}',
        '# Consider retiming: set_dont_touch false on {module}',
        '# Consider DSP inference: Restructure arithmetic in {module}'
    ],
    ViolationType.HOLD_VIOLATION: [
        'set_min_delay -from [get_registers {src_reg}] -to [get_registers {dst_reg}] {delay}',
        '# Insert buffer cells on fast path',
        'set_clock_uncertainty -hold {uncertainty} [get_clocks {clk}]'
    ]
}


def generate_register_name(prefix: str, depth: int = 3) -> str:
    """Generate a realistic hierarchical register name."""
    parts = [prefix]
    for _ in range(depth):
        parts.append(random.choice(['u_', 'i_', 'gen_', '']) + 
                    random.choice(['stage', 'pipe', 'reg', 'data', 'ctrl', 'addr']) +
                    f'[{random.randint(0, 31)}]')
    return '|'.join(parts) + '|' + random.choice(['q', 'data_out', 'dout', 'result'])


def generate_timing_path(path_id: int, violation_type: ViolationType, 
                         target_freq_mhz: float) -> TimingPath:
    """Generate a single timing path with realistic characteristics."""
    
    period_ns = 1000.0 / target_freq_mhz
    
    # Select clocks based on violation type
    if violation_type == ViolationType.CLOCK_DOMAIN_CROSSING:
        src_clk = random.choice(CLOCK_DOMAINS)
        dst_clk = random.choice([c for c in CLOCK_DOMAINS if c['name'] != src_clk['name']])
    else:
        src_clk = dst_clk = random.choice(CLOCK_DOMAINS)
    
    # Generate path characteristics
    logic_levels = random.randint(2, 15)
    
    # Resource types in path
    resources = []
    if logic_levels > 8:
        resources.append(ResourceType.DSP.value)
    if random.random() > 0.5:
        resources.append(ResourceType.BRAM.value)
    resources.append(ResourceType.ALM.value)
    
    # Calculate delays based on violation type
    base_delay = 0.3 * logic_levels + random.uniform(0.5, 2.0)
    
    if violation_type == ViolationType.ARCHITECTURAL_BOTTLENECK:
        # Long combinatorial paths
        data_delay = base_delay * random.uniform(1.5, 2.5)
        logic_levels = random.randint(10, 20)
    elif violation_type == ViolationType.CLOCK_DOMAIN_CROSSING:
        data_delay = base_delay * random.uniform(1.0, 1.5)
    else:
        data_delay = base_delay * random.uniform(0.8, 1.3)
    
    clock_skew = random.uniform(-0.2, 0.3)
    setup_time = random.uniform(0.05, 0.15)
    hold_time = random.uniform(0.02, 0.08)
    
    required_time = period_ns - setup_time
    arrival_time = data_delay + clock_skew
    slack = required_time - arrival_time
    
    # Make some paths fail (70% of generated violations are actual failures)
    if random.random() < 0.7:
        slack = -random.uniform(0.1, 2.0)
        arrival_time = required_time - slack
    
    # Generate register names
    src_prefix = random.choice(REGISTER_PREFIXES)
    dst_prefix = random.choice(REGISTER_PREFIXES)
    
    # Multicycle path settings
    is_multicycle = violation_type == ViolationType.MULTICYCLE_PATH
    multicycle_value = random.randint(2, 4) if is_multicycle else 1
    
    # Generate recommended fix
    fix_templates = SDC_FIX_PATTERNS[violation_type]
    fix_template = random.choice(fix_templates)
    
    return TimingPath(
        path_id=f"path_{path_id:05d}",
        source_register=generate_register_name(src_prefix),
        dest_register=generate_register_name(dst_prefix),
        source_clock=src_clk['name'],
        dest_clock=dst_clk['name'],
        logic_levels=logic_levels,
        data_delay_ns=round(data_delay, 3),
        clock_skew_ns=round(clock_skew, 3),
        setup_time_ns=round(setup_time, 3),
        hold_time_ns=round(hold_time, 3),
        required_time_ns=round(required_time, 3),
        arrival_time_ns=round(arrival_time, 3),
        slack_ns=round(slack, 3),
        resource_types=resources,
        violation_type=violation_type.value,
        is_false_path=violation_type == ViolationType.CLOCK_DOMAIN_CROSSING and random.random() > 0.5,
        is_multicycle=is_multicycle,
        multicycle_value=multicycle_value,
        recommended_fix=fix_template,
        fix_category=violation_type.value
    )


def generate_design_timing_report(design_name: str, design_category: str,
                                  target_freq_mhz: float, 
                                  num_violations: int = 50) -> TimingReport:
    """Generate a complete timing report for a design."""
    
    device = random.choice(FPGA_DEVICES)
    
    # Distribution of violation types based on design category
    if design_category == "networking":
        type_weights = [0.35, 0.25, 0.15, 0.15, 0.10]  # More CDC issues
    elif design_category == "signal_processing":
        type_weights = [0.15, 0.20, 0.40, 0.10, 0.15]  # More architectural bottlenecks
    else:  # ai_accelerator
        type_weights = [0.20, 0.30, 0.30, 0.10, 0.10]  # Mixed
    
    violation_types = list(ViolationType)
    
    # Generate paths
    paths = []
    for i in range(num_violations):
        v_type = random.choices(violation_types, weights=type_weights)[0]
        path = generate_timing_path(i, v_type, target_freq_mhz)
        paths.append(path)
    
    # Calculate summary statistics
    failing_paths = [p for p in paths if p.slack_ns < 0]
    wns = min(p.slack_ns for p in paths) if paths else 0
    tns = sum(p.slack_ns for p in failing_paths) if failing_paths else 0
    
    # Achieved frequency estimation
    if wns < 0:
        achieved_freq = target_freq_mhz * (1 + wns / (1000.0 / target_freq_mhz))
        achieved_freq = max(achieved_freq, target_freq_mhz * 0.7)
    else:
        achieved_freq = target_freq_mhz
    
    return TimingReport(
        design_name=design_name,
        device_family=device['family'],
        device_part=device['part'],
        target_frequency_mhz=target_freq_mhz,
        achieved_frequency_mhz=round(achieved_freq, 2),
        total_paths=num_violations * 100,  # Approximate total paths
        failing_paths=len(failing_paths),
        wns_ns=round(wns, 3),
        tns_ns=round(tns, 3),
        paths=paths
    )


def generate_dataset(num_designs: int = 12) -> List[Dict]:
    """
    Generate complete experimental dataset.
    
    Args:
        num_designs: Number of designs to generate (max 12)
        
    Returns:
        List of design dictionaries with timing reports
    """
    
    designs = [
        # Networking designs
        {"name": "eth_100g_mac", "category": "networking", "freq": 350.0, "violations": 45},
        {"name": "pcie_gen5_bridge", "category": "networking", "freq": 400.0, "violations": 62},
        {"name": "network_switch_fabric", "category": "networking", "freq": 300.0, "violations": 38},
        {"name": "rdma_nic_engine", "category": "networking", "freq": 325.0, "violations": 55},
        
        # Signal Processing designs
        {"name": "fft_4096pt_pipeline", "category": "signal_processing", "freq": 400.0, "violations": 48},
        {"name": "mimo_beamformer", "category": "signal_processing", "freq": 350.0, "violations": 72},
        {"name": "radar_pulse_compressor", "category": "signal_processing", "freq": 450.0, "violations": 58},
        {"name": "digital_filter_bank", "category": "signal_processing", "freq": 380.0, "violations": 41},
        
        # AI Accelerator designs
        {"name": "cnn_inference_engine", "category": "ai_accelerator", "freq": 350.0, "violations": 65},
        {"name": "transformer_attention", "category": "ai_accelerator", "freq": 300.0, "violations": 78},
        {"name": "gemm_systolic_array", "category": "ai_accelerator", "freq": 400.0, "violations": 52},
        {"name": "quantized_nn_accelerator", "category": "ai_accelerator", "freq": 375.0, "violations": 44},
    ]
    
    dataset = []
    for design in designs[:num_designs]:
        report = generate_design_timing_report(
            design['name'], 
            design['category'],
            design['freq'],
            design['violations']
        )
        dataset.append({
            'report': asdict(report),
            'category': design['category']
        })
    
    return dataset


if __name__ == "__main__":
    # Generate dataset
    dataset = generate_dataset(12)
    
    # Save to file
    with open('data/timing_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("TimingLLM Dataset Generator")
    print("=" * 60)
    print(f"\nGenerated {len(dataset)} design timing reports:\n")
    
    for d in dataset:
        report = d['report']
        failing = report['failing_paths']
        total = len(report['paths'])
        print(f"  {report['design_name']:30s} | {failing:3d}/{total:3d} violations | "
              f"WNS={report['wns_ns']:7.3f}ns | {d['category']}")
    
    total_violations = sum(len(d['report']['paths']) for d in dataset)
    print(f"\nTotal violations: {total_violations}")
    print(f"Dataset saved to: data/timing_dataset.json")
