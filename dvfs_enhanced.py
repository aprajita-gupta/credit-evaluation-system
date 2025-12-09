"""
Enhanced DVFS Module with GPU Support and DRL-based Control
Based on research papers:
1. Framework for ML Workflow in DVFS (IJARET)
2. GPU DVFS Impact on Deep Learning (arXiv:1905.11012)

Features:
- GPU DVFS support (NVIDIA A100)
- CPU DVFS optimization
- DRL-based adaptive control
- Comprehensive energy metrics
- Thermal management
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import json


@dataclass
class DVFSProfile:
    """DVFS Configuration Profile"""
    name: str
    core_freq_ghz: float  # Core frequency in GHz
    mem_freq_ghz: float   # Memory frequency in GHz
    voltage_v: float      # Voltage in Volts
    description: str


# GPU DVFS Profiles based on NVIDIA A100 specifications
# Based on research: optimal frequencies are not always at maximum
GPU_DVFS_PROFILES = {
    'power_save': DVFSProfile(
        name='Power Save',
        core_freq_ghz=0.8,
        mem_freq_ghz=0.8,
        voltage_v=0.85,
        description='Maximum energy efficiency, 19.6-26.4% energy savings'
    ),
    'balanced': DVFSProfile(
        name='Balanced',
        core_freq_ghz=1.0,
        mem_freq_ghz=1.0,
        voltage_v=1.0,
        description='Optimal for most workloads, 8.7-23.1% energy savings'
    ),
    'performance': DVFSProfile(
        name='Performance',
        core_freq_ghz=1.2,
        mem_freq_ghz=1.1,
        voltage_v=1.15,
        description='17.4-38.2% performance improvement for training'
    ),
    'max_performance': DVFSProfile(
        name='Max Performance',
        core_freq_ghz=1.5,
        mem_freq_ghz=1.2,
        voltage_v=1.25,
        description='22.5-33.0% performance improvement for inference'
    )
}


class DVFSEnergyModel:
    """
    Advanced DVFS Energy Model supporting both CPU and GPU
    Implements physics-based power modeling from research papers
    """
    
    def __init__(self, device_type='CPU', gpu_model='A100'):
        self.device_type = device_type
        self.gpu_model = gpu_model
        
        # CPU Parameters (existing)
        self.cpu_base_power = 15.0  # Watts
        self.cpu_idle_power = 5.0
        self.cpu_power_exponent = 2.5
        
        # GPU Parameters (A100 typical values)
        if device_type == 'GPU':
            if gpu_model == 'A100':
                self.gpu_base_power = 250.0  # A100 TDP: 250W (10GB model)
                self.gpu_idle_power = 50.0
                self.gpu_power_exponent = 2.8  # GPU power scales more with frequency
                self.gpu_mem_power_ratio = 0.3  # 30% of power from memory
            else:
                # Default GPU values
                self.gpu_base_power = 200.0
                self.gpu_idle_power = 40.0
                self.gpu_power_exponent = 2.7
                self.gpu_mem_power_ratio = 0.25
    
    def compute_cpu_energy(self, time_seconds: float, dvfs_level: float, 
                           utilization: float = 0.8) -> Dict[str, float]:
        """
        Compute CPU energy consumption
        
        Formula: P = P_dynamic + P_static
        P_dynamic = P_base × (freq^α) × utilization
        P_static = P_idle × (1 + 0.1 × (freq - 1))
        E = P × t
        """
        # Time adjustment (inversely proportional to frequency)
        adjusted_time = time_seconds / dvfs_level
        
        # Dynamic power
        dynamic_power = self.cpu_base_power * (dvfs_level ** self.cpu_power_exponent) * utilization
        
        # Static power (leakage)
        static_power = self.cpu_idle_power * (1 + 0.1 * (dvfs_level - 1.0))
        
        # Total
        total_power = dynamic_power + static_power
        energy = total_power * adjusted_time
        
        return {
            'energy_joules': energy,
            'time_seconds': adjusted_time,
            'power_watts': total_power,
            'dynamic_power': dynamic_power,
            'static_power': static_power,
            'efficiency_score': 1.0 / energy if energy > 0 else 0
        }
    
    def compute_gpu_energy(self, time_seconds: float, profile: DVFSProfile, 
                           gpu_utilization: float = 0.9,
                           memory_utilization: float = 0.7) -> Dict[str, float]:
        """
        Compute GPU energy consumption with separate core and memory frequency
        
        Based on research: GPU has two frequency domains
        - Core frequency (f_core): ALU cores
        - Memory frequency (f_mem): SDRAM module
        
        Formula: P = P_core + P_mem + P_static
        """
        # Core frequency impact on time (more significant)
        core_speedup = profile.core_freq_ghz / 1.0  # Relative to baseline
        mem_speedup = profile.mem_freq_ghz / 1.0
        
        # Overall speedup (weighted average, core has more impact)
        effective_speedup = 0.7 * core_speedup + 0.3 * mem_speedup
        adjusted_time = time_seconds / effective_speedup
        
        # Core power (scales with core frequency)
        core_power = (1 - self.gpu_mem_power_ratio) * self.gpu_base_power * \
                     (profile.core_freq_ghz ** self.gpu_power_exponent) * gpu_utilization
        
        # Memory power (scales with memory frequency)
        mem_power = self.gpu_mem_power_ratio * self.gpu_base_power * \
                    (profile.mem_freq_ghz ** 2.5) * memory_utilization
        
        # Static power (voltage-dependent leakage)
        voltage_factor = (profile.voltage_v / 1.0) ** 2
        static_power = self.gpu_idle_power * voltage_factor
        
        # Total power
        total_power = core_power + mem_power + static_power
        energy = total_power * adjusted_time
        
        # Thermal estimation (simplified)
        thermal_power = total_power * 0.95  # 95% converts to heat
        
        return {
            'energy_joules': energy,
            'time_seconds': adjusted_time,
            'power_watts': total_power,
            'core_power': core_power,
            'memory_power': mem_power,
            'static_power': static_power,
            'thermal_watts': thermal_power,
            'efficiency_score': 1.0 / energy if energy > 0 else 0,
            'profile_name': profile.name,
            'core_freq_ghz': profile.core_freq_ghz,
            'mem_freq_ghz': profile.mem_freq_ghz
        }


class DRLDVFSController:
    """
    Deep Reinforcement Learning-based DVFS Controller
    
    Implements adaptive DVFS control using Q-learning approximation
    Learns optimal frequency selection based on workload characteristics
    
    Based on: Framework for ML Workflow in DVFS (IJARET 2023)
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table: state -> action -> Q-value
        # State: (workload_type, current_power_level, performance_target)
        # Action: DVFS profile selection
        self.q_table = {}
        
        # History for learning
        self.history = deque(maxlen=100)
        
        # Performance metrics
        self.total_energy_saved = 0
        self.decisions_made = 0
    
    def get_state_key(self, workload_intensity: float, power_budget: float, 
                      perf_requirement: float) -> str:
        """Discretize continuous state into bins for Q-table"""
        workload_bin = int(workload_intensity * 10) // 2  # 0-4
        power_bin = int(power_budget / 50)  # bins of 50W
        perf_bin = int(perf_requirement * 10) // 2  # 0-4
        return f"{workload_bin}_{power_bin}_{perf_bin}"
    
    def select_profile(self, workload_intensity: float, power_budget: float,
                       perf_requirement: float, device_type='GPU') -> str:
        """
        Select optimal DVFS profile using epsilon-greedy DRL strategy
        
        Args:
            workload_intensity: 0-1, how compute-intensive the task is
            power_budget: Maximum power allowed (Watts)
            perf_requirement: 0-1, performance importance (0=energy first, 1=perf first)
        
        Returns:
            Profile name key
        """
        state = self.get_state_key(workload_intensity, power_budget, perf_requirement)
        
        # Initialize Q-values for new states
        if state not in self.q_table:
            if device_type == 'GPU':
                self.q_table[state] = {k: 0.0 for k in GPU_DVFS_PROFILES.keys()}
            else:
                self.q_table[state] = {
                    'low': 0.0, 'medium': 0.0, 'high': 0.0, 'max': 0.0
                }
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            profile = np.random.choice(list(self.q_table[state].keys()))
        else:
            # Exploit: best known action
            profile = max(self.q_table[state], key=self.q_table[state].get)
        
        self.decisions_made += 1
        return profile
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """
        Update Q-value based on observed reward
        
        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
        """
        if state not in self.q_table or next_state not in self.q_table:
            return
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def compute_reward(self, energy_used: float, time_taken: float,
                       power_budget: float, time_budget: float) -> float:
        """
        Compute reward for RL agent
        
        Reward balances:
        - Energy efficiency (negative reward for high energy)
        - Performance (negative reward for exceeding time budget)
        - Power compliance (penalty for exceeding power budget)
        """
        # Energy efficiency term (normalized)
        energy_term = -energy_used / 1000.0  # Normalize to reasonable scale
        
        # Performance term
        if time_budget > 0:
            time_penalty = -max(0, (time_taken - time_budget) / time_budget) * 10
        else:
            time_penalty = 0
        
        # Power compliance term
        power_penalty = -max(0, (energy_used / time_taken - power_budget)) / 10.0
        
        # Total reward
        reward = energy_term + time_penalty + power_penalty
        
        return reward
    
    def get_statistics(self) -> Dict:
        """Get controller performance statistics"""
        return {
            'decisions_made': self.decisions_made,
            'q_table_size': len(self.q_table),
            'exploration_rate': self.epsilon,
            'average_q_value': np.mean([max(v.values()) for v in self.q_table.values()]) 
                              if self.q_table else 0
        }


class DVFSStatistics:
    """
    Comprehensive DVFS Statistics Collection
    Tracks all metrics mentioned in research papers
    """
    
    def __init__(self):
        self.measurements = []
    
    def add_measurement(self, model_name: str, device_type: str, 
                       profile_name: str, metrics: Dict):
        """Add a measurement record"""
        record = {
            'model': model_name,
            'device': device_type,
            'profile': profile_name,
            'timestamp': time.time(),
            **metrics
        }
        self.measurements.append(record)
    
    def get_comparison_table(self) -> Dict[str, Dict]:
        """
        Generate comparison table showing energy savings vs default
        Similar to tables in research papers
        """
        if not self.measurements:
            return {}
        
        comparison = {}
        
        # Find baseline (balanced profile typically)
        baseline_energy = {}
        for m in self.measurements:
            if m['profile'] == 'Balanced' or m['profile'] == 'balanced':
                baseline_energy[m['model']] = m['energy_joules']
        
        # Compute savings for each model and profile
        for m in self.measurements:
            model = m['model']
            profile = m['profile']
            
            if model not in comparison:
                comparison[model] = {}
            
            baseline = baseline_energy.get(model, m['energy_joules'])
            energy_savings_pct = ((baseline - m['energy_joules']) / baseline * 100) if baseline > 0 else 0
            time_change_pct = ((m['time_seconds'] - 1.0) / 1.0 * 100)  # Assuming 1s baseline
            
            comparison[model][profile] = {
                'energy_j': m['energy_joules'],
                'time_s': m['time_seconds'],
                'power_w': m['power_watts'],
                'energy_savings_%': energy_savings_pct,
                'time_change_%': time_change_pct,
                'efficiency_score': m.get('efficiency_score', 0)
            }
        
        return comparison
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report with key findings
        Similar to conclusions in research papers
        """
        if not self.measurements:
            return {'status': 'No data collected'}
        
        # Aggregate statistics
        total_measurements = len(self.measurements)
        unique_models = len(set(m['model'] for m in self.measurements))
        unique_profiles = len(set(m['profile'] for m in self.measurements))
        
        # Energy statistics
        energies = [m['energy_joules'] for m in self.measurements]
        times = [m['time_seconds'] for m in self.measurements]
        powers = [m['power_watts'] for m in self.measurements]
        
        # Find optimal profile (best energy efficiency)
        optimal = max(self.measurements, key=lambda x: x.get('efficiency_score', 0))
        
        # Compute average savings vs default
        comparison = self.get_comparison_table()
        avg_savings = []
        for model_data in comparison.values():
            for profile_data in model_data.values():
                if profile_data['energy_savings_%'] > 0:
                    avg_savings.append(profile_data['energy_savings_%'])
        
        report = {
            'overview': {
                'total_measurements': total_measurements,
                'models_tested': unique_models,
                'profiles_tested': unique_profiles
            },
            'energy_metrics': {
                'min_energy_j': min(energies),
                'max_energy_j': max(energies),
                'avg_energy_j': np.mean(energies),
                'total_energy_j': sum(energies)
            },
            'performance_metrics': {
                'min_time_s': min(times),
                'max_time_s': max(times),
                'avg_time_s': np.mean(times)
            },
            'power_metrics': {
                'min_power_w': min(powers),
                'max_power_w': max(powers),
                'avg_power_w': np.mean(powers)
            },
            'optimal_configuration': {
                'model': optimal['model'],
                'profile': optimal['profile'],
                'energy_j': optimal['energy_joules'],
                'time_s': optimal['time_seconds'],
                'power_w': optimal['power_watts']
            },
            'savings_analysis': {
                'avg_energy_savings_%': np.mean(avg_savings) if avg_savings else 0,
                'max_energy_savings_%': max(avg_savings) if avg_savings else 0,
                'models_with_savings': len([s for s in avg_savings if s > 0])
            },
            'comparison_table': comparison
        }
        
        return report
    
    def export_to_json(self, filepath: str):
        """Export statistics to JSON file"""
        report = self.generate_summary_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def estimate_workload_intensity(model_type: str, n_features: int, n_samples: int) -> float:
    """
    Estimate workload computational intensity
    Used by DRL controller for decision making
    
    Returns value 0-1 indicating compute intensity
    """
    # Base complexity by model type
    complexity_map = {
        'DNN': 0.9,  # High compute
        'SVM': 0.7,  # Medium-high
        'Decision Tree': 0.4,  # Medium
        'Logistic Regression': 0.3  # Low
    }
    
    base_complexity = complexity_map.get(model_type, 0.5)
    
    # Adjust for dataset size
    size_factor = min(1.0, (n_features * n_samples) / 100000)
    
    return min(1.0, base_complexity * (0.7 + 0.3 * size_factor))


# Export main classes and functions
__all__ = [
    'DVFSProfile',
    'GPU_DVFS_PROFILES',
    'DVFSEnergyModel',
    'DRLDVFSController',
    'DVFSStatistics',
    'estimate_workload_intensity'
]