#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Metrics Visualization
Observer-approved visualization for MCP reward learning and enforcement metrics
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MCPMetricsVisualizer:
    """
    Observer-approved MCP metrics visualizer
    Shows MCP usage, enforcement, and learning progression over generations
    """
    
    def __init__(self, output_dir: str = "mcp_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization settings
        plt.style.use('default')
        self.colors = {
            'success': '#2E8B57',      # Sea Green
            'failure': '#DC143C',      # Crimson
            'gaming': '#FF4500',       # Orange Red
            'enforcement': '#4169E1',   # Royal Blue
            'learning': '#9370DB'      # Medium Purple
        }
        
        logger.info("MCP Metrics Visualizer initialized")
    
    def plot_mcp_usage_over_generations(
        self, 
        generation_data: List[Dict[str, Any]], 
        title: str = "MCP Usage Over Generations"
    ) -> str:
        """Plot MCP usage metrics over generations"""
        try:
            if not generation_data:
                logger.warning("No generation data provided for MCP usage plot")
                return ""
            
            # Extract data
            generations = [data.get('generation', i) for i, data in enumerate(generation_data)]
            mcp_calls = [data.get('mcp_calls', 0) for data in generation_data]
            success_rates = [data.get('success_rate', 0) for data in generation_data]
            enforcement_rates = [data.get('enforcement_rate', 0) for data in generation_data]
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot 1: MCP Calls per Generation
            ax1.plot(generations, mcp_calls, marker='o', linewidth=2, 
                    color=self.colors['success'], label='MCP Calls')
            ax1.set_ylabel('MCP Calls', fontweight='bold')
            ax1.set_title('MCP Usage Frequency')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Success Rate
            ax2.plot(generations, success_rates, marker='s', linewidth=2, 
                    color=self.colors['learning'], label='Success Rate')
            ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
            ax2.set_ylabel('Success Rate', fontweight='bold')
            ax2.set_title('MCP Success Rate Over Time')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Enforcement Rate
            ax3.plot(generations, enforcement_rates, marker='^', linewidth=2, 
                    color=self.colors['enforcement'], label='Enforcement Rate')
            ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
            ax3.set_xlabel('Generation', fontweight='bold')
            ax3.set_ylabel('Enforcement Rate', fontweight='bold')
            ax3.set_title('Anti-Hacking Enforcement Rate')
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_usage_generations_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"MCP usage plot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"MCP usage plot generation failed: {e}")
            return ""
    
    def plot_reward_learning_progression(
        self, 
        reward_history: List[Dict[str, Any]], 
        title: str = "MCP Reward Learning Progression"
    ) -> str:
        """Plot MCP reward learning progression"""
        try:
            if not reward_history:
                logger.warning("No reward history provided for learning plot")
                return ""
            
            # Extract data
            timestamps = [data.get('timestamp', datetime.now()) for data in reward_history]
            final_rewards = [data.get('final_reward', 0) for data in reward_history]
            base_rewards = [data.get('base_reward', 0) for data in reward_history]
            impact_bonuses = [data.get('impact_bonus', 0) for data in reward_history]
            hacking_penalties = [data.get('hacking_penalty', 0) for data in reward_history]
            proof_valid = [data.get('proof_valid', False) for data in reward_history]
            
            # Create time series
            time_indices = list(range(len(timestamps)))
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot 1: Reward Components Over Time
            ax1.plot(time_indices, final_rewards, linewidth=2, color=self.colors['learning'], 
                    label='Final Reward', alpha=0.8)
            ax1.plot(time_indices, base_rewards, linewidth=1, color=self.colors['success'], 
                    label='Base Reward', alpha=0.6)
            ax1.plot(time_indices, impact_bonuses, linewidth=1, color='green', 
                    label='Impact Bonus', alpha=0.6)
            ax1.plot(time_indices, hacking_penalties, linewidth=1, color=self.colors['failure'], 
                    label='Hacking Penalty', alpha=0.6)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_ylabel('Reward Value', fontweight='bold')
            ax1.set_title('MCP Reward Components Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Proof Validation Rate
            # Calculate rolling validation rate
            window_size = min(10, len(proof_valid))
            validation_rates = []
            for i in range(len(proof_valid)):
                start_idx = max(0, i - window_size + 1)
                window_valid = proof_valid[start_idx:i+1]
                validation_rate = sum(window_valid) / len(window_valid)
                validation_rates.append(validation_rate)
            
            ax2.plot(time_indices, validation_rates, linewidth=2, color=self.colors['enforcement'], 
                    label='Proof Validation Rate')
            ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
            ax2.fill_between(time_indices, validation_rates, alpha=0.3, color=self.colors['enforcement'])
            
            ax2.set_xlabel('MCP Evaluation Index', fontweight='bold')
            ax2.set_ylabel('Validation Rate', fontweight='bold')
            ax2.set_title('Proof Validation Rate (Rolling Average)')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_reward_learning_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"MCP reward learning plot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"MCP reward learning plot generation failed: {e}")
            return ""
    
    def plot_enforcement_effectiveness(
        self, 
        enforcement_data: Dict[str, Any], 
        title: str = "MCP Anti-Hacking Enforcement Effectiveness"
    ) -> str:
        """Plot enforcement effectiveness metrics"""
        try:
            if not enforcement_data:
                logger.warning("No enforcement data provided")
                return ""
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot 1: Enforcement Rate Gauge
            enforcement_rate = enforcement_data.get('enforcement_rate', 0)
            target_rate = 0.95
            
            # Create gauge-like visualization
            angles = np.linspace(0, np.pi, 100)
            values = np.linspace(0, 1, 100)
            
            ax1.plot(angles, np.ones_like(angles), color='lightgray', linewidth=10, alpha=0.3)
            
            # Color code the gauge
            for i, (angle, value) in enumerate(zip(angles, values)):
                if value <= enforcement_rate:
                    color = self.colors['success'] if value >= target_rate else self.colors['gaming']
                    ax1.plot([angle], [1], 'o', color=color, markersize=3)
            
            ax1.set_xlim(0, np.pi)
            ax1.set_ylim(0, 1.2)
            ax1.set_title(f'Enforcement Rate: {enforcement_rate:.1%}')
            ax1.text(np.pi/2, 0.5, f'{enforcement_rate:.1%}', ha='center', va='center', 
                    fontsize=20, fontweight='bold')
            ax1.set_xticks([0, np.pi/2, np.pi])
            ax1.set_xticklabels(['0%', '50%', '100%'])
            ax1.set_yticks([])
            
            # Plot 2: Gaming Detection Distribution
            gaming_types = enforcement_data.get('gaming_types_detected', {})
            if gaming_types:
                labels = list(gaming_types.keys())
                sizes = list(gaming_types.values())
                colors = [self.colors['failure'], self.colors['gaming'], self.colors['enforcement']][:len(labels)]
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Gaming Types Detected')
            else:
                ax2.text(0.5, 0.5, 'No Gaming\nDetected', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14, fontweight='bold')
                ax2.set_title('Gaming Detection')
            
            # Plot 3: Penalty Distribution
            penalty_data = enforcement_data.get('penalty_distribution', {})
            if penalty_data:
                penalty_types = list(penalty_data.keys())
                penalty_counts = list(penalty_data.values())
                
                bars = ax3.bar(penalty_types, penalty_counts, 
                              color=[self.colors['failure'], self.colors['gaming'], self.colors['enforcement']])
                ax3.set_ylabel('Count', fontweight='bold')
                ax3.set_title('Penalty Distribution')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No Penalties\nApplied', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14, fontweight='bold')
                ax3.set_title('Penalty Distribution')
            
            # Plot 4: Learning Effectiveness Over Time
            learning_metrics = enforcement_data.get('learning_progression', [])
            if learning_metrics:
                time_points = list(range(len(learning_metrics)))
                effectiveness = [m.get('effectiveness', 0) for m in learning_metrics]
                
                ax4.plot(time_points, effectiveness, marker='o', linewidth=2, 
                        color=self.colors['learning'], label='Learning Effectiveness')
                ax4.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
                ax4.set_xlabel('Time Period', fontweight='bold')
                ax4.set_ylabel('Effectiveness', fontweight='bold')
                ax4.set_title('Learning Effectiveness Progression')
                ax4.set_ylim(0, 1.1)
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Learning\nProgression\nData\nUnavailable', 
                        ha='center', va='center', transform=ax4.transAxes, 
                        fontsize=12, fontweight='bold')
                ax4.set_title('Learning Effectiveness')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_enforcement_effectiveness_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"MCP enforcement effectiveness plot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"MCP enforcement effectiveness plot generation failed: {e}")
            return ""
    
    def generate_mcp_learning_report(
        self, 
        generation_data: List[Dict[str, Any]], 
        reward_history: List[Dict[str, Any]], 
        enforcement_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive MCP learning report with all visualizations"""
        try:
            logger.info("Generating comprehensive MCP learning report...")
            
            # Generate all plots
            usage_plot = self.plot_mcp_usage_over_generations(generation_data)
            learning_plot = self.plot_reward_learning_progression(reward_history)
            enforcement_plot = self.plot_enforcement_effectiveness(enforcement_data)
            
            # Create summary report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"mcp_learning_report_{timestamp}.txt"
            report_filepath = os.path.join(self.output_dir, report_filename)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write("🎯 OBSERVER MCP LEARNING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Usage summary
                if generation_data:
                    total_gens = len(generation_data)
                    avg_calls = sum(g.get('mcp_calls', 0) for g in generation_data) / total_gens
                    avg_success = sum(g.get('success_rate', 0) for g in generation_data) / total_gens
                    avg_enforcement = sum(g.get('enforcement_rate', 0) for g in generation_data) / total_gens
                    
                    f.write("📊 USAGE SUMMARY:\n")
                    f.write(f"   Total Generations: {total_gens}\n")
                    f.write(f"   Average MCP Calls/Gen: {avg_calls:.1f}\n")
                    f.write(f"   Average Success Rate: {avg_success:.1%}\n")
                    f.write(f"   Average Enforcement Rate: {avg_enforcement:.1%}\n\n")
                
                # Learning summary
                if reward_history:
                    total_evaluations = len(reward_history)
                    avg_reward = sum(r.get('final_reward', 0) for r in reward_history) / total_evaluations
                    valid_proofs = sum(1 for r in reward_history if r.get('proof_valid', False))
                    validation_rate = valid_proofs / total_evaluations
                    
                    f.write("🧠 LEARNING SUMMARY:\n")
                    f.write(f"   Total MCP Evaluations: {total_evaluations}\n")
                    f.write(f"   Average Final Reward: {avg_reward:.3f}\n")
                    f.write(f"   Proof Validation Rate: {validation_rate:.1%}\n")
                    f.write(f"   Learning Effectiveness: {'✅ HIGH' if validation_rate >= 0.9 else '⚠️ MODERATE'}\n\n")
                
                # Enforcement summary
                if enforcement_data:
                    enforcement_rate = enforcement_data.get('enforcement_rate', 0)
                    gaming_detected = enforcement_data.get('gaming_attempts_detected', 0)
                    
                    f.write("🛡️ ENFORCEMENT SUMMARY:\n")
                    f.write(f"   Enforcement Rate: {enforcement_rate:.1%}\n")
                    f.write(f"   Gaming Attempts Detected: {gaming_detected}\n")
                    f.write(f"   Target Achievement: {'✅ ACHIEVED' if enforcement_rate >= 0.95 else '⚠️ PARTIAL'}\n\n")
                
                # Generated files
                f.write("📁 GENERATED FILES:\n")
                if usage_plot:
                    f.write(f"   Usage Plot: {os.path.basename(usage_plot)}\n")
                if learning_plot:
                    f.write(f"   Learning Plot: {os.path.basename(learning_plot)}\n")
                if enforcement_plot:
                    f.write(f"   Enforcement Plot: {os.path.basename(enforcement_plot)}\n")
                f.write(f"   Report: {os.path.basename(report_filepath)}\n")
            
            logger.info(f"MCP learning report generated: {report_filepath}")
            return report_filepath
            
        except Exception as e:
            logger.error(f"MCP learning report generation failed: {e}")
            return ""
