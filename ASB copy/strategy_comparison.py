#!/usr/bin/env python3
"""
Strategy Comparison Visualization: Daily Settlement Analysis

This script compares the neural network strategy with the linear strategy
on the same price path, showing daily settlement numbers and strategy differences.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import the neural networks
try:
    from ASB.neural_networks import ASREnv, ASREnv_Linear
    from ASB.market_process import update_cash_spent
except ImportError:
    from neural_networks import ASREnv, ASREnv_Linear
    from market_process import update_cash_spent


class StrategyComparator:
    """Compare neural network strategy with linear strategy on the same paths."""
    
    def __init__(self, 
                 F: float = 900_000_000.0,
                 N: int = 63,
                 S0: float = 45.0,
                 num_paths: int = 1000,
                 early_exercise_start: int = 22,
                 gamma: float = 2.5e-7):
        """Initialize the strategy comparator."""
        
        self.F = F
        self.N = N
        self.S0 = S0
        self.early_exercise_start = early_exercise_start
        
        # Create both models with the same parameters
        self.neural_env = ASREnv(
            F=F, N=N, S0=S0, num_paths=num_paths, 
            early_exercise_start=early_exercise_start, gamma=gamma
        )
        
        self.linear_env = ASREnv_Linear(
            F=F, N=N, S0=S0, num_paths=num_paths, 
            early_exercise_start=early_exercise_start, gamma=gamma
        )
        
        print("Strategy Comparator initialized!")
        print(f"Model parameters: F={F:,.0f}, N={N}, S0={S0}")
    
    def load_trained_model(self, model_path: str):
        """Load a trained neural network model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.neural_env.load_state_dict(checkpoint['asr_env_state_dict'])
            print(f"Loaded trained model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using randomly initialized networks.")
        except Exception as e:
            print(f"Warning: Error loading model: {e}. Using randomly initialized networks.")
    
    def simulate_single_path_strategies(self, path_idx: int = 0):
        """
        Simulate both strategies on a single path and return detailed daily information.
        
        Args:
            path_idx: Index of the path to analyze
            
        Returns:
            Dictionary with detailed simulation results
        """
        # Get the same path for both strategies
        stock_batch, volume_batch, avg_price_batch = self.neural_env.get_training_batch(
            batch_size=max(path_idx + 1, 10), shuffle=False
        )
        
        # Use the specified path
        stock_path = stock_batch[path_idx]  # Shape: (N+1,)
        volume_path = volume_batch[path_idx]  # Shape: (N+1,)
        avg_price_path = avg_price_batch[path_idx]  # Shape: (N+1,)
        
        # Simulate neural network strategy
        neural_results = self._simulate_strategy(
            stock_path, volume_path, avg_price_path, 
            use_neural=True, strategy_name="Neural Network"
        )
        
        # Simulate linear strategy
        linear_results = self._simulate_strategy(
            stock_path, volume_path, avg_price_path, 
            use_neural=False, strategy_name="Linear"
        )
        
        return {
            'stock_path': stock_path.numpy(),
            'volume_path': volume_path.numpy(),
            'avg_price_path': avg_price_path.numpy(),
            'neural_results': neural_results,
            'linear_results': linear_results,
            'path_idx': path_idx
        }
    
    def _simulate_strategy(self, stock_path, volume_path, avg_price_path, 
                          use_neural=True, strategy_name="Strategy"):
        """Simulate a single strategy on the given path."""
        
        device = stock_path.device
        
        # Initialize tracking variables
        total_cost = torch.tensor(0.0, device=device)
        shares_purchased = torch.tensor(0.0, device=device)
        
        # Storage for daily results
        daily_results = {
            'day': [],
            'stock_price': [],
            'avg_price': [],
            'shares_to_buy': [],
            'shares_purchased_cumulative': [],
            'cost_increment': [],
            'total_cost': [],
            'trading_rate_output': [],
            'stopping_prob': [],
            'target_cumulative_shares': [],
            'notional_progress': [],
            'daily_pnl': [],
            'cumulative_pnl': []
        }
        
        # Early exercise tracking
        exercise_day = None
        exercise_pnl = None
        
        print(f"\n{strategy_name} Strategy Simulation:")
        print("="*50)
        
        for t in range(self.N):
            current_price = stock_path[t]
            current_volume = volume_path[t]
            current_avg_price = avg_price_path[t]
            
            # Compute network inputs if using neural strategy
            if use_neural:
                trading_inputs, stopping_inputs = self.neural_env.compute_inputs(
                    current_price.unsqueeze(0), t, 
                    shares_purchased.unsqueeze(0), current_avg_price.unsqueeze(0)
                )
                
                # Trading decision
                trading_output = self.neural_env.trading_network(trading_inputs)
                target_cumulative_shares = (self.F / current_avg_price) * ((t + 1) / self.N) * (1.0 + trading_output.squeeze())
                
                # Stopping decision
                if t >= self.early_exercise_start:
                    stopping_output = self.neural_env.stopping_network(stopping_inputs)
                    stopping_prob = self.neural_env.modified_sigmoid(
                        self.neural_env.nu_phi * stopping_output
                    ).squeeze()
                else:
                    stopping_prob = torch.tensor(0.0)
                    
                trading_rate_output = trading_output.squeeze().item()
                stopping_prob_value = stopping_prob.item()
                
            else:
                # Linear strategy
                target_cumulative_shares = (self.F / current_avg_price) * ((t + 1) / self.N)
                trading_rate_output = 0.0  # No neural adjustment
                stopping_prob_value = 0.0  # No early exercise in linear strategy
            
            # Calculate shares to buy
            shares_to_buy = target_cumulative_shares - shares_purchased
            
            # Update cost
            new_total_cost = update_cash_spent(
                X_prev=total_cost.unsqueeze(0),
                v=shares_to_buy.unsqueeze(0),
                S_next=current_price.unsqueeze(0),
                V_next=current_volume.unsqueeze(0),
                dt=self.neural_env.market_process.dt,
                eta=self.neural_env.eta,
                phi=self.neural_env.phi
            )
            cost_increase = new_total_cost.squeeze() - total_cost
            
            # Check for early exercise (neural strategy only)
            if use_neural and t >= self.early_exercise_start and stopping_prob_value > 0.5:
                exercise_day = t
                shares_remaining = self.F / current_avg_price - shares_purchased
                terminal_penalty_coeff = 2e-7
                termination_cost = terminal_penalty_coeff * shares_remaining ** 2
                exercise_pnl = self.F - shares_remaining * current_price - total_cost - termination_cost
                print(f"  Early exercise at day {t}, PnL: {exercise_pnl.item():.2f}")
                break
            
            # Update tracking variables
            total_cost = new_total_cost.squeeze()
            shares_purchased += shares_to_buy
            
            # Calculate daily PnL (theoretical if exercised at this day)
            if t >= self.early_exercise_start:
                shares_remaining = self.F / current_avg_price - shares_purchased
                terminal_penalty_coeff = 2e-7
                termination_cost = terminal_penalty_coeff * shares_remaining ** 2
                daily_theoretical_pnl = self.F - shares_remaining * current_price - total_cost - termination_cost
            else:
                daily_theoretical_pnl = self.F - total_cost  # Simple PnL without early exercise
            
            cumulative_pnl = self.F - total_cost  # Running PnL if held to maturity
            
            # Store daily results
            daily_results['day'].append(t)
            daily_results['stock_price'].append(current_price.item())
            daily_results['avg_price'].append(current_avg_price.item())
            daily_results['shares_to_buy'].append(shares_to_buy.item())
            daily_results['shares_purchased_cumulative'].append(shares_purchased.item())
            daily_results['cost_increment'].append(cost_increase.item())
            daily_results['total_cost'].append(total_cost.item())
            daily_results['trading_rate_output'].append(trading_rate_output)
            daily_results['stopping_prob'].append(stopping_prob_value)
            daily_results['target_cumulative_shares'].append(target_cumulative_shares.item())
            daily_results['notional_progress'].append(
                (shares_purchased * current_avg_price / self.F).item()
            )
            daily_results['daily_pnl'].append(daily_theoretical_pnl.item())
            daily_results['cumulative_pnl'].append(cumulative_pnl.item())
            
            # Print progress every 10 days
            if t % 10 == 0:
                print(f"  Day {t:2d}: Price={current_price:.2f}, Shares={shares_to_buy:.0f}, "
                      f"Cumulative={shares_purchased:.0f}, Cost={total_cost:.0f}")
        
        # Final PnL calculation
        if exercise_day is None:
            final_pnl = self.F - total_cost
        else:
            final_pnl = exercise_pnl
            
        print(f"  Final PnL: {final_pnl.item():.2f}")
        
        return {
            'daily_results': daily_results,
            'final_pnl': final_pnl.item(),
            'exercise_day': exercise_day,
            'total_cost': total_cost.item(),
            'shares_purchased': shares_purchased.item()
        }
    
    def plot_strategy_comparison(self, comparison_results, save_path: str = None):
        """Plot comprehensive comparison of the two strategies."""
        
        neural = comparison_results['neural_results']
        linear = comparison_results['linear_results']
        
        # Create figure with more subplots
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        fig.suptitle(f'Strategy Comparison - Path {comparison_results["path_idx"]}', fontsize=16)
        
        # 1. Stock Price Path
        axes[0, 0].plot(comparison_results['stock_path'], 'b-', linewidth=2, label='Stock Price')
        axes[0, 0].plot(comparison_results['avg_price_path'], 'r--', linewidth=1, label='Avg Price (VWAP)')
        axes[0, 0].set_title('Stock Price Path')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Price (€)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Daily Shares Purchased
        days_neural = neural['daily_results']['day']
        days_linear = linear['daily_results']['day']
        
        axes[0, 1].bar([d - 0.2 for d in days_neural], neural['daily_results']['shares_to_buy'], 
                      width=0.4, label='Neural Strategy', alpha=0.7, color='blue')
        axes[0, 1].bar([d + 0.2 for d in days_linear], linear['daily_results']['shares_to_buy'], 
                      width=0.4, label='Linear Strategy', alpha=0.7, color='red')
        axes[0, 1].set_title('Daily Shares Purchased')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Shares')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative Shares
        axes[1, 0].plot(days_neural, neural['daily_results']['shares_purchased_cumulative'], 
                       'b-', linewidth=2, label='Neural Strategy')
        axes[1, 0].plot(days_linear, linear['daily_results']['shares_purchased_cumulative'], 
                       'r--', linewidth=2, label='Linear Strategy')
        axes[1, 0].set_title('Cumulative Shares Purchased')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Cumulative Shares')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Notional Progress
        axes[1, 1].plot(days_neural, [p * 100 for p in neural['daily_results']['notional_progress']], 
                       'b-', linewidth=2, label='Neural Strategy')
        axes[1, 1].plot(days_linear, [p * 100 for p in linear['daily_results']['notional_progress']], 
                       'r--', linewidth=2, label='Linear Strategy')
        axes[1, 1].axhline(y=100, color='black', linestyle=':', alpha=0.5, label='Target (100%)')
        axes[1, 1].set_title('Notional Progress (%)')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Progress (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Daily Cost Increment
        axes[2, 0].bar([d - 0.2 for d in days_neural], neural['daily_results']['cost_increment'], 
                      width=0.4, label='Neural Strategy', alpha=0.7, color='blue')
        axes[2, 0].bar([d + 0.2 for d in days_linear], linear['daily_results']['cost_increment'], 
                      width=0.4, label='Linear Strategy', alpha=0.7, color='red')
        axes[2, 0].set_title('Daily Cost Increment')
        axes[2, 0].set_xlabel('Day')
        axes[2, 0].set_ylabel('Cost Increment (€)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Strategy Differences
        if len(days_neural) == len(days_linear):
            share_diff = [n - l for n, l in zip(neural['daily_results']['shares_to_buy'], 
                                               linear['daily_results']['shares_to_buy'])]
            axes[2, 1].bar(days_neural, share_diff, width=0.6, 
                          color=['green' if x > 0 else 'red' for x in share_diff], alpha=0.7)
            axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2, 1].set_title('Daily Shares Difference (Neural - Linear)')
            axes[2, 1].set_xlabel('Day')
            axes[2, 1].set_ylabel('Share Difference')
            axes[2, 1].grid(True, alpha=0.3)
        
        # 7. Daily PnL Comparison (Exercise Period Only)
        # Only show PnL for days >= early_exercise_start
        exercise_days_neural = [d for d in days_neural if d >= self.early_exercise_start]
        exercise_days_linear = [d for d in days_linear if d >= self.early_exercise_start]
        
        if exercise_days_neural and exercise_days_linear:
            exercise_pnl_neural = [neural['daily_results']['daily_pnl'][i] 
                                 for i, d in enumerate(days_neural) if d >= self.early_exercise_start]
            exercise_pnl_linear = [linear['daily_results']['daily_pnl'][i] 
                                 for i, d in enumerate(days_linear) if d >= self.early_exercise_start]
            
            axes[3, 0].plot(exercise_days_neural, exercise_pnl_neural, 
                           'b-', linewidth=2, label='Neural Strategy', marker='o', markersize=3)
            axes[3, 0].plot(exercise_days_linear, exercise_pnl_linear, 
                           'r--', linewidth=2, label='Linear Strategy', marker='s', markersize=3)
            axes[3, 0].set_title('Daily PnL (Exercise Period Only)')
            axes[3, 0].set_xlabel('Day')
            axes[3, 0].set_ylabel('PnL if Exercised Today (€)')
            axes[3, 0].legend()
            axes[3, 0].grid(True, alpha=0.3)
        else:
            axes[3, 0].text(0.5, 0.5, 'No Exercise Period\nData Available', 
                           ha='center', va='center', transform=axes[3, 0].transAxes, fontsize=12)
            axes[3, 0].set_title('Daily PnL (Exercise Period)')
        
        # 8. Stopping Probability (Neural Strategy Only)
        if neural['daily_results']['stopping_prob']:
            # Filter for days >= early_exercise_start where stopping prob > 0
            stopping_days = []
            stopping_probs = []
            for i, day in enumerate(days_neural):
                if day >= self.early_exercise_start and neural['daily_results']['stopping_prob'][i] > 0:
                    stopping_days.append(day)
                    stopping_probs.append(neural['daily_results']['stopping_prob'][i] * 100)
            
            if stopping_days:
                axes[3, 1].plot(stopping_days, stopping_probs, 'purple', linewidth=2, 
                               marker='o', markersize=4, label='Stopping Probability')
                axes[3, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, 
                                  label='Exercise Threshold (50%)')
                axes[3, 1].set_title('Neural Strategy - Stopping Probability')
                axes[3, 1].set_xlabel('Day')
                axes[3, 1].set_ylabel('Stopping Probability (%)')
                axes[3, 1].set_ylim(0, 100)
                axes[3, 1].legend()
                axes[3, 1].grid(True, alpha=0.3)
                
                # Highlight early exercise day if it exists
                if neural['exercise_day'] is not None:
                    axes[3, 1].axvline(x=neural['exercise_day'], color='red', 
                                      linestyle=':', linewidth=3, alpha=0.8, 
                                      label=f'Exercised (Day {neural["exercise_day"]})')
                    axes[3, 1].legend()
            else:
                axes[3, 1].text(0.5, 0.5, 'No Stopping Decisions\n(Before Early Exercise Period)', 
                               ha='center', va='center', transform=axes[3, 1].transAxes, 
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[3, 1].set_title('Neural Strategy - Stopping Probability')
        else:
            axes[3, 1].text(0.5, 0.5, 'Linear Strategy\n(No Early Exercise)', 
                           ha='center', va='center', transform=axes[3, 1].transAxes, 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            axes[3, 1].set_title('Stopping Probability')
        
        # Add early exercise indicator if applicable
        if neural['exercise_day'] is not None:
            for ax in axes.flat:
                ax.axvline(x=neural['exercise_day'], color='purple', linestyle='--', 
                          alpha=0.7, label=f'Early Exercise (Day {neural["exercise_day"]})')
        
        plt.tight_layout()
        
        # Add summary text
        summary_text = f"""
        Strategy Comparison Summary:
        
        Neural Strategy PnL: €{neural['final_pnl']:,.0f}
        Linear Strategy PnL: €{linear['final_pnl']:,.0f}
        PnL Difference: €{neural['final_pnl'] - linear['final_pnl']:,.0f}
        
        Neural Total Cost: €{neural['total_cost']:,.0f}
        Linear Total Cost: €{linear['total_cost']:,.0f}
        
        Early Exercise: {'Yes' if neural['exercise_day'] is not None else 'No'}
        """
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Strategy comparison plot saved to {save_path}")
        
        plt.show()
        
        return {
            'neural_pnl': neural['final_pnl'],
            'linear_pnl': linear['final_pnl'],
            'pnl_difference': neural['final_pnl'] - linear['final_pnl'],
            'neural_exercise_day': neural['exercise_day']
        }
    
    def plot_detailed_analysis(self, comparison_results, save_path: str = None):
        """Plot detailed analysis focusing on PnL and stopping decisions."""
        
        neural = comparison_results['neural_results']
        linear = comparison_results['linear_results']
        
        # Create specialized figure for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Detailed Strategy Analysis - Path {comparison_results["path_idx"]}', fontsize=16)
        
        days_neural = neural['daily_results']['day']
        days_linear = linear['daily_results']['day']
        
        # 1. PnL Evolution
        axes[0, 0].plot(days_neural, neural['daily_results']['cumulative_pnl'], 
                       'b-', linewidth=2, label='Neural (Cumulative)')
        axes[0, 0].plot(days_linear, linear['daily_results']['cumulative_pnl'], 
                       'r--', linewidth=2, label='Linear (Cumulative)')
        axes[0, 0].set_title('Cumulative PnL Evolution')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Cumulative PnL (€)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Daily Exercise Value (Exercise Period Only)
        # Only show exercise values for days >= early_exercise_start
        exercise_days_neural = [d for d in days_neural if d >= self.early_exercise_start]
        exercise_days_linear = [d for d in days_linear if d >= self.early_exercise_start]
        
        if exercise_days_neural and exercise_days_linear:
            exercise_pnl_neural = [neural['daily_results']['daily_pnl'][i] 
                                 for i, d in enumerate(days_neural) if d >= self.early_exercise_start]
            exercise_pnl_linear = [linear['daily_results']['daily_pnl'][i] 
                                 for i, d in enumerate(days_linear) if d >= self.early_exercise_start]
            
            axes[0, 1].plot(exercise_days_neural, exercise_pnl_neural, 
                           'b-', linewidth=2, label='Neural Strategy', alpha=0.8, marker='o', markersize=4)
            axes[0, 1].plot(exercise_days_linear, exercise_pnl_linear, 
                           'r--', linewidth=2, label='Linear Strategy', alpha=0.8, marker='s', markersize=4)
            axes[0, 1].set_title('Exercise Value (Exercise Period Only)')
            axes[0, 1].set_xlabel('Day')
            axes[0, 1].set_ylabel('PnL if Exercised Today (€)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Exercise Period\nData Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Exercise Value (Exercise Period)')
        
        # 3. Stopping Probability Heat Map
        stopping_probs = neural['daily_results']['stopping_prob']
        if any(p > 0 for p in stopping_probs):
            # Create stopping probability visualization
            exercise_days = [d for d in days_neural if d >= self.early_exercise_start]
            exercise_probs = [stopping_probs[i] for i, d in enumerate(days_neural) if d >= self.early_exercise_start]
            
            if exercise_days:
                colors = ['red' if p > 0.5 else 'orange' if p > 0.2 else 'green' for p in exercise_probs]
                bars = axes[1, 0].bar(exercise_days, [p * 100 for p in exercise_probs], 
                                     color=colors, alpha=0.7)
                axes[1, 0].axhline(y=50, color='black', linestyle='--', 
                                  label='Exercise Threshold (50%)')
                axes[1, 0].set_title('Stopping Probability by Day')
                axes[1, 0].set_xlabel('Day')
                axes[1, 0].set_ylabel('Stopping Probability (%)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add color legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', alpha=0.7, label='High (>50%)'),
                                 Patch(facecolor='orange', alpha=0.7, label='Medium (20-50%)'),
                                 Patch(facecolor='green', alpha=0.7, label='Low (<20%)')]
                axes[1, 0].legend(handles=legend_elements, loc='upper right')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Early Exercise\nDecisions Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Stopping Probability')
        
        # 4. Trading Rate Adjustments (moved to a different subplot)
        neural_adjustments = neural['daily_results']['trading_rate_output']
        
        # Show this as text summary instead of plot since we ran out of subplot space
        adjustment_stats = f"""
        Trading Rate Adjustment Stats:
        
        Mean Adjustment: {np.mean(neural_adjustments):.4f}
        Std Adjustment: {np.std(neural_adjustments):.4f}
        Min Adjustment: {min(neural_adjustments):.4f}
        Max Adjustment: {max(neural_adjustments):.4f}
        
        Range: [{min(neural_adjustments):.3f}, {max(neural_adjustments):.3f}]
        """
        
        # 5. Summary Statistics
        # Display key metrics instead of removed plots
        summary_text = f"""
        Strategy Comparison Summary:
        
        Neural Strategy Final PnL: €{neural['final_pnl']:,.0f}
        Linear Strategy Final PnL: €{linear['final_pnl']:,.0f}
        PnL Difference: €{neural['final_pnl'] - linear['final_pnl']:,.0f}
        
        Early Exercise: {'Day ' + str(neural['exercise_day']) if neural['exercise_day'] is not None else 'No'}
        
        Neural Total Cost: €{neural['total_cost']:,.0f}
        Linear Total Cost: €{linear['total_cost']:,.0f}
        
        Max Stopping Prob: {max(neural['daily_results']['stopping_prob']) * 100:.1f}%
        Avg Trading Adjustment: {np.mean(neural['daily_results']['trading_rate_output']):.3f}
        
        {adjustment_stats}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            detailed_save_path = save_path.replace('.png', '_detailed.png')
            plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed analysis plot saved to {detailed_save_path}")
        
        plt.show()
    
    def analyze_multiple_paths(self, num_paths: int = 5):
        """Analyze strategy differences across multiple paths."""
        
        print(f"\nAnalyzing {num_paths} paths...")
        print("="*50)
        
        results = []
        
        for i in range(num_paths):
            print(f"\nPath {i+1}/{num_paths}")
            comparison = self.simulate_single_path_strategies(path_idx=i)
            plot_results = self.plot_strategy_comparison(
                comparison, 
                save_path=f"plots/strategy_comparison_path_{i}.png"
            )
            
            results.append({
                'path_idx': i,
                'neural_pnl': plot_results['neural_pnl'],
                'linear_pnl': plot_results['linear_pnl'],
                'pnl_difference': plot_results['pnl_difference'],
                'exercise_day': plot_results['neural_exercise_day']
            })
        
        # Summary analysis
        self._plot_summary_analysis(results)
        
        return results
    
    def _plot_summary_analysis(self, results):
        """Plot summary analysis across multiple paths."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Path Strategy Analysis Summary', fontsize=16)
        
        paths = [r['path_idx'] for r in results]
        neural_pnls = [r['neural_pnl'] for r in results]
        linear_pnls = [r['linear_pnl'] for r in results]
        pnl_diffs = [r['pnl_difference'] for r in results]
        exercise_days = [r['exercise_day'] if r['exercise_day'] is not None else -1 for r in results]
        
        # PnL Comparison
        width = 0.35
        x = np.arange(len(paths))
        axes[0, 0].bar(x - width/2, neural_pnls, width, label='Neural Strategy', alpha=0.7)
        axes[0, 0].bar(x + width/2, linear_pnls, width, label='Linear Strategy', alpha=0.7)
        axes[0, 0].set_title('PnL Comparison by Path')
        axes[0, 0].set_xlabel('Path Index')
        axes[0, 0].set_ylabel('PnL (€)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PnL Differences Distribution
        axes[0, 1].hist(pnl_diffs, bins=min(10, len(pnl_diffs)), alpha=0.7, color='purple', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        axes[0, 1].axvline(x=np.mean(pnl_diffs), color='green', linestyle='-', linewidth=2, 
                          label=f'Mean: €{np.mean(pnl_diffs):,.0f}')
        axes[0, 1].set_title('PnL Difference Distribution')
        axes[0, 1].set_xlabel('PnL Difference (€)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Early Exercise Analysis
        exercise_paths = [i for i, day in enumerate(exercise_days) if day >= 0]
        exercise_day_values = [day for day in exercise_days if day >= 0]
        
        if exercise_day_values:
            axes[1, 0].bar(exercise_paths, exercise_day_values, alpha=0.7, color='purple')
            axes[1, 0].set_title('Early Exercise Days')
            axes[1, 0].set_xlabel('Path Index')
            axes[1, 0].set_ylabel('Exercise Day')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Early Exercises', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Early Exercise Days')
        
        # Statistics (Remove win rate analysis due to stochastic stopping)
        avg_improvement = np.mean(pnl_diffs)
        exercise_rate = len(exercise_day_values) / len(results) * 100
        
        stats_text = f"""
        Summary Statistics:
        
        Avg PnL Difference: €{avg_improvement:,.0f}
        Best Performance: €{max(pnl_diffs):,.0f}
        Worst Performance: €{min(pnl_diffs):,.0f}
        Std Dev: €{np.std(pnl_diffs):,.0f}
        
        Early Exercise Rate: {exercise_rate:.1f}%
        
        Note: No win rate calculated
        due to stochastic stopping
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/multi_path_summary.png', dpi=300, bbox_inches='tight')
        print("Multi-path summary saved to plots/multi_path_summary.png")
        plt.show()


def main():
    """Main function to run strategy comparison."""
    
    print("ASR Strategy Comparison Tool")
    print("="*50)
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Initialize comparator
    comparator = StrategyComparator(
        F=900_000_000.0,
        N=63,
        S0=45.0,
        num_paths=1000,
        early_exercise_start=22,
        gamma=2.5e-7
    )
    
    # Try to load trained model
    comparator.load_trained_model("models/best_asr_model.pth")
    
    # Analyze single path in detail
    print("\n1. Single Path Analysis:")
    comparison_results = comparator.simulate_single_path_strategies(path_idx=0)
    plot_summary = comparator.plot_strategy_comparison(
        comparison_results, 
        save_path="plots/single_path_comparison.png"
    )
    
    # Detailed analysis with PnL and stopping decisions
    print("\n1.5. Detailed Analysis:")
    comparator.plot_detailed_analysis(
        comparison_results,
        save_path="plots/single_path_comparison.png"
    )
    
    # Analyze multiple paths
    print("\n2. Multiple Path Analysis:")
    multi_path_results = comparator.analyze_multiple_paths(num_paths=5)
    
    print("\nStrategy comparison completed!")


if __name__ == "__main__":
    main()
