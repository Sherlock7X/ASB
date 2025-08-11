#!/usr/bin/env python3
"""
Simple Strategy Comparison: Neural vs Linear on Single Path

This script compares the neural network strategy with the linear strategy
on a single price path, showing daily settlement numbers and key differences.
"""

import torch
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


class SimpleStrategyComparator:
    """Simple comparison of neural network strategy with linear strategy."""
    
    def __init__(self, 
                 F: float = 900_000_000.0,
                 N: int = 63,
                 S0: float = 45.0,
                 early_exercise_start: int = 22):
        """Initialize the strategy comparator."""
        
        self.F = F
        self.N = N
        self.S0 = S0
        self.early_exercise_start = early_exercise_start
        
        # Create both models with the same parameters
        self.neural_env = ASREnv(F=F, N=N, S0=S0, num_paths=1000, 
                                early_exercise_start=early_exercise_start,
                                eta=0.0, phi=0.5, terminal_penalty_coeff=0.0)
        
        self.linear_env = ASREnv_Linear(F=F, N=N, S0=S0, num_paths=1000, 
                                       early_exercise_start=early_exercise_start,
                                       eta=0.0, phi=0.5, terminal_penalty_coeff=0.0)
        
        print(f"Strategy Comparator initialized: F={F:,.0f}, N={N}, S0={S0}")
    
    def load_trained_model(self, model_path: str):
        """Load a trained neural network model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.neural_env.load_state_dict(checkpoint['asr_env_state_dict'])
            print(f"Loaded trained model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using random initialization.")
        except Exception as e:
            print(f"Warning: Error loading model: {e}. Using random initialization.")
    
    def simulate_strategies(self, path_idx: int = 0):
        """
        Simulate both strategies on a single path.
        
        Args:
            path_idx: Index of the path to analyze
            
        Returns:
            Dictionary with simulation results
        """
        # Get the same path for both strategies
        stock_batch, volume_batch, avg_price_batch = self.neural_env.get_training_batch(
            batch_size=max(path_idx + 1, 10), shuffle=False
        )
        
        # Use the specified path
        stock_path = stock_batch[path_idx]
        volume_path = volume_batch[path_idx]
        avg_price_path = avg_price_batch[path_idx]
        
        # Simulate neural network strategy
        neural_results = self._simulate_single_strategy(
            stock_path, volume_path, avg_price_path, 
            use_neural=True, strategy_name="Neural Network"
        )
        
        # Simulate linear strategy
        linear_results = self._simulate_single_strategy(
            stock_path, volume_path, avg_price_path, 
            use_neural=False, strategy_name="Linear"
        )
        
        return {
            'stock_path': stock_path.numpy(),
            'neural_results': neural_results,
            'linear_results': linear_results,
            'path_idx': path_idx
        }
    
    def _simulate_single_strategy(self, stock_path, volume_path, avg_price_path, 
                                 use_neural=True, strategy_name="Strategy"):
        """Simulate a single strategy on the given path."""
        
        device = stock_path.device
        total_cost = torch.tensor(0.0, device=device)
        shares_purchased = torch.tensor(0.0, device=device)
        
        # Storage for daily results
        daily_data = {
            'days': [],
            'stock_prices': [],
            'shares_bought': [],
            'cumulative_shares': [],
            'notional_progress': [],
            'costs': [],
            'stopping_probs': [],
            'pnl_values': []
        }
        
        exercise_day = None
        
        print(f"\n{strategy_name} Strategy:")
        print("-" * 30)
        
        for t in range(self.N):
            current_price = stock_path[t]
            current_volume = volume_path[t]
            current_avg_price = avg_price_path[t]
            
            # Calculate target shares
            if use_neural:
                trading_inputs, stopping_inputs = self.neural_env.compute_inputs(
                    current_price.unsqueeze(0), t, 
                    shares_purchased.unsqueeze(0), current_avg_price.unsqueeze(0)
                )
                
                trading_output = self.neural_env.trading_network(trading_inputs)
                target_shares = (self.F / current_avg_price) * ((t + 1) / self.N) * (1.0 + trading_output.squeeze())
                
                # Check early exercise
                if t >= self.early_exercise_start:
                    stopping_output = self.neural_env.stopping_network(stopping_inputs)
                    stopping_prob = self.neural_env.modified_sigmoid(
                        self.neural_env.nu_phi * stopping_output
                    ).squeeze()
                    
                    if stopping_prob.item() > 0.7:
                        exercise_day = t
                        print(f"  Early exercise at day {t}")
                        break
                else:
                    stopping_prob = torch.tensor(0.0)
            else:
                # Linear strategy
                target_shares = (self.F / current_avg_price) * ((t + 1) / self.N)
                stopping_prob = torch.tensor(0.0)
            
            # Calculate shares to buy
            shares_to_buy = target_shares - shares_purchased
            
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
            
            total_cost = new_total_cost.squeeze()
            shares_purchased += shares_to_buy
            
            # Calculate PnL
            if t >= self.early_exercise_start:
                shares_remaining = self.F / current_avg_price - shares_purchased
                terminal_penalty = 0.0  # Cancelled termination cost
                current_pnl = self.F - shares_remaining * current_price - total_cost - terminal_penalty
            else:
                current_pnl = self.F - total_cost
            
            # Calculate notional progress
            notional_progress = (shares_purchased * current_avg_price / self.F) * 100
            
            # Store results
            daily_data['days'].append(t)
            daily_data['stock_prices'].append(current_price.item())
            daily_data['shares_bought'].append(shares_to_buy.item())
            daily_data['cumulative_shares'].append(shares_purchased.item())
            daily_data['notional_progress'].append(notional_progress.item())
            daily_data['costs'].append(total_cost.item())
            daily_data['stopping_probs'].append(stopping_prob.item())
            daily_data['pnl_values'].append(current_pnl.item())
            
            # Print every 10 days
            if t % 10 == 0:
                print(f"  Day {t:2d}: Price={current_price:.2f}, Shares={shares_to_buy:.0f}, "
                      f"Total Cost={total_cost:.0f}")
        
        final_pnl = daily_data['pnl_values'][-1]
        
        # Ensure final_pnl is a float for consistent handling
        if hasattr(final_pnl, 'item'):
            final_pnl_value = final_pnl.item()
        else:
            final_pnl_value = float(final_pnl)
            
        print(f"  Final PnL: €{final_pnl_value:.0f}")
        
        return {
            'daily_data': daily_data,
            'final_pnl': final_pnl_value,
            'exercise_day': exercise_day,
            'total_cost': total_cost.item()
        }
    
    def plot_comparison(self, results, save_path: str = None):
        """Plot comparison of the two strategies."""
        
        neural = results['neural_results']
        linear = results['linear_results']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Strategy Comparison - Path {results["path_idx"]}', fontsize=16)
        
        # 1. Stock Price Path
        axes[0, 0].plot(results['stock_path'], 'b-', linewidth=2, label='Stock Price')
        axes[0, 0].set_title('Stock Price Path')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Price (€)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark early exercise if it happened
        if neural['exercise_day'] is not None:
            axes[0, 0].axvline(x=neural['exercise_day'], color='red', linestyle='--', 
                              alpha=0.7, label=f'Early Exercise (Day {neural["exercise_day"]})')
            axes[0, 0].legend()
        
        # 2. Daily Shares Purchased
        neural_days = neural['daily_data']['days']
        linear_days = linear['daily_data']['days']
        
        axes[0, 1].bar([d - 0.2 for d in neural_days], neural['daily_data']['shares_bought'], 
                      width=0.4, label='Neural', alpha=0.7, color='blue')
        axes[0, 1].bar([d + 0.2 for d in linear_days], linear['daily_data']['shares_bought'], 
                      width=0.4, label='Linear', alpha=0.7, color='red')
        axes[0, 1].set_title('Daily Shares Purchased')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Shares')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Notional Progress (%)
        axes[0, 2].plot(neural_days, neural['daily_data']['notional_progress'], 
                       'b-', linewidth=2, label='Neural')
        axes[0, 2].plot(linear_days, linear['daily_data']['notional_progress'], 
                       'r--', linewidth=2, label='Linear')
        axes[0, 2].axhline(y=100, color='black', linestyle=':', alpha=0.5, label='Target (100%)')
        axes[0, 2].set_title('Notional Progress (%)')
        axes[0, 2].set_xlabel('Day')
        axes[0, 2].set_ylabel('Progress (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Total Cost Evolution
        axes[1, 0].plot(neural_days, neural['daily_data']['costs'], 
                       'b-', linewidth=2, label='Neural')
        axes[1, 0].plot(linear_days, linear['daily_data']['costs'], 
                       'r--', linewidth=2, label='Linear')
        axes[1, 0].set_title('Total Cost Evolution')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Total Cost (€)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. PnL Evolution (Exercise Period Only)
        exercise_days_neural = [d for d in neural_days if d >= self.early_exercise_start]
        exercise_days_linear = [d for d in linear_days if d >= self.early_exercise_start]
        
        if exercise_days_neural and exercise_days_linear:
            exercise_pnl_neural = [neural['daily_data']['pnl_values'][i] 
                                 for i, d in enumerate(neural_days) if d >= self.early_exercise_start]
            exercise_pnl_linear = [linear['daily_data']['pnl_values'][i] 
                                 for i, d in enumerate(linear_days) if d >= self.early_exercise_start]
            
            axes[1, 1].plot(exercise_days_neural, exercise_pnl_neural, 
                           'b-', linewidth=2, label='Neural', marker='o', markersize=3)
            axes[1, 1].plot(exercise_days_linear, exercise_pnl_linear, 
                           'r--', linewidth=2, label='Linear', marker='s', markersize=3)
            axes[1, 1].set_title('PnL Evolution (Exercise Period)')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('PnL (€)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Exercise Period\nData Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('PnL Evolution (Exercise Period)')
        
        # 6. Stopping Probability (Neural Strategy Only)
        # Filter for days >= early_exercise_start where stopping prob > 0
        exercise_days_neural = [d for d in neural_days if d >= self.early_exercise_start]
        if exercise_days_neural:
            exercise_stopping_probs = [neural['daily_data']['stopping_probs'][i] * 100 
                                     for i, d in enumerate(neural_days) if d >= self.early_exercise_start]
            
            if any(p > 0 for p in exercise_stopping_probs):
                colors = ['red' if p > 50 else 'orange' if p > 20 else 'green' for p in exercise_stopping_probs]
                axes[1, 2].bar(exercise_days_neural, exercise_stopping_probs, 
                              color=colors, alpha=0.7, width=0.8)
                axes[1, 2].set_title('Stopping Probability (Exercise Period)')
                axes[1, 2].set_xlabel('Day')
                axes[1, 2].set_ylabel('Stopping Probability (%)')
                axes[1, 2].set_ylim(0, 100)
                axes[1, 2].grid(True, alpha=0.3)
                
                # Mark early exercise day if it exists
                if neural['exercise_day'] is not None:
                    axes[1, 2].axvline(x=neural['exercise_day'], color='red', 
                                      linestyle=':', linewidth=3, alpha=0.8, 
                                      label=f'Exercised (Day {neural["exercise_day"]})')
                    axes[1, 2].legend()
            else:
                axes[1, 2].text(0.5, 0.5, 'No Stopping Decisions\nMade in Exercise Period', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, 
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[1, 2].set_title('Stopping Probability')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Exercise Period\nData Available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes, 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            axes[1, 2].set_title('Stopping Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Strategy comparison plot saved to {save_path}")
        
        # plt.show()
        
        return {
            'neural_pnl': neural['final_pnl'],
            'linear_pnl': linear['final_pnl'],
            'pnl_difference': neural['final_pnl'] - linear['final_pnl']
        }


def main():
    """Main function to run strategy comparison."""
    
    print("Simple ASR Strategy Comparison")
    print("=" * 50)
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Initialize comparator
    comparator = SimpleStrategyComparator(
        F=900_000_000.0,
        N=63,
        S0=45.0,
        early_exercise_start=2
    )
    
    # Try to load trained model
    comparator.load_trained_model("models/best_asr_model.pth")
    
    # Analyze single path
    print("\nSingle Path Analysis:")
    results = comparator.simulate_strategies(path_idx=3)
    
    # Plot comparison
    plot_results = comparator.plot_comparison(
        results, 
        save_path="plots/simple_strategy_comparison.png"
    )
    
    print(f"\nComparison completed!")
    print(f"PnL Difference (Neural - Linear): €{plot_results['pnl_difference']:,.0f}")


if __name__ == "__main__":
    main()
