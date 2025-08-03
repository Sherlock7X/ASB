import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

from asr_simulator_vectorized import (
    ASRSimulator, 
    SimulationConfig, 
    BenchmarkStrategy,
    ASRPricingModel
)


class ASRVisualizer:
    """
    Visualization tools for ASR strategy comparison.
    Plots various metrics for the same market path under different strategies.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulator = ASRSimulator(config)
        
    def simulate_single_path_strategies(self, path_index: int = 0) -> Dict[str, Any]:
        """
        Simulate multiple strategies on a single market path for detailed comparison.
        
        Args:
            path_index: Index of the path to analyze (0-based)
            
        Returns:
            Dictionary containing results for each strategy
        """
        print(f"Analyzing path {path_index} with different strategies...")
        
        # Generate market paths
        stock_paths = self.simulator.stock_process.simulate_path(
            self.config.N, self.config.num_paths, 
            random_seed=self.config.random_seed
        )
        volume_paths = self.simulator.stock_process.simulate_volume(
            self.config.N, self.config.num_paths, 
            V0=self.config.V0
        )
        
        # Extract single path
        single_stock_path = stock_paths[path_index, :]
        single_volume_path = volume_paths[path_index, :]
        
        results = {}
        
        # 1. Benchmark Strategy
        benchmark_model = BenchmarkStrategy(
            F=self.config.F, N=self.config.N, S0=self.config.S0,
            early_exercise_start=self.config.early_exercise_start
        )
        benchmark_results = self._simulate_single_path(
            benchmark_model, single_stock_path, single_volume_path
        )
        results['benchmark'] = benchmark_results
        
        # 2. Neural Network Strategy
        try:
            model = ASRPricingModel(
                F=self.config.F, N=self.config.N, S0=self.config.S0, 
                early_exercise_start=self.config.early_exercise_start
            )
            model_file = Path("models/best_asr_model.pth")
            if model_file.exists():
                model.load_state_dict(torch.load(model_file))
                print("Neural network model loaded successfully.")
                
                nn_results = self._simulate_single_path(
                    model, single_stock_path, single_volume_path
                )
                results['neural_network'] = nn_results
            else:
                print("No trained model found. Skipping neural network analysis.")
                
        except Exception as e:
            print(f"Error loading neural network model: {e}")
        
        return results
    
    def _simulate_single_path(self, model, stock_path: np.ndarray, volume_path: np.ndarray) -> Dict[str, Any]:
        """
        Simulate a single path with given model and return detailed results.
        """
        N = self.config.N
        F = self.config.F
        dt = self.config.dt
        
        # Initialize state variables
        A = stock_path[0]  # Running average
        q = 0.0           # Shares purchased
        X = 0.0           # Cash spent
        
        # Storage for time series
        time_series = {
            'time': list(range(N + 1)),
            'stock_price': stock_path.tolist(),
            'running_average': [A],
            'shares_purchased': [q],
            'cash_spent': [X],
            'trading_rate': [0.0],
            'stopping_prob': [0.0],
            'pnl': [0.0]
        }
        
        for n in range(1, N + 1):
            # Update running average
            A = (A * n + stock_path[n]) / (n + 1)
            
            # Prepare inputs for model
            n_tensor = torch.tensor([float(n)], dtype=torch.float32)
            S_tensor = torch.tensor([stock_path[n]], dtype=torch.float32)
            A_tensor = torch.tensor([A], dtype=torch.float32)
            X_tensor = torch.tensor([X], dtype=torch.float32)
            q_tensor = torch.tensor([q], dtype=torch.float32)
            
            with torch.no_grad():
                v = model.compute_trading_rate(n_tensor, S_tensor, A_tensor, X_tensor, q_tensor).item()
                stopping_prob = model.compute_stopping_probability(n_tensor, S_tensor, A_tensor, X_tensor, q_tensor).item()
            
            # Enforce early exercise window
            if not (self.config.early_exercise_start <= n <= self.config.early_exercise_end):
                stopping_prob = 0.0
            
            # Update shares and cash
            q += v * dt
            execution_cost = self.config.eta * (v * volume_path[n]) ** self.config.phi
            X += v * stock_path[n] * dt + execution_cost * dt
            
            # Compute current PnL
            remaining_shares = F / A - q
            remaining_cost = remaining_shares * stock_path[n]
            terminal_penalty = self.config.penalty_coeff * (remaining_shares ** 2)
            pnl = F - X - remaining_cost - terminal_penalty
            
            # Store results
            time_series['running_average'].append(A)
            time_series['shares_purchased'].append(q)
            time_series['cash_spent'].append(X)
            time_series['trading_rate'].append(v)
            time_series['stopping_prob'].append(stopping_prob)
            time_series['pnl'].append(pnl)
        
        return time_series
    
    def plot_strategy_comparison(self, results: Dict[str, Any], save_path: str = "strategy_comparison.png"):
        """
        Create comprehensive comparison plots for different strategies.
        """
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('ASR Strategy Comparison - Single Path Analysis', fontsize=16, fontweight='bold')
        
        strategies = list(results.keys())
        colors = ['blue', 'red', 'green', 'orange']
        
        # 1. Stock Price and Running Average
        ax1 = axes[0, 0]
        for i, strategy in enumerate(strategies):
            data = results[strategy]
            ax1.plot(data['time'], data['stock_price'], '--', alpha=0.7, color='black', 
                    label='Stock Price' if i == 0 else "")
            ax1.plot(data['time'], data['running_average'], '-', linewidth=2, 
                    color=colors[i], label=f'{strategy.title()} - Running Avg')
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Price (€)')
        ax1.set_title('Stock Price vs Running Average')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trading Rate Comparison
        ax2 = axes[0, 1]
        for i, strategy in enumerate(strategies):
            data = results[strategy]
            ax2.plot(data['time'][1:], data['trading_rate'][1:], '-', linewidth=2, 
                    color=colors[i], label=f'{strategy.title()}')
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Trading Rate (shares/day)')
        ax2.set_title('Trading Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Shares Purchased
        ax3 = axes[1, 0]
        for i, strategy in enumerate(strategies):
            data = results[strategy]
            ax3.plot(data['time'], data['shares_purchased'], '-', linewidth=2, 
                    color=colors[i], label=f'{strategy.title()}')
        
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Cumulative Shares')
        ax3.set_title('Cumulative Shares Purchased')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cash Spent
        ax4 = axes[1, 1]
        for i, strategy in enumerate(strategies):
            data = results[strategy]
            cash_millions = [x / 1e6 for x in data['cash_spent']]
            ax4.plot(data['time'], cash_millions, '-', linewidth=2, 
                    color=colors[i], label=f'{strategy.title()}')
        
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Cash Spent (€M)')
        ax4.set_title('Cumulative Cash Spent')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. PnL Evolution
        ax5 = axes[2, 0]
        for i, strategy in enumerate(strategies):
            data = results[strategy]
            pnl_millions = [x / 1e6 for x in data['pnl']]
            ax5.plot(data['time'], pnl_millions, '-', linewidth=2, 
                    color=colors[i], label=f'{strategy.title()}')
        
        ax5.set_xlabel('Time (days)')
        ax5.set_ylabel('PnL (€M)')
        ax5.set_title('Profit & Loss Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Stopping Probability (only for neural network)
        ax6 = axes[2, 1]
        if 'neural_network' in results:
            data = results['neural_network']
            ax6.plot(data['time'], data['stopping_prob'], '-', linewidth=2, 
                    color='red', label='Neural Network')
            ax6.axvspan(self.config.early_exercise_start, self.config.early_exercise_end, 
                       alpha=0.2, color='green', label='Early Exercise Window')
        
        ax6.set_xlabel('Time (days)')
        ax6.set_ylabel('Stopping Probability')
        ax6.set_title('Early Exercise Probability')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Strategy comparison plot saved to: {save_path}")
    
    def plot_multiple_paths_comparison(self, num_paths: int = 5, save_path: str = "multiple_paths_comparison.png"):
        """
        Compare strategies across multiple random paths.
        """
        plt.figure(figsize=(15, 10))
        
        # Generate multiple paths
        stock_paths = self.simulator.stock_process.simulate_path(
            self.config.N, num_paths, random_seed=self.config.random_seed
        )
        volume_paths = self.simulator.stock_process.simulate_volume(
            self.config.N, num_paths, V0=self.config.V0
        )
        
        # Load models
        benchmark_model = BenchmarkStrategy(
            F=self.config.F, N=self.config.N, S0=self.config.S0,
            early_exercise_start=self.config.early_exercise_start
        )
        
        neural_model = None
        try:
            neural_model = ASRPricingModel(
                F=self.config.F, N=self.config.N, S0=self.config.S0, 
                early_exercise_start=self.config.early_exercise_start
            )
            model_file = Path("models/best_asr_model.pth")
            if model_file.exists():
                checkpoint = torch.load(model_file, weights_only=False)
                neural_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Could not load neural network model: {e}")
        
        # Simulate each path
        benchmark_pnls = []
        neural_pnls = []
        
        for i in range(num_paths):
            # Benchmark
            benchmark_results = self._simulate_single_path(
                benchmark_model, stock_paths[i, :], volume_paths[i, :]
            )
            benchmark_pnls.append([x / 1e6 for x in benchmark_results['pnl']])
            
            # Neural Network
            if neural_model is not None:
                neural_results = self._simulate_single_path(
                    neural_model, stock_paths[i, :], volume_paths[i, :]
                )
                neural_pnls.append([x / 1e6 for x in neural_results['pnl']])
        
        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Stock prices
        for i in range(num_paths):
            ax1.plot(range(self.config.N + 1), stock_paths[i, :], 
                    alpha=0.7, linewidth=1, label=f'Path {i+1}')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Stock Price (€)')
        ax1.set_title('Stock Price Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Benchmark PnL
        for i, pnl in enumerate(benchmark_pnls):
            ax2.plot(range(self.config.N + 1), pnl, 
                    alpha=0.7, linewidth=2, label=f'Path {i+1}')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('PnL (€M)')
        ax2.set_title('Benchmark Strategy PnL')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Neural Network PnL
        if neural_pnls:
            for i, pnl in enumerate(neural_pnls):
                ax3.plot(range(self.config.N + 1), pnl, 
                        alpha=0.7, linewidth=2, label=f'Path {i+1}')
            ax3.set_xlabel('Time (days)')
            ax3.set_ylabel('PnL (€M)')
            ax3.set_title('Neural Network Strategy PnL')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Neural Network\nModel Not Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Multiple paths comparison plot saved to: {save_path}")
    
    def create_summary_statistics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create summary statistics table for strategy comparison.
        """
        summary_data = []
        
        for strategy, data in results.items():
            final_pnl = data['pnl'][-1] / 1e6  # Convert to millions
            final_shares = data['shares_purchased'][-1]
            final_cash = data['cash_spent'][-1] / 1e6
            max_trading_rate = max(data['trading_rate'][1:]) if len(data['trading_rate']) > 1 else 0
            avg_trading_rate = np.mean(data['trading_rate'][1:]) if len(data['trading_rate']) > 1 else 0
            
            summary_data.append({
                'Strategy': strategy.title(),
                'Final PnL (€M)': f"{final_pnl:.2f}",
                'Total Shares': f"{final_shares:,.0f}",
                'Total Cash (€M)': f"{final_cash:.2f}",
                'Max Trading Rate': f"{max_trading_rate:,.0f}",
                'Avg Trading Rate': f"{avg_trading_rate:,.0f}"
            })
        
        df = pd.DataFrame(summary_data)
        return df


def main():
    """
    Main function to run visualization analysis.
    """
    print("ASR Strategy Visualization Analysis")
    print("=" * 50)
    
    # Configuration for visualization
    config = SimulationConfig(
        S0=45.0, sigma=0.6, V0=4_000_000,
        F=900_000_000, N=63, gamma=2.5e-7,
        early_exercise_start=22, early_exercise_end=63,
        eta=2e-7, phi=0.5, penalty_coeff=2e-7,
        num_paths=10, random_seed=2024,  # Use testing seed
        load_model=False
    )
    
    visualizer = ASRVisualizer(config)
    
    # 1. Single path detailed analysis
    print("1. Analyzing single path with different strategies...")
    path_index = 0  # Analyze first path
    single_path_results = visualizer.simulate_single_path_strategies(path_index)
    
    # Create detailed comparison plot
    visualizer.plot_strategy_comparison(
        single_path_results, 
        save_path="plots/single_path_strategy_comparison.png"
    )
    
    # Print summary statistics
    print("\nSummary Statistics for Single Path:")
    summary_df = visualizer.create_summary_statistics(single_path_results)
    print(summary_df.to_string(index=False))
    print()
    
    # 2. Multiple paths comparison
    print("2. Analyzing multiple paths...")
    visualizer.plot_multiple_paths_comparison(
        num_paths=5, 
        save_path="plots/multiple_paths_comparison.png"
    )
    
    print("\nVisualization analysis complete!")
    print("Check the 'plots/' directory for generated charts.")


if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)
    main()
