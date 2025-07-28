import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm

from market_process import (
    StockPriceProcess, 
    MarketState,
    update_running_average,
    update_shares_purchased, 
    update_cash_spent,
    execution_cost_function
)
from neural_networks import ASRPricingModel, ModifiedSigmoid


@dataclass
class SimulationConfig:
    """
    Configuration parameters for ASR simulation.
    
    Based on realistic example from README:
    - Risk Aversion: γ = 2.5e-7 (realistic for institutional investors)
    - Notional Amount: €900M (large-scale repurchase program)
    - Time Horizon: 63 trading days (~3 months)
    - Early Exercise Window: Days 22-62 (flexible execution period)
    """
    # Market parameters
    S0: float = 45.0                    # Initial stock price (€)
    sigma: float = 0.6                  # Daily volatility (60% annualized ~21%)
    V0: float = 4_000_000              # Constant volume (4M shares/day)
    dt: float = 1.0                    # Time step (1 day)
    
    # Contract parameters  
    F: float = 900_000_000             # Notional amount (€900M)
    N: int = 63                        # Time horizon (63 trading days)
    gamma: float = 2.5e-7              # Risk aversion parameter
    early_exercise_start: int = 22     # Early exercise window starts day 22
    early_exercise_end: int = 62       # Early exercise window ends day 62
    
    # Execution cost parameters
    eta: float = 2e-7                  # Cost coefficient
    phi: float = 0.5                   # Nonlinearity parameter
    penalty_coeff: float = 2e-7        # Terminal penalty coefficient C
    
    # Simulation parameters
    num_paths: int = 10000              # Monte Carlo paths
    random_seed: Optional[int] = 42    # Random seed for reproducibility
    
    # Neural network parameters
    hidden_dim: int = 50               # Hidden layer size
    learning_rate: float = 0.01        # Higher learning rate
    num_epochs: int = 30               # More epochs for better convergence


class ASRSimulator:
    """
    Monte Carlo simulation engine for ASR pricing with CARA utility approximation.
    
    This class handles the forward simulation of ASR contracts, computing profit/loss
    for each path and time step, and evaluating the CARA utility approximation to
    avoid numerical overflow issues.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize ASR simulator with configuration.
        
        Args:
            config: Simulation configuration parameters
        """
        self.config = config
        self.stock_process = StockPriceProcess(
            S0=config.S0,
            sigma=config.sigma, 
            dt=config.dt
        )
        self.modified_sigmoid = ModifiedSigmoid()
        
    def simulate_single_path(self, 
                           model: ASRPricingModel,
                           stock_path: np.ndarray,
                           volume_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Simulate a single Monte Carlo path for ASR contract.
        
        Args:
            model: ASR pricing model (neural networks)
            stock_path: Stock price path of shape (N+1,)
            volume_path: Volume path of shape (N+1,)
            
        Returns:
            Tuple of (pnl_path, stopping_probs, stopping_time)
            - pnl_path: Profit/loss at each time step
            - stopping_probs: Stopping probabilities at each time step
            - stopping_time: Actual stopping time (-1 if never stopped)
        """
        N = self.config.N
        F = self.config.F
        S0 = self.config.S0
        dt = self.config.dt
        
        # Initialize state variables
        pnl_path = np.zeros(N + 1)
        stopping_probs = np.zeros(N + 1)
        
        # Track running state
        A = stock_path[0]  # Running average starts at S0
        q = 0.0           # Shares purchased so far
        X = 0.0           # Cash spent so far
        stopped = False
        stopping_time = -1
        
        for n in range(1, N + 1):
            if stopped:
                # If already stopped, carry forward the final PnL
                pnl_path[n] = pnl_path[stopping_time]
                stopping_probs[n] = 0.0
                continue
                
            # Update running average
            A = update_running_average(A, stock_path[n], n)
            
            # Prepare inputs for neural networks
            time_input = n / N - 0.5
            price_input = stock_path[n] / S0 - 1.0
            spread_input = (A - stock_path[n]) / S0
            
            # Trading rate network inputs (4D)
            position_input = q * A / F - 0.5
            trading_inputs = torch.tensor([
                time_input, price_input, spread_input, position_input
            ], dtype=torch.float32).unsqueeze(0)
            
            # Stopping policy network inputs (3D)  
            stopping_inputs = torch.tensor([
                time_input, price_input, spread_input
            ], dtype=torch.float32).unsqueeze(0)
            
            # Get neural network outputs
            with torch.no_grad():
                # Convert to tensors for model input
                n_tensor = torch.tensor([float(n)], dtype=torch.float32)
                S_tensor = torch.tensor([stock_path[n]], dtype=torch.float32)
                A_tensor = torch.tensor([A], dtype=torch.float32)
                X_tensor = torch.tensor([X], dtype=torch.float32)
                q_tensor = torch.tensor([q], dtype=torch.float32)
                
                # Get trading rate from model
                v = model.compute_trading_rate(n_tensor, S_tensor, A_tensor, X_tensor, q_tensor).item()
                
                # Get stopping probability from model
                stopping_prob = model.compute_stopping_probability(
                    n_tensor, S_tensor, A_tensor, X_tensor, q_tensor
                ).item()
                
                # Clamp stopping probability for early exercise window
                if not (self.config.early_exercise_start <= n <= self.config.early_exercise_end):
                    stopping_prob = 0.0
                    
            stopping_probs[n] = stopping_prob
            
            # Update shares purchased
            q = update_shares_purchased(q, v, dt)
            
            # Update cash spent (includes execution costs)
            X = update_cash_spent(
                X, v, stock_path[n], volume_path[n], dt,
                eta=self.config.eta, phi=self.config.phi
            )
            
            # Compute PnL if stopping at this time
            # PnL_n^F = F - X_n - (F/A_n - q_n) S_n - ℓ(F/A_n - q_n)
            remaining_shares = F / A - q
            remaining_cost = remaining_shares * stock_path[n]
            terminal_penalty = self.config.penalty_coeff * (remaining_shares ** 2)
            
            pnl_n = F - X - remaining_cost - terminal_penalty
            pnl_path[n] = pnl_n
            
            # Check if we stop at this time step (stochastic stopping)
            if stopping_prob > 0 and np.random.random() < stopping_prob:
                stopped = True
                stopping_time = n
                
        # If never stopped early, must exercise at maturity
        if stopping_time == -1:
            stopping_time = N
            
        return pnl_path, stopping_probs, stopping_time
        
    def simulate_monte_carlo(self, model: ASRPricingModel) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation for ASR pricing.
        
        Args:
            model: ASR pricing model (neural networks)
            
        Returns:
            Dictionary with simulation results including CARA utility approximation
        """
        num_paths = self.config.num_paths
        N = self.config.N
        
        # Generate all stock and volume paths
        stock_paths = self.stock_process.simulate_path(
            N, num_paths, random_seed=self.config.random_seed
        )
        volume_paths = self.stock_process.simulate_volume(
            N, num_paths, V0=self.config.V0
        )
        
        # Storage for results
        all_pnl_paths = np.zeros((num_paths, N + 1))
        all_stopping_probs = np.zeros((num_paths, N + 1)) 
        all_stopping_times = np.zeros(num_paths, dtype=int)
        
        # Simulate each path
        for i in tqdm(range(num_paths), desc="Simulating ASR paths"):
            pnl_path, stopping_probs, stopping_time = self.simulate_single_path(
                model, stock_paths[i], volume_paths[i]
            )
            all_pnl_paths[i] = pnl_path
            all_stopping_probs[i] = stopping_probs
            all_stopping_times[i] = stopping_time
            
        # Compute CARA utility approximation
        cara_utility = compute_cara_utility_approximation(
            all_pnl_paths, all_stopping_probs, all_stopping_times, 
            self.config.gamma
        )
        
        return {
            'cara_utility': cara_utility,
            'pnl_paths': all_pnl_paths,
            'stopping_probs': all_stopping_probs, 
            'stopping_times': all_stopping_times,
            'stock_paths': stock_paths,
            'volume_paths': volume_paths,
            'expected_pnl': np.mean([all_pnl_paths[i, all_stopping_times[i]] 
                                   for i in range(num_paths)]),
            'pnl_std': np.std([all_pnl_paths[i, all_stopping_times[i]] 
                             for i in range(num_paths)])
        }


def compute_cara_utility_approximation(pnl_paths: np.ndarray,
                                     stopping_probs: np.ndarray,
                                     stopping_times: np.ndarray,
                                     gamma: float) -> float:
    """
    Compute CARA utility approximation using second-order Taylor expansion.
    
    The approximation is:
    CARA_approx = E[PnL] - γ/2 * Var[PnL]
    
    In Monte Carlo framework:
    CARA_approx = 1/I ∑∑∏(1-p_k^i)p_n^i PnL_n^i 
                - γ/2[1/I ∑∑∏(1-p_k^i)p_n^i (PnL_n^i)² 
                - (1/I ∑∑∏(1-p_k^i)p_n^i PnL_n^i)²]
    
    Args:
        pnl_paths: PnL paths of shape (num_paths, N+1)
        stopping_probs: Stopping probabilities of shape (num_paths, N+1)
        stopping_times: Actual stopping times for each path
        gamma: Risk aversion parameter
        
    Returns:
        CARA utility approximation value
    """
    num_paths, N_plus_1 = pnl_paths.shape
    N = N_plus_1 - 1
    
    # Compute weighted expectations over all possible stopping times
    weighted_pnl_sum = 0.0
    weighted_pnl_squared_sum = 0.0
    total_weight = 0.0
    
    for i in range(num_paths):
        for n in range(1, N + 1):
            # Compute probability of stopping at time n for path i
            # P(stop at n) = (∏_{k=1}^{n-1} (1-p_k)) * p_n
            
            prob_not_stopped_before_n = 1.0
            for k in range(1, n):
                prob_not_stopped_before_n *= (1.0 - stopping_probs[i, k])
                
            prob_stop_at_n = prob_not_stopped_before_n * stopping_probs[i, n]
            
            # For paths that reach maturity without early exercise
            if n == N and stopping_times[i] == N:
                # Compute probability of reaching maturity
                prob_reach_maturity = 1.0
                for k in range(1, N):
                    prob_reach_maturity *= (1.0 - stopping_probs[i, k])
                prob_stop_at_n = prob_reach_maturity
            elif stopping_times[i] != n:
                # Skip time steps where we didn't actually stop
                continue
                
            if prob_stop_at_n > 1e-12:  # Avoid numerical issues
                weight = prob_stop_at_n
                pnl_value = pnl_paths[i, n]
                
                weighted_pnl_sum += weight * pnl_value
                weighted_pnl_squared_sum += weight * (pnl_value ** 2)
                total_weight += weight
    
    if total_weight > 1e-12:
        # Normalize by total weight
        expected_pnl = weighted_pnl_sum / total_weight
        expected_pnl_squared = weighted_pnl_squared_sum / total_weight
        variance_pnl = expected_pnl_squared - (expected_pnl ** 2)
    else:
        # Fallback to simple average if weighting fails
        final_pnls = [pnl_paths[i, stopping_times[i]] for i in range(num_paths)]
        expected_pnl = np.mean(final_pnls)
        variance_pnl = np.var(final_pnls)
    
    # CARA utility approximation: E[PnL] - γ/2 * Var[PnL]
    cara_approx = expected_pnl - 0.5 * gamma * variance_pnl
    
    return cara_approx


class BenchmarkStrategy:
    """Benchmark strategies for comparison with neural network approach."""
    
    def __init__(self, F: float, N: int, S0: float, early_exercise_start: int = 22):
        self.F = F
        self.N = N
        self.S0 = S0
        self.early_exercise_start = early_exercise_start
    
    def compute_trading_rate(self, n: torch.Tensor, S: torch.Tensor, A: torch.Tensor, 
                           X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Linear trading strategy: v = (F/A * (n+1)/N - q) / dt"""
        # Convert to numpy for computation
        n_val = n.item()
        A_val = A.item()
        q_val = q.item()
        
        # Linear schedule: buy F/N shares each period, adjusted for current average price
        target_shares = self.F / A_val * (n_val + 1) / self.N
        remaining_shares = target_shares - q_val
        
        # Trading rate (shares per day, dt=1)
        v = remaining_shares  # Buy remaining shares linearly
        
        return torch.tensor([v], dtype=torch.float32)
    
    def compute_stopping_probability(self, n: torch.Tensor, S: torch.Tensor, A: torch.Tensor,
                                   X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Never exercise early - always go to maturity"""
        n_val = n.item()
        
        # Never exercise early, only at maturity
        if n_val >= self.N - 1:
            return torch.tensor([1.0], dtype=torch.float32)
        else:
            return torch.tensor([0.0], dtype=torch.float32)


def run_benchmark_comparison():
    """
    Compare neural network strategy with benchmark strategies.
    """
    print("ASR Benchmark Comparison")
    print("=" * 60)
    
    # Initialize configuration
    config = SimulationConfig(
        S0=45.0, sigma=0.6, V0=4_000_000,
        F=900_000_000, N=63, gamma=2.5e-7,
        early_exercise_start=22, early_exercise_end=62,
        eta=2e-7, phi=0.5, penalty_coeff=2e-7,
        num_paths=500, num_epochs=10, random_seed=42  # Reduced for faster testing
    )
    
    print(f"Configuration:")
    print(f"  Notional: €{config.F:,.0f}, Time Horizon: {config.N} days")
    print(f"  Risk Aversion: γ = {config.gamma:.2e}")
    print(f"  Monte Carlo Paths: {config.num_paths:,}")
    print()
    
    simulator = ASRSimulator(config)
    results = {}
    
    # 1. Benchmark: Linear Strategy + Never Early Exercise
    print("1. Testing Benchmark: Linear Strategy + Never Early Exercise")
    print("-" * 50)
    
    benchmark_model = BenchmarkStrategy(
        F=config.F, N=config.N, S0=config.S0,
        early_exercise_start=config.early_exercise_start
    )
    
    benchmark_results = simulator.simulate_monte_carlo(benchmark_model)
    results['benchmark'] = benchmark_results
    
    print(f"  CARA Utility: {benchmark_results['cara_utility']:.2f}")
    print(f"  Expected PnL: €{benchmark_results['expected_pnl']:,.0f}")
    print(f"  PnL Std Dev: €{benchmark_results['pnl_std']:,.0f}")
    print(f"  Avg Stopping Time: {np.mean(benchmark_results['stopping_times']):.1f} days")
    print()


if __name__ == "__main__":
    run_benchmark_comparison()