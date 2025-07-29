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
        
    def simulate_monte_carlo(self, model: ASRPricingModel) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation for ASR pricing using vectorized operations.
        
        Args:
            model: ASR pricing model (neural networks)
            
        Returns:
            Dictionary with simulation results including CARA utility approximation
        """
        num_paths = self.config.num_paths
        N = self.config.N
        F = self.config.F
        S0 = self.config.S0
        dt = self.config.dt

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
        
        # Initialize state variables for all paths
        A = np.full(num_paths, stock_paths[:, 0])
        q = np.zeros(num_paths)
        X = np.zeros(num_paths)
        
        stopped_mask = np.zeros(num_paths, dtype=bool)
        final_pnl = np.zeros(num_paths)

        for n in range(1, N + 1):
            active_mask = ~stopped_mask
            if not np.any(active_mask):
                break

            # Update running average for active paths
            A[active_mask] = update_running_average(A[active_mask], stock_paths[active_mask, n], n)
            
            # Prepare inputs for neural networks for active paths
            active_S = stock_paths[active_mask, n]
            active_A = A[active_mask]
            active_q = q[active_mask]
            active_X = X[active_mask]
            
            n_tensor = torch.full((active_mask.sum(),), float(n), dtype=torch.float32)
            S_tensor = torch.tensor(active_S, dtype=torch.float32)
            A_tensor = torch.tensor(active_A, dtype=torch.float32)
            X_tensor = torch.tensor(active_X, dtype=torch.float32)
            q_tensor = torch.tensor(active_q, dtype=torch.float32)

            with torch.no_grad():
                v = model.compute_trading_rate(n_tensor, S_tensor, A_tensor, X_tensor, q_tensor).numpy().flatten()
                stopping_prob = model.compute_stopping_probability(n_tensor, S_tensor, A_tensor, X_tensor, q_tensor).numpy().flatten()

            if not (self.config.early_exercise_start <= n <= self.config.early_exercise_end):
                stopping_prob[:] = 0.0
            
            all_stopping_probs[active_mask, n] = stopping_prob

            # Update shares purchased and cash spent for active paths
            q[active_mask] = update_shares_purchased(q[active_mask], v, dt)
            X[active_mask] = update_cash_spent(
                X[active_mask], v, stock_paths[active_mask, n], volume_paths[active_mask, n], dt,
                eta=self.config.eta, phi=self.config.phi
            )

            # Compute PnL if stopping at this time for all paths
            remaining_shares = F / A - q
            remaining_cost = remaining_shares * stock_paths[:, n]
            terminal_penalty = self.config.penalty_coeff * (remaining_shares ** 2)
            pnl_n = F - X - remaining_cost - terminal_penalty
            all_pnl_paths[:, n] = pnl_n

        # Compute CARA utility approximation
        cara_utility, expected_pnl = compute_cara_utility_approximation(
            all_pnl_paths, all_stopping_probs, 
            self.config.gamma
        )
        
        return {
            'cara_utility': cara_utility,
            'pnl_paths': all_pnl_paths,
            'stopping_probs': all_stopping_probs, 
            'stock_paths': stock_paths,
            'volume_paths': volume_paths,
            'expected_pnl': expected_pnl
        }


def compute_cara_utility_approximation(pnl_paths: np.ndarray,
                                     stopping_probs: np.ndarray,
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
        raise ValueError("Total weight is zero, check stopping probabilities and PnL paths.")
    
    # CARA utility approximation: E[PnL] - γ/2 * Var[PnL]
    cara_approx = expected_pnl - 0.5 * gamma * variance_pnl
    
    return cara_approx, expected_pnl


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
        # Handle vectorized operations for multiple paths
        n_vals = n  # Keep as tensor for vectorized operations
        A_vals = A  # Keep as tensor for vectorized operations
        q_vals = q  # Keep as tensor for vectorized operations
        
        # Linear schedule: buy F/N shares each period, adjusted for current average price
        target_shares = self.F / A_vals * (n_vals + 1) / self.N
        remaining_shares = target_shares - q_vals
        
        # Trading rate (shares per day, dt=1)
        v = remaining_shares  # Buy remaining shares linearly
        
        return v
    
    def compute_stopping_probability(self, n: torch.Tensor, S: torch.Tensor, A: torch.Tensor,
                                   X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Never exercise early - always go to maturity"""
        # Handle vectorized operations for multiple paths
        n_vals = n  # Keep as tensor for vectorized operations
        
        # Never exercise early, only at maturity (N=63)
        # Return 1.0 for all paths at maturity, 0.0 otherwise
        stopping_probs = torch.where(n_vals >= self.N, 
                                   torch.ones_like(n_vals, dtype=torch.float32),
                                   torch.zeros_like(n_vals, dtype=torch.float32))
        return stopping_probs


def train_neural_network(config: SimulationConfig) -> ASRPricingModel:
    """
    Train the neural network to maximize CARA utility using policy gradient approach.
    
    Since the Monte Carlo simulation returns numpy arrays, we use a policy gradient
    approach with finite differences for gradient estimation.
    
    Args:
        config: Simulation configuration parameters
        
    Returns:
        Trained ASR pricing model
    """
    print("Training Neural Network for ASR Pricing")
    print("=" * 50)
    
    # Initialize model and optimizer
    model = ASRPricingModel(
        F=config.F, 
        N=config.N, 
        S0=config.S0, 
        early_exercise_start=config.early_exercise_start
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    simulator = ASRSimulator(config)
    
    best_utility = float('-inf')
    best_model_state = None
    utilities_history = []
    
    print(f"Training Configuration:")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Monte Carlo Paths: {config.num_paths}")
    print(f"  - Hidden Dimension: {config.hidden_dim}")
    print()
    
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        model.train()
        
        # Evaluate current policy
        with torch.no_grad():
            results = simulator.simulate_monte_carlo(model)
            current_utility = results['cara_utility']
            expected_pnl = results['expected_pnl']
        
        # Store the utility as our "reward"
        utilities_history.append(current_utility)
        
        # Save best model
        if current_utility > best_utility:
            best_utility = current_utility
            best_model_state = model.state_dict().copy()
        
        # Simple parameter update using random perturbations (evolutionary strategy)
        if epoch > 0:  # Skip first iteration
            # Generate random perturbations
            perturbations = []
            utilities_perturbed = []
            
            # Try a few random perturbations
            num_perturbations = 3
            perturbation_std = 0.01
            
            for _ in range(num_perturbations):
                # Save current parameters
                original_params = [p.clone() for p in model.parameters()]
                
                # Apply random perturbation
                perturbation = []
                for param in model.parameters():
                    noise = torch.randn_like(param) * perturbation_std
                    param.data += noise
                    perturbation.append(noise)
                
                # Evaluate perturbed policy
                with torch.no_grad():
                    perturbed_results = simulator.simulate_monte_carlo(model)
                    perturbed_utility = perturbed_results['cara_utility']
                
                perturbations.append(perturbation)
                utilities_perturbed.append(perturbed_utility)
                
                # Restore original parameters
                for param, orig in zip(model.parameters(), original_params):
                    param.data = orig
            
            # Update parameters in direction of best perturbation
            if utilities_perturbed:
                best_idx = np.argmax(utilities_perturbed)
                if utilities_perturbed[best_idx] > current_utility:
                    # Move in direction of best perturbation
                    with torch.no_grad():
                        for param, noise in zip(model.parameters(), perturbations[best_idx]):
                            param.data += config.learning_rate * noise
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: CARA Utility = {current_utility:10.2f}, Expected PnL = €{expected_pnl:,.0f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    
    print(f"\nTraining Complete!")
    print(f"Best CARA Utility: {best_utility:.2f}")
    print(f"Final CARA Utility: {utilities_history[-1]:.2f}")
    print(f"Improvement: {utilities_history[-1] - utilities_history[0]:.2f}")
    
    return model


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
        early_exercise_start=22, early_exercise_end=63,
        eta=2e-7, phi=0.5, penalty_coeff=2e-7,
        num_paths=1000, num_epochs=200, random_seed=42  # Increased paths and epochs for better training
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
    print()

    # 2. Neural Network Strategy (Trained)
    print("2. Testing Trained Neural Network Strategy")
    print("-" * 50) 
    
    # Train the neural network model
    print("Training neural network...")
    model = train_neural_network(config)
    
    # Test the trained model
    NN_results = simulator.simulate_monte_carlo(model)
    results['neural_network'] = NN_results
    print(f"  CARA Utility: {NN_results['cara_utility']:.2f}")
    print(f"  Expected PnL: €{NN_results['expected_pnl']:,.0f}")
    print()
    
    # 3. Performance Comparison
    print("3. Performance Comparison")
    print("-" * 50)
    print(f"Strategy                    | CARA Utility | Expected PnL")
    print(f"--------------------------- | ------------ | -------------")
    print(f"Benchmark (Linear + Never)  | {benchmark_results['cara_utility']:11.2f} | €{benchmark_results['expected_pnl']:10,.0f}")
    print(f"Neural Network (Trained)    | {NN_results['cara_utility']:11.2f} | €{NN_results['expected_pnl']:10,.0f}")
    
    utility_improvement = NN_results['cara_utility'] - benchmark_results['cara_utility']
    pnl_improvement = NN_results['expected_pnl'] - benchmark_results['expected_pnl']
    
    print(f"--------------------------- | ------------ | -------------")
    print(f"Improvement (NN - Benchmark)| {utility_improvement:11.2f} | €{pnl_improvement:10,.0f}")
    print()
    
    return results


if __name__ == "__main__":
    run_benchmark_comparison()