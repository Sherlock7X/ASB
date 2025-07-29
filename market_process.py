import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class StockPriceProcess:
    """
    Arithmetic Brownian Motion for stock price evolution.
    
    S_{n+1} = S_n + σ√δt ε_{n+1}, where ε_n are i.i.d. N(0,1) random variables
    """
    
    def __init__(self, 
                 S0: float = 45.0, 
                 mu: float = 0.0, 
                 sigma: float = 0.6, 
                 dt: float = 1.0):
        """
        Initialize stock price process parameters.
        
        Args:
            S0: Initial stock price
            mu: Drift parameter (not used in arithmetic BM, kept for compatibility)
            sigma: Volatility parameter (σ in the formula) - default 2% daily volatility
            dt: Time step size (δt in the formula)
        """
        self.S0 = S0
        self.mu = mu  # Not used in arithmetic BM but kept for compatibility
        self.sigma = sigma
        self.dt = dt
        
    def simulate_path(self, 
                     N: int, 
                     num_paths: int = 1, 
                     random_seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate stock price paths using Arithmetic Brownian Motion.
        
        S_{n+1} = S_n + σ√δt ε_{n+1}, where ε_n are i.i.d. N(0,1)
        
        Args:
            N: Number of time steps
            num_paths: Number of simulation paths
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of shape (num_paths, N+1) with stock prices
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Generate i.i.d. N(0,1) random variables
        epsilon = np.random.normal(0, 1, (num_paths, N))
        
        # Initialize price array
        S = np.zeros((num_paths, N + 1))
        S[:, 0] = self.S0
        
        # Simulate Arithmetic Brownian Motion
        for i in range(N):
            S[:, i + 1] = S[:, i] + self.sigma * np.sqrt(self.dt) * epsilon[:, i]
            
        return S
    
    def simulate_volume(self, 
                       N: int, 
                       num_paths: int = 1,
                       V0: float = 4_000_000,
                       vol_volatility: float = 0.1) -> np.ndarray:
        """
        Simulate constant market volume process.
        
        Args:
            N: Number of time steps
            num_paths: Number of simulation paths  
            V0: Constant volume level (default 4M shares/day from realistic example)
            vol_volatility: Not used (kept for compatibility)
            
        Returns:
            Array of shape (num_paths, N+1) with constant volumes
        """
        V = np.zeros((num_paths, N + 1))
        V[:, :] = V0  # All volumes are constant at V0
        
        return V


class MarketState:
    """
    Container for market state variables at each time step.
    """
    
    def __init__(self, n: int, S: float, A: float, X: float, q: float, V: float):
        self.n = n  # Time step
        self.S = S  # Current stock price
        self.A = A  # Running average price
        self.X = X  # Cumulative cash spent
        self.q = q  # Shares purchased so far
        self.V = V  # Market volume
        
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        return torch.tensor([self.n, self.S, self.A, self.X, self.q], dtype=torch.float32)


def update_running_average(A_prev: float, S_new: float, n: int) -> float:
    """
    Update running average price.
    
    A_n = (1/n) * sum_{k=1}^n S_k = ((n-1)/n) * A_{n-1} + (1/n) * S_n
    """
    if n == 1:
        return S_new
    return ((n - 1) / n) * A_prev + (1 / n) * S_new


def update_shares_purchased(q_prev: float, v: float, dt: float) -> float:
    """
    Update cumulative shares purchased.
    
    q_{n+1} = q_n + v_n * dt
    """
    return q_prev + v * dt


def execution_cost_function(rho, eta: float = 2e-7, phi: float = 0.5):
    """
    Execution cost function L(ρ) = η|ρ|^(1+φ)
    
    Supports both scalar and vectorized operations.
    
    Args:
        rho: Participation rate (v/V) - can be scalar or array
        eta: Cost parameter (default 2×10⁻⁷ from realistic example)
        phi: Cost exponent (default 0.5 from realistic example)
    """
    import numpy as np
    return eta * (np.abs(rho) ** (1 + phi))


def update_cash_spent(X_prev, v, S_next, V_next, dt: float, 
                     eta: float = 2e-7, phi: float = 0.5):
    """
    Update cumulative cash spent.
    
    X_{n+1} = X_n + v_n * S_{n+1} * dt + L(v_n/V_{n+1}) * V_{n+1} * dt
    
    Supports both scalar and vectorized operations.
    
    Args:
        eta: Execution cost parameter (default 2×10⁻⁷ from realistic example)
        phi: Execution cost exponent (default 0.5 from realistic example)
    """
    import numpy as np
    
    # Convert to numpy arrays to support both scalar and vector operations
    X_prev = np.asarray(X_prev)
    v = np.asarray(v)
    S_next = np.asarray(S_next)
    V_next = np.asarray(V_next)
    
    # Handle division by zero with vectorized operations
    rho = np.where(V_next > 0, v / V_next, 0)  # Participation rate
    execution_cost = np.where(V_next > 0, 
                             execution_cost_function(rho, eta, phi) * V_next * dt,
                             0)
        
    return X_prev + v * S_next * dt + execution_cost


if __name__ == "__main__":
    # Test case matching the realistic example from README
    # Risk aversion: γ = 2.5e-7, Notional: €900M, Time horizon: 63 days
    
    # Initialize with realistic parameters
    process = StockPriceProcess(S0=45.0, mu=0.0, sigma=0.6, dt=1.0)  # 60% daily volatility

    # Simulation parameters from realistic example
    N = 63  # 63 trading days (~3 months)
    F = 900_000_000  # €900M notional amount
    gamma = 2.5e-7  # Risk aversion parameter
    
    # Generate multiple paths for Monte Carlo
    num_paths = 5
    paths = process.simulate_path(N, num_paths=num_paths, random_seed=22)
    volumes = process.simulate_volume(N, num_paths=num_paths, V0=4_000_000)  # 4M shares/day
    
    print(f"ASR Realistic Example Test Case")
    print(f"=" * 40)
    print(f"Stock Price Process: Arithmetic Brownian Motion")
    print(f"Formula: S_{{n+1}} = S_n + σ√δt ε_{{n+1}}")
    print(f"Parameters:")
    print(f"  - Initial price (S₀): €{process.S0:.2f}")
    print(f"  - Daily volatility (σ): {process.sigma:.1%}")
    print(f"  - Time step (δt): {process.dt} day")
    print(f"  - Trading days (N): {N}")
    print(f"  - Notional amount (F): €{F:,.0f}")
    print(f"  - Risk aversion (γ): {gamma:.1e}")
    print(f"  - Constant volume (V): {volumes[0, 0]:,.0f} shares/day")
    print()
    
    # Show sample price paths
    print(f"Sample Price Paths (5 paths):")
    print(f"Initial prices: {['€{:.2f}'.format(p) for p in paths[:, 0]]}")
    print(f"Final prices:   {['€{:.2f}'.format(p) for p in paths[:, -1]]}")
    print(f"Price changes:  {['€{:.2f}'.format(paths[i, -1] - paths[i, 0]) for i in range(num_paths)]}")
    print()
