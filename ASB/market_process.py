import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


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
                     random_seed: Optional[int] = None,
                     device: str = 'cpu') -> torch.Tensor:
        """
        Simulate stock price paths using Arithmetic Brownian Motion with tensors.
        
        S_{n+1} = S_n + σ√δt ε_{n+1}, where ε_n are i.i.d. N(0,1)
        
        Args:
            N: Number of time steps
            num_paths: Number of simulation paths
            random_seed: Random seed for reproducibility
            device: Device to run calculations on ('cpu' or 'cuda')
            
        Returns:
            Tensor of shape (num_paths, N+1) with stock prices
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        # Generate i.i.d. N(0,1) random variables
        epsilon = torch.randn(num_paths, N, device=device)
        
        # Initialize price tensor
        S = torch.zeros(num_paths, N + 1, device=device)
        S[:, 0] = self.S0
        
        # Simulate Arithmetic Brownian Motion
        drift_term = self.sigma * torch.sqrt(torch.tensor(self.dt, device=device))
        
        for i in range(N):
            S[:, i + 1] = S[:, i] + drift_term * epsilon[:, i]
            
        return S
    
    def simulate_volume(self, 
                       N: int, 
                       num_paths: int = 1,
                       V0: float = 4_000_000,
                       vol_volatility: float = 0.1,
                       device: str = 'cpu') -> torch.Tensor:
        """
        Simulate constant market volume process with tensors.
        
        Args:
            N: Number of time steps
            num_paths: Number of simulation paths  
            V0: Constant volume level (default 4M shares/day from realistic example)
            vol_volatility: Not used (kept for compatibility)
            device: Device to run calculations on ('cpu' or 'cuda')
            
        Returns:
            Tensor of shape (num_paths, N+1) with constant volumes
        """
        V = torch.zeros(num_paths, N + 1, device=device)
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

def update_shares_purchased(q_prev: float, v: float, dt: float) -> float:
    """
    Update cumulative shares purchased.
    
    q_{n+1} = q_n + v_n * dt
    """
    return q_prev + v * dt


def execution_cost_function(rho: Union[torch.Tensor, float, np.ndarray], 
                           eta: float = 0.1, 
                           phi: float = 0.5) -> Union[torch.Tensor, float]:
    """
    Execution cost function L(ρ) = η|ρ|^(1+φ)
    
    Supports tensors, scalars and numpy arrays.
    
    Args:
        rho: Participation rate (v/V) - can be tensor, scalar or array
        eta: Cost parameter (default 0.1 from realistic example)
        phi: Cost exponent (default 0.5 from realistic example)
    """
    if isinstance(rho, torch.Tensor):
        return eta * (torch.abs(rho) ** (1 + phi))
    else:
        import numpy as np
        return eta * (np.abs(rho) ** (1 + phi))


def update_cash_spent(X_prev: Union[torch.Tensor, float, np.ndarray], 
                     v: Union[torch.Tensor, float, np.ndarray], 
                     S_next: Union[torch.Tensor, float, np.ndarray], 
                     V_next: Union[torch.Tensor, float, np.ndarray], 
                     dt: float, 
                     eta: float = 0.1, 
                     phi: float = 0.5) -> Union[torch.Tensor, float, np.ndarray]:
    """
    Update cumulative cash spent with tensor support.
    
    X_{n+1} = X_n + v_n * S_{n+1} * dt + L(v_n/V_{n+1}) * V_{n+1} * dt
    
    Supports tensors, scalars and numpy arrays.
    
    Args:
        eta: Execution cost parameter (default 0.1 from realistic example)
        phi: Execution cost exponent (default 0.5 from realistic example)
    """
    if isinstance(X_prev, torch.Tensor):
        # Tensor operations
        # Handle division by zero with tensor operations
        rho = torch.where(V_next > 0, v / V_next, torch.zeros_like(v))  # Participation rate
        execution_cost = torch.where(V_next > 0, 
                                   execution_cost_function(rho, eta, phi) * V_next * dt,
                                   torch.zeros_like(V_next))
        
        return X_prev + v * S_next * dt + execution_cost
    else:
        # Numpy operations (backward compatibility)
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
    
    # Generate multiple paths for Monte Carlo (using tensors)
    num_paths = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    paths = process.simulate_path(N, num_paths=num_paths, random_seed=22, device=device)
    volumes = process.simulate_volume(N, num_paths=num_paths, V0=4_000_000, device=device)  # 4M shares/day
    
    print(f"ASR Realistic Example Test Case (Tensor Version)")
    print(f"=" * 50)
    print(f"Device: {device}")
    print(f"Stock Price Process: Arithmetic Brownian Motion")
    print(f"Formula: S_{{n+1}} = S_n + σ√δt ε_{{n+1}}")
    print(f"Parameters:")
    print(f"  - Initial price (S₀): €{process.S0:.2f}")
    print(f"  - Daily volatility (σ): {process.sigma:.1%}")
    print(f"  - Time step (δt): {process.dt} day")
    print(f"  - Trading days (N): {N}")
    print(f"  - Notional amount (F): €{F:,.0f}")
    print(f"  - Risk aversion (γ): {gamma:.1e}")
    print(f"  - Constant volume (V): {volumes[0, 0].item():,.0f} shares/day")
    print()
    
    # Show sample price paths
    print(f"Sample Price Paths (5 paths):")
    print(f"Initial prices: {['€{:.2f}'.format(p.item()) for p in paths[:, 0]]}")
    print(f"Final prices:   {['€{:.2f}'.format(p.item()) for p in paths[:, -1]]}")
    print(f"Price changes:  {['€{:.2f}'.format((paths[i, -1] - paths[i, 0]).item()) for i in range(num_paths)]}")
    print()
    
    # Test tensor execution cost function
    print("Testing tensor execution cost function:")
    test_v = torch.tensor([1000.0, 2000.0, 3000.0], device=device)
    test_V = torch.tensor([4000000.0, 4000000.0, 4000000.0], device=device)
    test_rho = test_v / test_V
    test_cost = execution_cost_function(test_rho, eta=0.1, phi=0.5)
    print(f"Trading rates: {test_v.tolist()}")
    print(f"Participation rates: {test_rho.tolist()}")
    print(f"Execution costs: {test_cost.tolist()}")
    print()
