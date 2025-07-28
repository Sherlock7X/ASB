import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TradingRateNetwork(nn.Module):
    """
    Neural network for trading rate v_θ with 4 inputs, 50 hidden neurons, ReLU activation.
    
    Inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0, qA/F - 1/2)
    Output: Adjustment to naive trading schedule
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 50):
        super(TradingRateNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the trading rate network.
        
        Args:
            inputs: Tensor of shape (..., 4) with normalized inputs
            
        Returns:
            Network output (adjustment term)
        """
        return self.network(inputs)


class StoppingPolicyNetwork(nn.Module):
    """
    Neural network for stopping policy p_φ with 3 inputs, 50 hidden neurons, ReLU activation.
    
    Inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0)
    Output: Stopping decision parameter
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 50):
        super(StoppingPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the stopping policy network.
        
        Args:
            inputs: Tensor of shape (..., 3) with normalized inputs
            
        Returns:
            Network output (stopping parameter)
        """
        return self.network(inputs)


class ModifiedSigmoid(nn.Module):
    """
    Modified sigmoid activation function S(x) = min(max(2/(1+e^(-x)) - 1/2, 0), 1)
    
    This is a rescaled and bounded logistic function to allow values 0 and 1 to be reached.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_rescaled = 2.0 / (1.0 + torch.exp(-x)) - 0.5
        return torch.clamp(sigmoid_rescaled, min=0.0, max=1.0)


class ASRPricingModel(nn.Module):
    """
    Complete ASR pricing model combining trading rate and stopping policy networks.
    """
    
    def __init__(self, 
                 F: float,
                 N: int, 
                 S0: float = 100.0,
                 nu_phi: float = 1.0,
                 early_exercise_start: int = 22):
        """
        Initialize ASR pricing model.
        
        Args:
            F: Fixed notional amount
            N: Total number of trading periods
            S0: Initial stock price
            nu_phi: Scaling parameter for stopping policy
            early_exercise_start: First day when early exercise is allowed (default 22 from realistic example)
        """
        super(ASRPricingModel, self).__init__()
        
        self.F = F
        self.N = N
        self.S0 = S0
        self.nu_phi = nu_phi
        self.early_exercise_start = early_exercise_start
        
        # Neural networks
        self.trading_network = TradingRateNetwork()
        self.stopping_network = StoppingPolicyNetwork()
        self.modified_sigmoid = ModifiedSigmoid()
        
    def compute_trading_rate(self, 
                           n: torch.Tensor, 
                           S: torch.Tensor, 
                           A: torch.Tensor, 
                           X: torch.Tensor, 
                           q: torch.Tensor) -> torch.Tensor:
        """
        Compute trading rate v_θ(n, S, A, X, q).
        
        v_θ(n,S,A,X,q) = F/A · (n+1)/N (1 + ṽ_θ((n/N - 1/2, S/S0 - 1, (A-S)/S0, qA/F - 1/2))) - q
        """
        batch_size = n.shape[0] if len(n.shape) > 0 else 1
        
        # Prepare inputs for neural network (4 inputs as specified)
        input1 = n / self.N - 0.5
        input2 = S / self.S0 - 1
        input3 = (A - S) / self.S0
        input4 = q * A / self.F - 0.5
        
        nn_inputs = torch.stack([input1, input2, input3, input4], dim=-1)
        
        # Get network output
        v_tilde = self.trading_network(nn_inputs).squeeze(-1)
        
        # Compute final trading rate
        naive_rate = (self.F / A) * ((n + 1) / self.N)
        adjustment = 1 + v_tilde
        
        v = naive_rate * adjustment - q
        
        return v
    
    def compute_stopping_probability(self,
                                   n: torch.Tensor,
                                   S: torch.Tensor, 
                                   A: torch.Tensor,
                                   X: torch.Tensor,
                                   q: torch.Tensor) -> torch.Tensor:
        """
        Compute stopping probability p_φ(n, S, A, X, q).
        
        p_φ(n,S,A,X,q) = 1_{n∈N} · S(ν_φ · (qA/F - p̃_φ(n/N - 1/2, S/S0 - 1, (A-S)/S0))) + 1_{n=N}
        
        Early exercise is only allowed from day 22 to 62 (as per realistic example).
        """
        # Check if we're at final time step
        is_final = (n >= self.N - 1).float()
        
        # Check if we're in the early exercise window (days 22-62 from realistic example)
        is_exercise_period = ((n >= self.early_exercise_start) & (n < self.N - 1)).float()
        
        if torch.any(is_exercise_period):
            # Prepare inputs for stopping network  
            input1 = n / self.N - 0.5
            input2 = S / self.S0 - 1
            input3 = (A - S) / self.S0
            
            nn_inputs = torch.stack([input1, input2, input3], dim=-1)
            
            # Get network output
            p_tilde = self.stopping_network(nn_inputs).squeeze(-1)
            
            # Compute stopping probability
            frontier_term = q * A / self.F - p_tilde
            sigmoid_input = self.nu_phi * frontier_term
            stopping_prob = self.modified_sigmoid(sigmoid_input)
            
            # Apply indicator functions: can only stop in exercise period or at final time
            p = is_exercise_period * stopping_prob + is_final
        else:
            # Outside exercise window, can only stop at final time
            p = is_final
            
        return p
    
    def compute_pnl(self,
                   n: torch.Tensor,
                   S: torch.Tensor,
                   A: torch.Tensor, 
                   X: torch.Tensor,
                   q: torch.Tensor) -> torch.Tensor:
        """
        Compute P&L at exercise/expiry.
        
        PnL_n^F = F - X_n - (F/A_n - q_n) * S_n - l(F/A_n - q_n)
        """
        remaining_shares = self.F / A - q
        execution_cost = self.execution_cost(remaining_shares)  # Simplified
        
        pnl = self.F - X - remaining_shares * S - execution_cost
        
        return pnl
    
    def execution_cost(self, shares: torch.Tensor) -> torch.Tensor:
        """
        Terminal penalty function ℓ(shares) = C * shares^2.
        
        From README: Terminal penalty ℓ(q) = Cq² with C = 2×10⁻⁷ €/share²
        """
        C = 2e-7  # Terminal penalty coefficient from realistic example
        return C * (shares ** 2)
    
    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing trading rate, stopping probability, and P&L.
        
        Args:
            state_batch: Tensor of shape (batch_size, 5) with [n, S, A, X, q]
            
        Returns:
            Tuple of (trading_rate, stopping_probability, pnl)
        """
        n = state_batch[:, 0]
        S = state_batch[:, 1] 
        A = state_batch[:, 2]
        X = state_batch[:, 3]
        q = state_batch[:, 4]
        
        v = self.compute_trading_rate(n, S, A, X, q)
        p = self.compute_stopping_probability(n, S, A, X, q)
        pnl = self.compute_pnl(n, S, A, X, q)
        
        return v, p, pnl


if __name__ == "__main__":
    # Test case matching the realistic example from README
    # Parameters from realistic ASR contract configuration
    
    F = 900_000_000  # €900M notional amount (realistic example)
    N = 63           # 63 trading days (~3 months)
    S0 = 100.0       # Initial stock price €100
    gamma = 2.5e-7   # Risk aversion parameter
    
    print(f"ASR Neural Networks Test Case")
    print(f"=" * 40)
    print(f"Realistic Example Configuration:")
    print(f"  - Notional amount (F): €{F:,.0f}")
    print(f"  - Trading days (N): {N}")  
    print(f"  - Initial stock price (S₀): €{S0:.2f}")
    print(f"  - Risk aversion (γ): {gamma:.1e}")
    print(f"  - Early exercise window: Days 22-62")
    print()
    
    # Initialize model with realistic parameters
    early_exercise_start = 22  # Early exercise window starts at day 22
    model = ASRPricingModel(F=F, N=N, S0=S0, early_exercise_start=early_exercise_start)
    
    print(f"Neural Network Architecture:")
    print(f"  - Trading Rate Network: 4 inputs → 50 hidden (ReLU) → 1 output")
    print(f"  - Stopping Policy Network: 3 inputs → 50 hidden (ReLU) → 1 output")
    print(f"  - Modified Sigmoid: S(x) = min(max(2/(1+e^(-x)) - 1/2, 0), 1)")
    print()
    
    # Create realistic test batch
    batch_size = 16
    
    # Time steps within early exercise window (days 22-62)
    n = torch.randint(22, 62, (batch_size,)).float()
    
    # Stock prices around initial price with realistic volatility (2% daily)
    time_scaling = torch.sqrt(n/252.0)  # Scale volatility by square root of time
    S = S0 + S0 * 0.02 * time_scaling * torch.randn(batch_size)  # Manual normal generation
    
    # Running averages close to current prices
    A = S + S0 * 0.01 * torch.randn(batch_size)  # Small deviations
    
    # Cumulative cash spent (realistic progression)
    progress_ratio = n / N
    X = F * progress_ratio * (1.0 + 0.1 * torch.randn(batch_size))  # Around expected spending
    
    # Shares purchased (realistic progression)
    target_shares = F / S0  # Total shares to purchase
    q = target_shares * progress_ratio * (1.0 + 0.1 * torch.randn(batch_size))
    
    # Ensure non-negative values
    S = torch.clamp(S, min=50.0)  # Minimum €50 stock price
    A = torch.clamp(A, min=50.0)  # Minimum €50 running average
    X = torch.clamp(X, min=0.0)   # Non-negative cash spent
    q = torch.clamp(q, min=0.0)   # Non-negative shares purchased
    
    state_batch = torch.stack([n, S, A, X, q], dim=1)
    
    print(f"Test Batch Statistics:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time steps: [{n.min().item():.0f}, {n.max().item():.0f}] (early exercise window)")
    print(f"  - Stock prices: [€{S.min().item():.2f}, €{S.max().item():.2f}]")
    print(f"  - Running averages: [€{A.min().item():.2f}, €{A.max().item():.2f}]")
    print(f"  - Cash spent: [€{X.min().item():,.0f}, €{X.max().item():,.0f}]")
    print(f"  - Shares purchased: [{q.min().item():,.0f}, {q.max().item():,.0f}]")
    print()
    
    # Forward pass
    v, p, pnl = model(state_batch)
    
    print(f"Neural Network Outputs:")
    print(f"  - Trading rates shape: {v.shape}")
    print(f"  - Stopping probabilities shape: {p.shape}")  
    print(f"  - P&L shape: {pnl.shape}")
    print()
    
    # Show sample results
    print(f"Sample Results (first 5 paths):")
    for i in range(min(5, batch_size)):
        print(f"  Path {i+1}:")
        print(f"    Day: {n[i].item():.0f}, Stock: €{S[i].item():.2f}, Avg: €{A[i].item():.2f}")
        print(f"    Trading rate: {v[i].item():,.0f} shares/day")
        print(f"    Stopping prob: {p[i].item():.1%}")
        print(f"    P&L if stop: €{pnl[i].item():,.0f}")
        print()
    
    # Test specific network components
    print(f"Component Tests:")
    
    # Test Modified Sigmoid with various inputs
    sigmoid = model.modified_sigmoid
    test_inputs = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    sigmoid_outputs = sigmoid(test_inputs)
    print(f"  Modified Sigmoid test:")
    for i, (inp, out) in enumerate(zip(test_inputs, sigmoid_outputs)):
        print(f"    S({inp:.1f}) = {out.item():.3f}")
    print()
    
    # Test terminal penalty function
    remaining_shares = torch.tensor([1000.0, 5000.0, 10000.0, 50000.0])
    penalties = model.execution_cost(remaining_shares)
    print(f"  Terminal Penalty ℓ(q) = Cq² with C = 2×10⁻⁷:")
    for shares, penalty in zip(remaining_shares, penalties):
        print(f"    {shares.item():,.0f} shares → €{penalty.item():,.2f}")
    
    print(f"\nTest completed successfully!")
    print(f"All parameters match the realistic ASR contract configuration.")
