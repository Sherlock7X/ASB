import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# Handle imports for both package and direct execution
try:
    from .market_process import StockPriceProcess, update_cash_spent
except ImportError:
    # If relative import fails (when running directly), try absolute import
    from market_process import StockPriceProcess, update_cash_spent


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


class ASREnv(nn.Module):
    """
    Complete ASR pricing model combining trading rate and stopping policy networks.
    """
    
    def __init__(self, 
                 F: float,
                 N: int, 
                 S0: float = 45.0,  # Use realistic example default
                 nu_phi: float = 2.0,
                 early_exercise_start: int = 22,
                 num_paths: int = 10_000,
                 # Market process parameters
                 sigma: float = 0.2,
                 dt: float = 1/252,  # Daily time step (1/252 of a year)
                 V0: float = 4_000_000,  # Constant volume from realistic example
                 eta: float = 0.0,  # Execution cost parameter
                 phi: float = 0.5,  # Execution cost exponent
                 gamma: float = 2.5e-7,  # Risk aversion parameter
                 terminal_penalty_coeff: float = 0.0,  # Terminal penalty parameter
                 ):
        """
        Initialize ASR pricing model.
        
        Args:
            F: Fixed notional amount
            N: Total number of trading periods
            S0: Initial stock price (default from realistic example)
            nu_phi: Scaling parameter for stopping policy
            early_exercise_start: First day when early exercise is allowed
            num_paths: Number of paths to simulate
            sigma: Annual volatility for Geometric Brownian Motion
            dt: Time step size (1/252 for daily steps, 252 trading days per year)
            V0: Constant market volume
            eta: Execution cost parameter
            phi: Execution cost exponent
            gamma: Risk aversion parameter
            terminal_penalty_coeff: Terminal penalty parameter
        """
        super(ASREnv, self).__init__()

        self.F = F
        self.N = N
        self.S0 = S0
        self.nu_phi = nu_phi
        self.early_exercise_start = early_exercise_start
        self.num_paths = num_paths
        self.V0 = V0
        self.eta = eta
        self.phi = phi
        self.gamma = gamma
        # Terminal penalty parameter C = 2·10⁻⁷ €·share⁻²
        self.terminal_penalty_coeff = terminal_penalty_coeff
        
        # Initialize market process
        self.market_process = StockPriceProcess(S0=S0, sigma=sigma, dt=dt)
        
        # Storage for simulated paths
        self.stock_paths = None
        self.volume_paths = None
        self.paths_generated = False
        
        # Neural networks
        self.trading_network = TradingRateNetwork()
        self.stopping_network = StoppingPolicyNetwork()
        self.modified_sigmoid = ModifiedSigmoid()
        
        self._initialize_networks()
        self._initialize_paths()
        
    def _initialize_networks(self):
        """Initialize networks with small weights to start close to benchmark behavior."""
        for module in [self.trading_network, self.stopping_network]:
            for param in module.parameters():
                if param.dim() > 1:  # Weight matrices
                    nn.init.normal_(param, mean=0.0, std=0.01)  # Very small weights
                else:  # Bias vectors
                    nn.init.constant_(param, 0.0)

    def _initialize_paths(self):
        """Initialize stock price and volume paths using the market process with tensors."""
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        # Generate stock price paths using the market process (now returns tensors)
        self.stock_paths = self.market_process.simulate_path(
            N=self.N, 
            num_paths=self.num_paths, 
            random_seed=42,
            device=device
        )
        
        # Generate volume paths using the market process (now returns tensors)
        self.volume_paths = self.market_process.simulate_volume(
            N=self.N,
            num_paths=self.num_paths,
            V0=self.V0,
            device=device
        )

        self.avg_price = self._update_avg_price()
        
        self.paths_generated = True

    def _update_avg_price(self):
        """
        Calculate cumulative Volume-Weighted Average Price (VWAP) for each path at each time step.
        
        For each time step t, calculate: VWAP_t = Σ(Price_i * Volume_i) / Σ(Volume_i) for i = 0 to t
        
        Returns:
            Tensor of shape (num_paths, N+1) with cumulative VWAP for each path at each time step
        """
        if self.stock_paths is None or self.volume_paths is None:
            return None
            
        # Calculate cumulative VWAP for each path at each time step
        # stock_paths and volume_paths are (num_paths, N+1)
        batch_size, time_steps = self.stock_paths.shape
        
        # Initialize cumulative VWAP tensor
        cumulative_vwap = torch.zeros_like(self.stock_paths)
        
        # For each time step, calculate cumulative VWAP up to that point
        for t in range(time_steps):
            if t == 0:
                # At t=0, VWAP is just the initial price
                cumulative_vwap[:, t] = self.stock_paths[:, t]
            else:
                # Calculate cumulative price*volume and volume up to time t
                price_volume_cumsum = torch.sum(
                    self.stock_paths[:, :t+1] * self.volume_paths[:, :t+1], 
                    dim=1
                )
                volume_cumsum = torch.sum(self.volume_paths[:, :t+1], dim=1)
                
                # Calculate VWAP up to time t
                cumulative_vwap[:, t] = price_volume_cumsum / (volume_cumsum + 1e-8)
        
        return cumulative_vwap

    def get_training_batch(self, batch_size: int, shuffle: bool = True):
        """
        Get a batch of paths for training.
        
        Args:
            batch_size: Number of paths in the batch
            shuffle: Whether to shuffle the paths
            
        Returns:
            Batch of stock paths and volume paths
        """
        if not self.paths_generated:
            raise ValueError("Paths not generated. Call _initialize_paths() first.")
            
        if shuffle:
            indices = torch.randperm(self.num_paths)[:batch_size]
        else:
            indices = torch.arange(min(batch_size, self.num_paths))
            
        return self.stock_paths[indices], self.volume_paths[indices], self.avg_price[indices]
    
    def compute_inputs(self, stock_batch, time_step, shares_purchased, avg_price):
        """
        Compute normalized inputs for the neural networks.
        
        Args:
            stock_batch: Current stock prices (batch_size,)
            time_step: Current time step n
            shares_purchased: Number of shares already bought (q in the formula)
            avg_price: Average price so far (batch_size,)
            
        Returns:
            trading_inputs: Inputs for trading network (batch_size, 4)
            stopping_inputs: Inputs for stopping network (batch_size, 3)
        """
        batch_size = stock_batch.shape[0]
        
        # Normalized time: n/N - 1/2
        time_norm = time_step / self.N - 0.5
        
        # Normalized stock price: S/S0 - 1
        price_norm = stock_batch / self.S0 - 1.0
        
        # Price difference normalized: (A-S)/S0
        price_diff_norm = (avg_price - stock_batch) / self.S0
        
        # Purchased shares normalized: qA/F - 1/2 
        # where q = shares_purchased, A = avg_price, F = total notional
        purchased_notional = shares_purchased * avg_price
        shares_norm = purchased_notional / self.F - 0.5
        
        # Trading network inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0, qA/F - 1/2)
        trading_inputs = torch.stack([
            torch.full((batch_size,), time_norm),
            price_norm,
            price_diff_norm,
            shares_norm
        ], dim=1)
        
        # Stopping network inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0)
        stopping_inputs = torch.stack([
            torch.full((batch_size,), time_norm),
            price_norm,
            price_diff_norm
        ], dim=1)
        
        return trading_inputs, stopping_inputs

    def eval_strategy(self, batch_size: int = 1000):
        """
        Evaluate trading strategy using single loop with exercise PnL matrix.
        
        The loss function implements the expectation in the given formula:
        -1/γ log E[exp(-γ PnL)] ≈ 1/I ∑∑∏(1-p_k^i)p_n^i PnL_n^i - γ/2[...] - (...)^2
        
        Args:
            batch_size: Size of batch to evaluate
            
        Returns:
            loss: Negative expected utility (loss to minimize)
        """
        stock_batch, volume_batch, avg_price_batch = self.get_training_batch(batch_size)
        device = stock_batch.device
        
        # Initialize tracking variables
        total_cost = torch.zeros(batch_size, device=device)
        shares_purchased = torch.zeros(batch_size, device=device)
        
        # Matrix to store PnL for each possible exercise time
        # Shape: (batch_size, N) - PnL if exercised at time t
        exercise_pnl_matrix = torch.zeros(batch_size, self.N, device=device)
        
        # Matrix to store stopping probabilities
        # Shape: (batch_size, N) - probability of stopping at time t
        stop_prob_matrix = torch.zeros(batch_size, self.N, device=device)
        
        # Single loop through all time steps
        for t in range(self.N):
            current_price = stock_batch[:, t]
            current_volume = volume_batch[:, t]
            
            current_avg_price = avg_price_batch[:, t]

            # Get network inputs
            trading_inputs, stopping_inputs = self.compute_inputs(
                current_price, t, shares_purchased, current_avg_price
            )
            
            # Trading decision using the formula:
            # v_θ(n,S,A,X,q) = F/A · (n+1)/N · (1 + ṽ_θ(...)) - q
            trading_output = self.trading_network(trading_inputs)
            
            # Implement the trading rate formula
            # Calculate the target cumulative shares: F/A · (n+1)/N · (1 + ṽ_θ(...))
            target_cumulative_shares = (self.F / current_avg_price) * ((t + 1) / self.N) * (1.0 + trading_output.squeeze())
            
            # Trading rate is the incremental purchase: target - already_purchased
            # v_θ = F/A · (n+1)/N · (1 + ṽ_θ(...)) - q
            trading_rate = target_cumulative_shares - shares_purchased
            shares_to_buy = trading_rate
            
            # If continue to buy shares, not exercise
            cost_increase = update_cash_spent(
                X_prev=total_cost,
                v=shares_to_buy,
                S_next=current_price,
                V_next=current_volume,
                dt=self.market_process.dt,
                eta=self.eta,
                phi=self.phi
            ) - total_cost
            
            # Record exercise PnL if we can exercise at this time
            if t >= self.early_exercise_start:
                # ASR payoff if exercised at time t: F - total_cost - termination_cost
                share_remaining = self.F / current_avg_price - shares_purchased
                termination_cost = self.terminal_penalty_coeff * share_remaining ** 2
                exercise_pnl = self.F - share_remaining * current_price - total_cost - termination_cost
                exercise_pnl_matrix[:, t] = exercise_pnl
                
                # Stopping probability at time t
                stopping_output = self.stopping_network(stopping_inputs)
                stop_prob = self.modified_sigmoid(self.nu_phi * stopping_output).squeeze()
                if t == self.N - 1:
                    stop_prob = torch.ones_like(stop_prob)
                else:
                    stop_prob = stop_prob
                stop_prob_matrix[:, t] = stop_prob
            
            # Update tracking variables
            total_cost += cost_increase
            shares_purchased += shares_to_buy
        
        # Calculate expected utility using the recorded PnL matrix
        # For each batch, calculate: ∑_n ∏_{k=early_start}^{n-1} (1-p_k) * p_n * PnL_n
        
        expected_pnl = torch.zeros(batch_size, device=device)
        expected_pnl_squared = torch.zeros(batch_size, device=device)
        
        for n in range(self.early_exercise_start, self.N):
            # Calculate survival probability: ∏_{k=early_start}^{n-1} (1-p_k)
            if n == self.early_exercise_start:
                survival_prob = torch.ones(batch_size, device=device)
            else:
                # Product of (1 - p_k) for k from early_start to n-1
                survival_prob = torch.prod(
                    1.0 - stop_prob_matrix[:, self.early_exercise_start:n], 
                    dim=1
                )
            
            # Probability of stopping at time n
            stop_prob_n = stop_prob_matrix[:, n]
            
            # Weight for this stopping scenario
            weight = survival_prob * stop_prob_n
            
            # PnL if stopped at time n
            pnl_n = exercise_pnl_matrix[:, n]
            
            # Add to expectation
            expected_pnl += weight * pnl_n
            expected_pnl_squared += weight * (pnl_n ** 2)
        
        # Average over batch
        batch_expected_pnl = expected_pnl.mean()
        batch_expected_pnl_squared = expected_pnl_squared.mean()
        
        # Approximate utility using second-order Taylor expansion
        # U ≈ E[PnL] - γ/2 * (E[PnL^2] - E[PnL]^2)
        utility = batch_expected_pnl - 0.5 * self.gamma * (batch_expected_pnl_squared - batch_expected_pnl ** 2)
        
        # Return negative utility as loss (to minimize)
        return -utility
    
class ASREnv_Linear(nn.Module):
    """
    Complete ASR pricing model with linear trading rate and no early exercise.
    """
    
    def __init__(self, 
                 F: float,
                 N: int, 
                 S0: float = 45.0,  # Use realistic example default
                 nu_phi: float = 2.0,
                 early_exercise_start: int = 22,
                 num_paths: int = 10_000,
                 # Market process parameters
                 sigma: float = 0.2,  # Annual volatility from realistic example
                 dt: float = 1/252,  # Daily time step (1/252 of a year)
                 V0: float = 4_000_000,  # Constant volume from realistic example
                 eta: float = 0.0,  # Execution cost parameter
                 phi: float = 0.5,  # Execution cost exponent
                 gamma: float = 2.5e-7,  # Risk aversion parameter
                 terminal_penalty_coeff: float = 0.0,  # Terminal penalty parameter
                 ):
        """
        Initialize ASR pricing model.
        
        Args:
            F: Fixed notional amount
            N: Total number of trading periods
            S0: Initial stock price (default from realistic example)
            nu_phi: Scaling parameter for stopping policy
            early_exercise_start: First day when early exercise is allowed
            num_paths: Number of paths to simulate
            sigma: Annual volatility for Geometric Brownian Motion
            dt: Time step size (1/252 for daily steps, 252 trading days per year)
            V0: Constant market volume
            eta: Execution cost parameter
            phi: Execution cost exponent
            gamma: Risk aversion parameter
            terminal_penalty_coeff: Terminal penalty parameter
        """
        super(ASREnv_Linear, self).__init__()

        self.F = F
        self.N = N
        self.S0 = S0
        self.nu_phi = nu_phi
        self.early_exercise_start = early_exercise_start
        self.num_paths = num_paths
        self.V0 = V0
        self.eta = eta
        self.phi = phi
        self.gamma = gamma
        self.terminal_penalty_coeff = terminal_penalty_coeff
        
        # Initialize market process
        self.market_process = StockPriceProcess(S0=S0, sigma=sigma, dt=dt)
        
        # Storage for simulated paths
        self.stock_paths = None
        self.volume_paths = None
        self.paths_generated = False
        
        self._initialize_paths()

    def _initialize_paths(self):
        """Initialize stock price and volume paths using the market process with tensors."""
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        # Generate stock price paths using the market process (now returns tensors)
        self.stock_paths = self.market_process.simulate_path(
            N=self.N, 
            num_paths=self.num_paths, 
            random_seed=42,
            device=device
        )
        
        # Generate volume paths using the market process (now returns tensors)
        self.volume_paths = self.market_process.simulate_volume(
            N=self.N,
            num_paths=self.num_paths,
            V0=self.V0,
            device=device
        )

        self.avg_price = self._update_avg_price()
        
        self.paths_generated = True

    def _update_avg_price(self):
        """
        Calculate cumulative Volume-Weighted Average Price (VWAP) for each path at each time step.
        
        For each time step t, calculate: VWAP_t = Σ(Price_i * Volume_i) / Σ(Volume_i) for i = 0 to t
        
        Returns:
            Tensor of shape (num_paths, N+1) with cumulative VWAP for each path at each time step
        """
        if self.stock_paths is None or self.volume_paths is None:
            return None
            
        # Calculate cumulative VWAP for each path at each time step
        # stock_paths and volume_paths are (num_paths, N+1)
        batch_size, time_steps = self.stock_paths.shape
        
        # Initialize cumulative VWAP tensor
        cumulative_vwap = torch.zeros_like(self.stock_paths)
        
        # For each time step, calculate cumulative VWAP up to that point
        for t in range(time_steps):
            if t == 0:
                # At t=0, VWAP is just the initial price
                cumulative_vwap[:, t] = self.stock_paths[:, t]
            else:
                # Calculate cumulative price*volume and volume up to time t
                price_volume_cumsum = torch.sum(
                    self.stock_paths[:, :t+1] * self.volume_paths[:, :t+1], 
                    dim=1
                )
                volume_cumsum = torch.sum(self.volume_paths[:, :t+1], dim=1)
                
                # Calculate VWAP up to time t
                cumulative_vwap[:, t] = price_volume_cumsum / (volume_cumsum + 1e-8)
        
        return cumulative_vwap

    def get_training_batch(self, batch_size: int, shuffle: bool = True):
        """
        Get a batch of paths for training.
        
        Args:
            batch_size: Number of paths in the batch
            shuffle: Whether to shuffle the paths
            
        Returns:
            Batch of stock paths and volume paths
        """
        if not self.paths_generated:
            raise ValueError("Paths not generated. Call _initialize_paths() first.")
            
        if shuffle:
            indices = torch.randperm(self.num_paths)[:batch_size]
        else:
            indices = torch.arange(min(batch_size, self.num_paths))
            
        return self.stock_paths[indices], self.volume_paths[indices], self.avg_price[indices]
    
    def eval_strategy(self, batch_size: int = 1000):
        """
        Evaluate trading strategy using single loop with exercise PnL matrix.
        
        The loss function implements the expectation in the given formula:
        -1/γ log E[exp(-γ PnL)] ≈ 1/I ∑∑∏(1-p_k^i)p_n^i PnL_n^i - γ/2[...] - (...)^2
        
        Args:
            batch_size: Size of batch to evaluate
            
        Returns:
            loss: Negative expected utility (loss to minimize)
        """
        stock_batch, volume_batch, avg_price_batch = self.get_training_batch(batch_size)
        device = stock_batch.device
        
        # Initialize tracking variables
        total_cost = torch.zeros(batch_size, device=device)
        shares_purchased = torch.zeros(batch_size, device=device)
        
        # Single loop through all time steps
        for t in range(self.N):
            current_price = stock_batch[:, t]
            current_volume = volume_batch[:, t]
            
            current_avg_price = avg_price_batch[:, t]

            target_cumulative_shares = (self.F / current_avg_price) * ((t + 1) / self.N)
 
            trading_rate = target_cumulative_shares - shares_purchased
            shares_to_buy = trading_rate
            
            # If continue to buy shares, not exercise
            cost_increase = update_cash_spent(
                X_prev=total_cost,
                v=shares_to_buy,
                S_next=current_price,
                V_next=current_volume,
                dt=self.market_process.dt,
                eta=self.eta,
                phi=self.phi
            ) - total_cost
            
            # # Record exercise PnL if we can exercise at this time
            # if t >= self.early_exercise_start:
            #     # ASR payoff if exercised at time t: F - total_cost - termination_cost
            #     share_remaining = self.F / current_avg_price - shares_purchased
            #     termination_cost = terminal_penalty_coeff * share_remaining ** 2
            #     exercise_pnl = self.F - share_remaining * current_price - total_cost - termination_cost
            #     exercise_pnl_matrix[:, t] = exercise_pnl
            
            # Update tracking variables
            total_cost += cost_increase
            shares_purchased += shares_to_buy

        pnl = self.F -  total_cost
        # Approximate utility using second-order Taylor expansion
        # U ≈ E[PnL] - γ/2 * (E[PnL^2] - E[PnL]^2)
        utility = pnl.mean() - 0.5 * self.gamma * (pnl.var() - pnl.mean() ** 2)
        
        # Return negative utility as loss (to minimize)
        return -utility


if __name__ == "__main__":
    def test_eval_strategy():
        """Test the eval_strategy method with minimal parameters."""
        print("\n" + "=" * 50)
        print("Testing Strategy Evaluation")
        print("=" * 50)

        # Create minimal ASREnv
        asr_env = ASREnv(
            F=900_000_000.0,
            N=63,                # Very short horizon
            S0=45.0,
            num_paths=1000,        # Very few paths
            early_exercise_start=22,  # Early exercise from day 22
            gamma=0.0,  # Set gamma to 0 for testing
        )
        
        print("\n1. Running eval_strategy...")
        try:
            loss = asr_env.eval_strategy(batch_size=100)
            print(f"  Loss value: {loss.item():.6f}")
            print(f"  Loss type: {type(loss)}")
            print(f"  Loss requires_grad: {loss.requires_grad}")
            print("  ✓ eval_strategy completed successfully!")
            
            # Test backward pass
            print("\n2. Testing backward pass...")
            loss.backward()
            print("  ✓ Backward pass completed!")
            
        except Exception as e:
            print(f"  ✗ Error in eval_strategy: {e}")
    
    def test_asr_linear():
        """Test the ASREnv_Linear class with minimal parameters."""
        print("\n" + "=" * 50)
        print("Testing ASREnv_Linear")
        print("=" * 50)
        
        # Create minimal ASREnv_Linear
        try:
            print("\n1. Creating ASREnv_Linear instance...")
            asr_linear = ASREnv_Linear(
                F=900_000_000.0,
                N=63,                # Very short horizon
                S0=45.0,
                num_paths=1000,
                early_exercise_start=22,  # Early exercise from day 22
                gamma=0.0  # Set gamma to 0 for testing
            )
            print("  ✓ ASREnv_Linear created successfully!")
            
            # Test eval_strategy
            print("\n4. Running eval_strategy...")
            loss = asr_linear.eval_strategy(batch_size=100)
            print(f"  Loss value: {loss.item():.6f}")
            print(f"  Loss type: {type(loss)}")
            print(f"  Loss requires_grad: {loss.requires_grad}")
            print("  ✓ eval_strategy completed successfully!")
        
        except Exception as e:
            print(f"  ✗ Error in ASREnv_Linear test: {e}")
        
    # Run the test
    print("Starting Neural Network Tests...")
    print(f"PyTorch version: {torch.__version__}")
    test_eval_strategy()
    test_asr_linear()