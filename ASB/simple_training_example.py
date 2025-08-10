#!/usr/bin/env python3
"""
Simple training example for ASR Neural Networks.

This is a minimal example showing how to train the networks.
"""

import torch
import torch.optim as optim

# Import the neural networks
try:
    from ASB.neural_networks import ASREnv
except ImportError:
    from neural_networks import ASREnv


def simple_training_example():
    """Simple training example."""
    print("Simple ASR Network Training Example")
    print("="*40)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the ASR environment
    asr_env = ASREnv(
        F=900_000_000.0,        # Notional amount
        N=63,                   # Trading periods
        S0=45.0,               # Initial stock price
        num_paths=10_000,       # Number of simulation paths
        early_exercise_start=22, # Early exercise start
        gamma=2.5e-7           # Risk aversion
    ).to(device)
    
    # Get all trainable parameters from both networks
    parameters = list(asr_env.trading_network.parameters()) + \
                list(asr_env.stopping_network.parameters())
    
    # Create optimizer
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    
    print(f"Total trainable parameters: {sum(p.numel() for p in parameters)}")
    print("Starting training...")
    
    # Training loop
    num_epochs = 500
    batch_size = 1000
    
    for epoch in range(num_epochs):
        # Set to training mode
        asr_env.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass - compute loss
        loss = asr_env.eval_strategy(batch_size=batch_size)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if epoch % 50 == 0:
            with torch.no_grad():
                val_loss = asr_env.eval_strategy(batch_size=2000)
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | "
                  f"Val Loss: {val_loss.item():.6f}")
    
    print("Training completed!")
    
    # Save the trained model
    torch.save({
        'model_state_dict': asr_env.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'trained_asr_model.pth')
    print("Model saved as 'trained_asr_model.pth'")


if __name__ == "__main__":
    simple_training_example()
