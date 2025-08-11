#!/usr/bin/env python3
"""
Training script for ASR Neural Networks.

This script demonstrates how to train the TradingRateNetwork and StoppingPolicyNetwork
within the ASREnv model using gradient descent optimization.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import the neural networks
try:
    from ASB.neural_networks import ASREnv, ASREnv_Linear
except ImportError:
    from neural_networks import ASREnv, ASREnv_Linear


class ASRTrainer:
    """Trainer class for ASR neural networks."""
    
    def __init__(self, 
                 F: float = 900_000_000.0,
                 N: int = 63,
                 S0: float = 45.0,
                 num_paths: int = 50_000,
                 early_exercise_start: int = 22,
                 gamma: float = 2.5e-7,
                 learning_rate: float = 1e-4,
                 device: str = None):
        """
        Initialize the trainer.
        
        Args:
            F: Fixed notional amount
            N: Total number of trading periods
            S0: Initial stock price
            num_paths: Number of simulation paths
            early_exercise_start: First day when early exercise is allowed
            gamma: Risk aversion parameter
            learning_rate: Learning rate for optimization
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize the ASR environment
        self.asr_env = ASREnv(
            F=F,
            N=N,
            S0=S0,
            num_paths=num_paths,
            early_exercise_start=early_exercise_start,
            gamma=gamma
        ).to(self.device)
        
        # Initialize optimizer for both networks
        # Get parameters from both trading and stopping networks
        params = list(self.asr_env.trading_network.parameters()) + \
                list(self.asr_env.stopping_network.parameters())
        
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create baseline for comparison
        self.baseline_env = ASREnv_Linear(
            F=F, N=N, S0=S0, num_paths=num_paths, 
            early_exercise_start=early_exercise_start, gamma=gamma
        ).to(self.device)
        
        print("ASR Trainer initialized successfully!")
        
    def train_step(self, batch_size: int = 1000):
        """
        Perform one training step.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            loss: Training loss value
        """
        self.asr_env.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        loss = self.asr_env.eval_strategy(batch_size=batch_size)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.asr_env.trading_network.parameters()) + 
            list(self.asr_env.stopping_network.parameters()), 
            max_norm=1.0
        )
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, batch_size: int = 2000):
        """
        Perform validation step.
        
        Args:
            batch_size: Size of validation batch
            
        Returns:
            loss: Validation loss value
        """
        self.asr_env.eval()
        with torch.no_grad():
            val_loss = self.asr_env.eval_strategy(batch_size=batch_size)
        return val_loss.item()
    
    def get_baseline_performance(self, batch_size: int = 2000):
        """Get baseline linear strategy performance."""
        with torch.no_grad():
            baseline_loss = self.baseline_env.eval_strategy(batch_size=batch_size)
        return baseline_loss.item()
    
    def train(self, 
              num_epochs: int = 1000,
              batch_size: int = 1000,
              val_batch_size: int = 2000,
              val_freq: int = 50,
              print_freq: int = 100,
              lr_schedule: bool = True,
              save_best: bool = True,
              save_dir: str = "models"):
        """
        Train the neural networks.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            val_batch_size: Validation batch size
            val_freq: Validation frequency (every N epochs)
            print_freq: Print frequency (every N epochs)
            lr_schedule: Whether to use learning rate scheduling
            save_best: Whether to save the best model
            save_dir: Directory to save models
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Batch size: {batch_size}, Validation frequency: {val_freq}")
        
        # Create save directory
        if save_best:
            Path(save_dir).mkdir(exist_ok=True)
        
        # Learning rate scheduler
        if lr_schedule:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=100
            )
        
        # Get baseline performance
        baseline_loss = self.get_baseline_performance(val_batch_size)
        print(f"Baseline (linear strategy) loss: {baseline_loss:.6f}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training step
            train_loss = self.train_step(batch_size)
            self.train_losses.append(train_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Validation step
            if epoch % val_freq == 0:
                val_loss = self.validate(val_batch_size)
                self.val_losses.append(val_loss)
                
                # Update learning rate scheduler
                if lr_schedule:
                    scheduler.step(val_loss)
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"{save_dir}/best_asr_model.pth")
                
                # Print progress
                if epoch % print_freq == 0:
                    improvement = (baseline_loss - val_loss) / abs(baseline_loss) * 100
                    print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"Improvement vs Baseline: {improvement:+.2f}% | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            elif epoch % print_freq == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Improvement vs baseline: {(baseline_loss - best_val_loss) / abs(baseline_loss) * 100:+.2f}%")
        
        # Save final model
        if save_best:
            self.save_model(f"{save_dir}/final_asr_model.pth")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'asr_env_state_dict': self.asr_env.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.asr_env.load_state_dict(checkpoint['asr_env_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Validation loss
        if self.val_losses:
            val_epochs = np.arange(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses))[:len(self.val_losses)]
            axes[0, 1].plot(val_epochs, self.val_losses, 'r-', label='Validation')
            axes[0, 1].axhline(y=self.get_baseline_performance(), color='k', linestyle='--', label='Baseline')
            axes[0, 1].set_title('Validation Loss vs Baseline')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].semilogy(self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss comparison (recent epochs)
        if len(self.train_losses) > 100:
            recent_start = len(self.train_losses) - 100
            axes[1, 1].plot(self.train_losses[recent_start:], label='Train (recent)')
            if self.val_losses:
                recent_val_start = max(0, len(self.val_losses) - 20)
                recent_val_epochs = np.arange(recent_start, len(self.train_losses), 5)[:len(self.val_losses[recent_val_start:])]
                if len(recent_val_epochs) == len(self.val_losses[recent_val_start:]):
                    axes[1, 1].plot(recent_val_epochs, self.val_losses[recent_val_start:], 'r-', label='Val (recent)')
            axes[1, 1].set_title('Recent Training Progress')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
    
    def analyze_networks(self):
        """Analyze the trained networks."""
        print("\n" + "="*50)
        print("NETWORK ANALYSIS")
        print("="*50)
        
        # Check gradient norms
        trading_grad_norm = 0
        stopping_grad_norm = 0
        
        # Get sample inputs to compute gradients
        sample_batch_size = 100
        stock_batch, volume_batch, avg_price_batch = self.asr_env.get_training_batch(sample_batch_size)
        
        # Sample inputs for both networks
        trading_inputs, stopping_inputs = self.asr_env.compute_inputs(
            stock_batch[:, 10], 10, torch.zeros(sample_batch_size), avg_price_batch[:, 10]
        )
        
        # Forward pass to compute gradients
        trading_output = self.asr_env.trading_network(trading_inputs)
        stopping_output = self.asr_env.stopping_network(stopping_inputs)
        
        # Compute gradient norms
        loss = self.asr_env.eval_strategy(batch_size=sample_batch_size)
        loss.backward()
        
        for param in self.asr_env.trading_network.parameters():
            if param.grad is not None:
                trading_grad_norm += param.grad.norm().item() ** 2
        
        for param in self.asr_env.stopping_network.parameters():
            if param.grad is not None:
                stopping_grad_norm += param.grad.norm().item() ** 2
        
        trading_grad_norm = trading_grad_norm ** 0.5
        stopping_grad_norm = stopping_grad_norm ** 0.5
        
        print(f"Trading Network Gradient Norm: {trading_grad_norm:.6f}")
        print(f"Stopping Network Gradient Norm: {stopping_grad_norm:.6f}")
        
        # Parameter statistics
        trading_params = sum(p.numel() for p in self.asr_env.trading_network.parameters())
        stopping_params = sum(p.numel() for p in self.asr_env.stopping_network.parameters())
        
        print(f"Trading Network Parameters: {trading_params}")
        print(f"Stopping Network Parameters: {stopping_params}")
        print(f"Total Parameters: {trading_params + stopping_params}")


def main():
    """Main training function."""
    print("ASR Neural Network Training")
    print("="*50)
    
    # Training configuration
    config = {
        'F': 900_000_000.0,
        'N': 63,
        'S0': 45.0,
        'num_paths': 50_000,
        'early_exercise_start': 2,
        'gamma': 2.5e-8,
        'learning_rate': 1e-3,
        'num_epochs': 1500,
        'batch_size': 1000,
        'val_batch_size': 2000,
        'val_freq': 50,
        'print_freq': 100,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = ASRTrainer(
        F=config['F'],
        N=config['N'],
        S0=config['S0'],
        num_paths=config['num_paths'],
        early_exercise_start=config['early_exercise_start'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate']
    )
    
    # Train the networks
    trainer.train(
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        val_freq=config['val_freq'],
        print_freq=config['print_freq']
    )
    
    # Analyze results
    trainer.analyze_networks()
    
    # Plot training progress
    trainer.plot_training_progress('plots/training_progress.png')
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
