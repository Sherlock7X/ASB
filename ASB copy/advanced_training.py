#!/usr/bin/env python3
"""
Advanced training script with hyperparameter optimization for ASR Neural Networks.
"""

import torch
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from datetime import datetime
import logging

# Import the neural networks
try:
    from ASB.neural_networks import ASREnv, ASREnv_Linear
except ImportError:
    from neural_networks import ASREnv, ASREnv_Linear


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 200, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class AdvancedASRTrainer:
    """Advanced trainer with hyperparameter optimization."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize baseline for comparison
        self.baseline_env = ASREnv_Linear(**config['model']).to(self.device)
        self.baseline_loss = self._get_baseline_performance()
        self.logger.info(f"Baseline performance: {self.baseline_loss:.6f}")
        
    def _get_baseline_performance(self, batch_size: int = 5000):
        """Get baseline linear strategy performance."""
        with torch.no_grad():
            return self.baseline_env.eval_strategy(batch_size=batch_size).item()
    
    def create_model(self, model_config: dict = None):
        """Create ASR model with given configuration."""
        config = model_config or self.config['model']
        return ASREnv(**config).to(self.device)
    
    def train_model(self, 
                   model: ASREnv, 
                   training_config: dict,
                   save_path: str = None) -> dict:
        """Train a single model configuration."""
        
        # Get parameters from both networks
        params = list(model.trading_network.parameters()) + \
                list(model.stopping_network.parameters())
        
        # Create optimizer
        optimizer = optim.Adam(params, lr=training_config['learning_rate'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=training_config['lr_scheduler']['factor'],
            patience=training_config['lr_scheduler']['patience'],
            min_lr=training_config['lr_scheduler']['min_lr']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=training_config.get('early_stopping_patience', 200)
        )
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        best_val_loss = float('inf')
        
        self.logger.info(f"Starting training for {training_config['num_epochs']} epochs")
        
        for epoch in range(training_config['num_epochs']):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            train_loss = model.eval_strategy(batch_size=training_config['batch_size'])
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=training_config['gradient_clip_norm']
            )
            
            optimizer.step()
            
            history['train_losses'].append(train_loss.item())
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Validation step
            if epoch % training_config['val_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = model.eval_strategy(
                        batch_size=training_config['val_batch_size']
                    ).item()
                
                history['val_losses'].append(val_loss)
                history['epochs'].append(epoch)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        self._save_model(model, optimizer, history, save_path)
                
                # Print progress
                if epoch % training_config['print_freq'] == 0:
                    improvement = (self.baseline_loss - val_loss) / abs(self.baseline_loss) * 100
                    self.logger.info(
                        f"Epoch {epoch:4d} | Train: {train_loss.item():.6f} | "
                        f"Val: {val_loss:.6f} | Improvement: {improvement:+.2f}% | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                
                # Early stopping check
                if early_stopping(val_loss):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        history['best_val_loss'] = best_val_loss
        history['improvement'] = (self.baseline_loss - best_val_loss) / abs(self.baseline_loss) * 100
        
        return history
    
    def _save_model(self, model, optimizer, history, filepath):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': self.config
        }, filepath)
    
    def hyperparameter_search(self, search_space: dict, max_trials: int = 10):
        """Perform hyperparameter search."""
        self.logger.info("Starting hyperparameter search")
        
        # Generate combinations
        keys = list(search_space.keys())
        values = [search_space[key] for key in keys]
        combinations = list(itertools.product(*values))
        
        # Limit trials
        if len(combinations) > max_trials:
            combinations = combinations[:max_trials]
        
        results = []
        
        for i, combination in enumerate(combinations):
            self.logger.info(f"Trial {i+1}/{len(combinations)}")
            
            # Create configuration for this trial
            trial_config = self.config['training'].copy()
            for key, value in zip(keys, combination):
                if '.' in key:  # Nested key like 'lr_scheduler.factor'
                    parts = key.split('.')
                    trial_config[parts[0]][parts[1]] = value
                else:
                    trial_config[key] = value
            
            self.logger.info(f"Trial config: {trial_config}")
            
            # Create and train model
            model = self.create_model()
            
            try:
                history = self.train_model(
                    model, 
                    trial_config,
                    f"models/trial_{i}_model.pth"
                )
                
                results.append({
                    'trial': i,
                    'config': trial_config,
                    'best_val_loss': history['best_val_loss'],
                    'improvement': history['improvement'],
                    'history': history
                })
                
                self.logger.info(
                    f"Trial {i} completed | Best Val Loss: {history['best_val_loss']:.6f} | "
                    f"Improvement: {history['improvement']:.2f}%"
                )
                
            except Exception as e:
                self.logger.error(f"Trial {i} failed: {e}")
                results.append({
                    'trial': i,
                    'config': trial_config,
                    'best_val_loss': float('inf'),
                    'improvement': -float('inf'),
                    'error': str(e)
                })
        
        # Find best configuration
        best_result = min(results, key=lambda x: x.get('best_val_loss', float('inf')))
        
        self.logger.info("Hyperparameter search completed")
        self.logger.info(f"Best configuration: {best_result['config']}")
        self.logger.info(f"Best validation loss: {best_result['best_val_loss']:.6f}")
        self.logger.info(f"Best improvement: {best_result['improvement']:.2f}%")
        
        return results, best_result
    
    def plot_hyperparameter_results(self, results: list, save_path: str = None):
        """Plot hyperparameter search results."""
        if not results:
            return
        
        # Extract data
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            self.logger.warning("No valid results to plot")
            return
        
        trial_nums = [r['trial'] for r in valid_results]
        val_losses = [r['best_val_loss'] for r in valid_results]
        improvements = [r['improvement'] for r in valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation losses
        ax1.bar(trial_nums, val_losses)
        ax1.axhline(y=self.baseline_loss, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Best Validation Loss')
        ax1.set_title('Hyperparameter Search - Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvements
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.bar(trial_nums, improvements, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Improvement over Baseline (%)')
        ax2.set_title('Hyperparameter Search - Improvement')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Hyperparameter results saved to {save_path}")
        
        plt.show()


def main():
    """Main function for advanced training."""
    
    # Setup
    logger = setup_logging()
    
    # Load configuration
    with open('training_config.json', 'r') as f:
        config = json.load(f)
    
    logger.info("Loaded configuration")
    
    # Create directories
    for path in config['paths'].values():
        Path(path).mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = AdvancedASRTrainer(config, logger)
    
    # Define hyperparameter search space
    search_space = {
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'batch_size': [500, 1000, 2000],
        'lr_scheduler.factor': [0.3, 0.5, 0.7],
        'lr_scheduler.patience': [50, 100, 150]
    }
    
    # Perform hyperparameter search
    results, best_result = trainer.hyperparameter_search(
        search_space, max_trials=15
    )
    
    # Plot results
    trainer.plot_hyperparameter_results(
        results, 
        save_path=f"{config['paths']['plot_dir']}/hyperparameter_search.png"
    )
    
    # Train final model with best configuration
    logger.info("Training final model with best configuration")
    best_model = trainer.create_model()
    final_history = trainer.train_model(
        best_model,
        best_result['config'],
        save_path=f"{config['paths']['save_dir']}/best_final_model.pth"
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model improvement: {final_history['improvement']:.2f}%")


if __name__ == "__main__":
    main()
