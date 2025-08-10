# Training ASR Neural Networks

This guide explains how to train the TradingRateNetwork and StoppingPolicyNetwork in the ASR (Accumulator Swap Rate) model.

## Overview

The ASR model contains two neural networks:

1. **TradingRateNetwork** (4 inputs → 50 hidden → 1 output)
   - Inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0, qA/F - 1/2)
   - Outputs: Adjustment to naive trading schedule

2. **StoppingPolicyNetwork** (3 inputs → 50 hidden → 1 output)
   - Inputs: (n/N - 1/2, S/S0 - 1, (A-S)/S0)
   - Outputs: Stopping decision parameter

## Quick Start

### 1. Simple Training Example

Run the basic training script:

```bash
python simple_training_example.py
```

This will:
- Train both networks for 500 epochs
- Use Adam optimizer with learning rate 1e-4
- Apply gradient clipping for stability
- Save the trained model as 'trained_asr_model.pth'

### 2. Full Training Pipeline

For comprehensive training with validation and monitoring:

```bash
python train_asr_networks.py
```

Features:
- Validation monitoring
- Learning rate scheduling
- Early stopping
- Training progress plots
- Model checkpointing
- Performance comparison with baseline

### 3. Advanced Training with Hyperparameter Search

For optimal performance:

```bash
python advanced_training.py
```

This includes:
- Hyperparameter optimization
- Multiple trial runs
- Automatic best model selection
- Comprehensive logging
- Result visualization

## Training Configuration

### Key Parameters

```python
# Model parameters
F = 900_000_000.0        # Notional amount (€900M)
N = 63                   # Trading periods (days)
S0 = 45.0               # Initial stock price
num_paths = 50_000      # Simulation paths
gamma = 2.5e-7          # Risk aversion parameter

# Training parameters
learning_rate = 1e-4    # Adam learning rate
batch_size = 1000       # Training batch size
num_epochs = 2000       # Training epochs
```

### Configuration File

Modify `training_config.json` to customize training:

```json
{
  "training": {
    "num_epochs": 2000,
    "batch_size": 1000,
    "learning_rate": 1e-4,
    "gradient_clip_norm": 1.0
  },
  "model": {
    "F": 900000000.0,
    "N": 63,
    "gamma": 2.5e-7
  }
}
```

## Training Process

### 1. Loss Function

The networks are trained to minimize the negative expected utility:

```
Loss = -E[U(PnL)] ≈ -E[PnL] + γ/2 * Var[PnL]
```

Where:
- `PnL` is the profit/loss from the ASR strategy
- `γ` is the risk aversion parameter
- The expectation is over all possible exercise scenarios

### 2. Optimization

- **Optimizer**: Adam with adaptive learning rates
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
- **Early Stopping**: Stops training when no improvement for 200 epochs

### 3. Validation

- Periodic validation on larger batch sizes
- Comparison with linear baseline strategy
- Monitoring of improvement percentage

## Key Training Considerations

### 1. Gradient Stability

The ASR loss function can have high variance, so:
- Use gradient clipping (max norm 1.0)
- Start with small learning rates (1e-4 or 1e-5)
- Monitor gradient norms during training

### 2. Initialization

Networks are initialized with small weights:
- Weight matrices: Normal(0, 0.01)
- Biases: Constant(0)
- This ensures starting behavior close to the linear benchmark

### 3. Batch Size Trade-offs

- **Larger batches**: More stable gradients, slower training
- **Smaller batches**: Faster training, more noisy gradients
- Recommended: 1000-2000 for training, 2000-5000 for validation

### 4. Early Exercise Modeling

The stopping network only affects periods t ≥ early_exercise_start:
- Default: early_exercise_start = 22 (day 22 onwards)
- Before this, no early exercise is possible
- Network learns optimal stopping policy for remaining periods

## Monitoring Training

### 1. Loss Metrics

Watch for:
- Decreasing training loss
- Stable validation loss
- Positive improvement over baseline
- Convergence (loss plateaus)

### 2. Network Analysis

Check:
- Gradient norms (should be reasonable, not zero or exploding)
- Parameter magnitudes
- Learning rate decay
- Validation vs training gap

### 3. Performance Comparison

The baseline is a linear trading strategy:
- `v_linear = F/A * (n+1)/N - q`
- No early exercise optimization
- Networks should improve upon this baseline

## Common Issues and Solutions

### 1. Exploding Gradients

**Symptoms**: Loss becomes NaN or very large
**Solutions**:
- Reduce learning rate
- Increase gradient clipping strength
- Check network initialization

### 2. No Learning

**Symptoms**: Loss doesn't decrease
**Solutions**:
- Increase learning rate
- Check gradient flow
- Verify loss function implementation
- Increase batch size for stability

### 3. Overfitting

**Symptoms**: Training loss decreases but validation loss increases
**Solutions**:
- Reduce model complexity
- Add regularization
- Early stopping
- More simulation paths

### 4. Slow Convergence

**Symptoms**: Very slow improvement
**Solutions**:
- Increase learning rate
- Adjust learning rate schedule
- Larger batch sizes
- Better initialization

## Advanced Features

### 1. Hyperparameter Search

The advanced training script searches over:
- Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- Batch sizes: [500, 1000, 2000]
- Scheduler parameters
- Optimization settings

### 2. Multi-GPU Training

For large-scale training:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### 3. Mixed Precision Training

For faster training with large models:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model.eval_strategy(batch_size)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Results Interpretation

### Success Metrics

1. **Positive Improvement**: Networks should beat baseline (>0% improvement)
2. **Convergence**: Training and validation losses should stabilize
3. **Reasonable Magnitude**: Networks shouldn't make extreme adjustments
4. **Gradient Health**: Gradient norms should be stable

### Expected Performance

- **Good Result**: 5-15% improvement over linear baseline
- **Excellent Result**: >15% improvement
- **Poor Result**: <0% improvement (worse than baseline)

## File Structure

```
ASB/
├── neural_networks.py              # Core network definitions
├── simple_training_example.py      # Basic training script
├── train_asr_networks.py          # Full training pipeline
├── advanced_training.py           # Hyperparameter optimization
├── training_config.json           # Configuration file
├── models/                         # Saved models
├── plots/                          # Training plots
└── logs/                          # Training logs
```

## Next Steps

After training:

1. **Evaluate Performance**: Compare with baseline on test data
2. **Analyze Strategies**: Visualize learned trading and stopping policies
3. **Sensitivity Analysis**: Test robustness to market parameters
4. **Deployment**: Use trained model for ASR pricing

For questions or issues, refer to the main documentation or create an issue.
