#!/usr/bin/env python3
"""
Simple test case for debugging the neural networks in ASB.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the Python path to make ASB a package
sys.path.insert(0, os.path.dirname(__file__))

from ASB.neural_networks import TradingRateNetwork, StoppingPolicyNetwork, ModifiedSigmoid, ASREnv


def test_eval_strategy():
    """Test the eval_strategy method with minimal parameters."""
    print("\n" + "=" * 50)
    print("Testing Strategy Evaluation")
    print("=" * 50)
    
    # Create minimal ASREnv
    asr_env = ASREnv(
        F=1000.0,
        N=3,                # Very short horizon
        S0=100.0,
        num_paths=5,        # Very few paths
        early_exercise_start=1  # Early exercise from day 1
    )
    
    print("\n1. Running eval_strategy...")
    try:
        loss = asr_env.eval_strategy(batch_size=3)
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
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Starting Neural Network Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        test_eval_strategy()
        
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    
    return 0


if __name__ == "__main__":
    main()
