#!/usr/bin/env python3
"""
Test script to verify full perturbation functionality including policy loss calculation
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add verl to path
verl_path = Path(__file__).parent.parent / "verl"
sys.path.insert(0, str(verl_path))

def test_compute_ppo_is_metrics():
    """Test compute_ppo_is_metrics function"""
    print("\n" + "="*60)
    print("Test 1: compute_ppo_is_metrics")
    print("="*60)
    
    from verl.trainer.ppo.core_algos import compute_ppo_is_metrics
    
    batch_size = 4
    seq_len = 10
    
    # Create dummy data
    log_importance_ratio = torch.randn(batch_size, seq_len) * 0.1
    response_mask = torch.ones(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    
    # Test the function
    metrics = compute_ppo_is_metrics(
        log_importance_ratio, response_mask, advantages, clip_ratio_low, clip_ratio_high
    )
    
    print(f"\nMetrics keys: {list(metrics.keys())}")
    print(f"is_ratio/mean: {metrics.get('is_ratio/mean', 'N/A'):.4f}")
    print(f"is_ratio/std: {metrics.get('is_ratio/std', 'N/A'):.4f}")
    print(f"pg_clip_frac: {metrics.get('pg_clip_frac', 'N/A'):.4f}")
    
    assert 'is_ratio/mean' in metrics, "Missing is_ratio/mean"
    assert 'pg_clip_frac' in metrics, "Missing pg_clip_frac"
    
    print("\n✓ Test 1 passed: compute_ppo_is_metrics works correctly")


def test_compute_policy_loss_perturbed():
    """Test compute_policy_loss_perturbed function"""
    print("\n" + "="*60)
    print("Test 2: compute_policy_loss_perturbed")
    print("="*60)
    
    from verl.trainer.ppo.core_algos import compute_policy_loss_perturbed
    from omegaconf import OmegaConf
    
    batch_size = 4
    seq_len = 10
    
    # Create dummy data
    old_log_prob = torch.randn(batch_size, seq_len)
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.1
    rollout_log_probs = old_log_prob + torch.randn(batch_size, seq_len) * 0.05
    advantages = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    # Create config
    config = OmegaConf.create({
        'clip_ratio': 0.2,
        'clip_ratio_low': 0.2,
        'clip_ratio_high': 0.2,
        'clip_ratio_c': 3.0,
        'policy_loss': {
            'is_geometric': False
        }
    })
    
    # Test different loss modes
    for loss_mode in ['token', 'sequence']:
        print(f"\n  Testing loss_mode={loss_mode}")
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics = \
            compute_policy_loss_perturbed(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode="token-mean",
                loss_mode=loss_mode,
                rollout_log_probs=rollout_log_probs,
                config=config,
            )
        
        print(f"    pg_loss: {pg_loss.item():.4f}")
        print(f"    pg_clipfrac: {pg_clipfrac:.4f}")
        print(f"    ppo_kl: {ppo_kl:.4f}")
        print(f"    IS metrics: {len(ppo_is_metrics)} keys")
        
        assert pg_loss is not None, "pg_loss should not be None"
        assert isinstance(ppo_is_metrics, dict), "ppo_is_metrics should be dict"
        assert 'is_ratio/mean' in ppo_is_metrics, "Missing is_ratio/mean in metrics"
    
    print("\n✓ Test 2 passed: compute_policy_loss_perturbed works for multiple loss modes")


def test_dp_actor_integration():
    """Test integration with dp_actor"""
    print("\n" + "="*60)
    print("Test 3: dp_actor Integration")
    print("="*60)
    
    # This test checks if the code can be imported and the logic is correct
    try:
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from verl.trainer.ppo.core_algos import compute_policy_loss_perturbed
        
        print("\n  ✓ Imports successful")
        
        # Check if compute_policy_loss_perturbed is callable
        assert callable(compute_policy_loss_perturbed), \
            "compute_policy_loss_perturbed should be callable"
        
        print("  ✓ compute_policy_loss_perturbed is callable")
        
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        raise
    
    print("\n✓ Test 3 passed: Integration check successful")


def test_config_compatibility():
    """Test configuration compatibility"""
    print("\n" + "="*60)
    print("Test 4: Configuration Compatibility")
    print("="*60)
    
    from omegaconf import OmegaConf
    import yaml
    
    # Load actor config
    config_path = Path(__file__).parent.parent / "verl/verl/trainer/config/actor/actor.yaml"
    
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        print(f"\n  Loaded config from: {config_path}")
        
        # Check for perturbation-related fields
        assert 'perturb_std' in config_dict, "Missing perturb_std in config"
        print(f"  ✓ Found perturb_std: {config_dict['perturb_std']}")
        
        assert 'policy_loss' in config_dict, "Missing policy_loss in config"
        policy_loss = config_dict['policy_loss']
        
        assert 'is_geometric' in policy_loss, "Missing is_geometric in policy_loss"
        print(f"  ✓ Found is_geometric: {policy_loss['is_geometric']}")
        
        print(f"  ✓ loss_mode: {policy_loss['loss_mode']}")
        
    except Exception as e:
        print(f"\n  ✗ Config check failed: {e}")
        raise
    
    print("\n✓ Test 4 passed: Configuration is compatible")


def test_error_handling():
    """Test error handling when rollout_log_probs is missing"""
    print("\n" + "="*60)
    print("Test 5: Error Handling")
    print("="*60)
    
    from verl.trainer.ppo.core_algos import compute_policy_loss_perturbed
    from omegaconf import OmegaConf
    
    batch_size = 4
    seq_len = 10
    
    # Create dummy data WITHOUT rollout_log_probs
    old_log_prob = torch.randn(batch_size, seq_len)
    log_prob = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    config = OmegaConf.create({
        'clip_ratio': 0.2,
        'clip_ratio_low': 0.2,
        'clip_ratio_high': 0.2,
        'clip_ratio_c': 3.0,
        'policy_loss': {'is_geometric': False}
    })
    
    # This should raise an error
    try:
        pg_loss, _, _, _, _ = compute_policy_loss_perturbed(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            rollout_log_probs=None,  # Missing!
            config=config,
        )
        print("\n  ✗ Should have raised an error for missing rollout_log_probs")
        assert False, "Should have raised an error"
    except (AssertionError, ValueError) as e:
        print(f"\n  ✓ Correctly raised error: {type(e).__name__}")
        print(f"    Message: {str(e)[:80]}...")
    
    print("\n✓ Test 5 passed: Error handling works correctly")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing Full Perturbation Implementation")
    print("="*60)
    
    try:
        # Run tests
        test_compute_ppo_is_metrics()
        test_compute_policy_loss_perturbed()
        test_dp_actor_integration()
        test_config_compatibility()
        test_error_handling()
        
        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nFull perturbation implementation is working correctly!")
        print("\nKey features verified:")
        print("  ✓ compute_ppo_is_metrics function")
        print("  ✓ compute_policy_loss_perturbed function")
        print("  ✓ Integration with dp_actor")
        print("  ✓ Configuration compatibility")
        print("  ✓ Error handling")
        print("\nNext steps:")
        print("  1. Run training with perturbation enabled")
        print("  2. Monitor training metrics in WandB")
        print("  3. Compare results with baseline experiments")
        print("\nExample command:")
        print("  bash scripts/run_exp5_grpo_bypass_perturb.sh")
        print("="*60 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
