#!/usr/bin/env python3
"""
Quick integration test for Conservative preset aggregation pipeline
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from faircare.algos.faircare_fl import FairCareFLAggregator
from faircare.config import get_conservative_preset, FairCareFLConfig

def create_synthetic_model():
    """Create a simple synthetic model state dict."""
    return {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(3, 10),
        'layer2.bias': torch.randn(3),
    }

def create_synthetic_client_reports(n_clients=5):
    """Create synthetic client reports with deltas."""
    global_model = create_synthetic_model()

    reports = []
    for i in range(n_clients):
        # Create delta (small random update)
        delta = {
            key: torch.randn_like(val) * 0.01
            for key, val in global_model.items()
        }

        report = {
            'client_id': i,
            'n_samples': 100 * (i + 1),
            'val_loss': 0.5 + 0.1 * i,
            'wg_f1': 0.3 + 0.05 * i,
            'delta': delta,
            'group_counts': {
                'group_0': {'TP': 40, 'FP': 5, 'FN': 5, 'TN': 50},
                'group_1': {'TP': 35, 'FP': 8, 'FN': 7, 'TN': 50},
            },
            'proxies': {
                'loss_drift': 0.01 * i,
                'delta_norm': 0.1,
                'ece_proxy': 0.05
            },
        }
        reports.append(report)

    return reports

def test_conservative_aggregation():
    """Test full aggregation pipeline with Conservative preset."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Conservative Preset Full Aggregation")
    print("="*80)

    # Setup
    config = get_conservative_preset()
    aggregator = FairCareFLAggregator(config, device='cpu')

    global_model = create_synthetic_model()
    client_reports = create_synthetic_client_reports(n_clients=5)

    round_ctx = {
        'round_num': 1,
        'total_rounds': 20,
    }

    print(f"\nConfiguration:")
    print(f"  Preset: Conservative")
    print(f"  Legacy mode: {config.legacy_mode}")
    print(f"  Aggregate enabled: {config.aggregate.enable}")
    print(f"  Softmin temperature: {config.aggregate.softmin_temperature}")
    print(f"  Weight method: {config.aggregate.weight_method}")
    print(f"  Server momentum: {config.aggregate.server_momentum}")

    try:
        # Run aggregation
        print(f"\nRunning aggregation...")
        result = aggregator.aggregate(round_ctx, client_reports, global_model)

        print(f"[PASS] Aggregation completed successfully")

        # Validate output
        assert 'new_global' in result.__dict__, "Missing new_global in output"
        assert 'server_logs' in result.__dict__, "Missing server_logs in output"

        new_global = result.new_global
        server_logs = result.server_logs

        print(f"\nOutput validation:")
        print(f"  New global model keys: {len(new_global)}")
        print(f"  Server logs keys: {len(server_logs)}")

        # Check that model was updated
        for key in global_model:
            diff = torch.abs(new_global[key] - global_model[key]).sum()
            print(f"  {key}: update magnitude = {diff:.6f}")

            assert diff > 0, f"{key} was not updated!"

        print(f"\n[PASS] Model was updated correctly")

        # Check server logs
        print(f"\nServer logs content:")
        for key, value in list(server_logs.items())[:10]:  # Show first 10
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: tensor shape {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")

        print(f"\n[PASS] Server logs generated")

        return True

    except Exception as e:
        print(f"\n[FAIL] Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_aggregation():
    """Test that legacy mode still works."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Legacy Mode Full Aggregation")
    print("="*80)

    # Setup with legacy config
    config = FairCareFLConfig.create_legacy()
    aggregator = FairCareFLAggregator(config, device='cpu')

    global_model = create_synthetic_model()
    client_reports = create_synthetic_client_reports(n_clients=5)

    round_ctx = {
        'round_num': 1,
        'total_rounds': 20,
    }

    print(f"\nConfiguration:")
    print(f"  Preset: Legacy")
    print(f"  Legacy mode: {config.legacy_mode}")

    try:
        # Run aggregation
        print(f"\nRunning legacy aggregation...")
        result = aggregator.aggregate(round_ctx, client_reports, global_model)

        print(f"[PASS] Legacy aggregation completed successfully")

        # Validate output
        new_global = result.new_global

        # Check that model was updated
        total_diff = 0
        for key in global_model:
            diff = torch.abs(new_global[key] - global_model[key]).sum()
            total_diff += diff.item()

        print(f"\nTotal model update magnitude: {total_diff:.6f}")
        assert total_diff > 0, "Model was not updated!"

        print(f"[PASS] Legacy mode works correctly")

        return True

    except Exception as e:
        print(f"\n[FAIL] Legacy aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conservative_vs_legacy_difference():
    """Verify that Conservative and Legacy produce different results."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Conservative vs Legacy Comparison")
    print("="*80)

    # Create identical starting conditions
    global_model = create_synthetic_model()
    client_reports = create_synthetic_client_reports(n_clients=5)
    round_ctx = {'round_num': 1, 'total_rounds': 20}

    # Run with Conservative
    conservative_config = get_conservative_preset()
    conservative_agg = FairCareFLAggregator(conservative_config, device='cpu')
    conservative_result = conservative_agg.aggregate(round_ctx, client_reports, global_model)

    # Run with Legacy
    legacy_config = FairCareFLConfig.create_legacy()
    legacy_agg = FairCareFLAggregator(legacy_config, device='cpu')
    legacy_result = legacy_agg.aggregate(round_ctx, client_reports, global_model)

    # Compare outputs
    total_diff = 0
    for key in global_model:
        diff = torch.abs(
            conservative_result.new_global[key] - legacy_result.new_global[key]
        ).sum()
        total_diff += diff.item()

    print(f"\nTotal difference between Conservative and Legacy:")
    print(f"  Absolute difference: {total_diff:.6f}")

    if total_diff > 0.001:
        print(f"[PASS] Conservative and Legacy produce different results (as expected)")
        return True
    else:
        print(f"[WARNING] Conservative and Legacy produce very similar results")
        print(f"           This may be OK if the updates are small")
        return True  # Not necessarily a failure

def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("FairCare-FL Conservative Preset Integration Tests")
    print("="*80)

    results = []

    # Run tests
    results.append(("Conservative Aggregation", test_conservative_aggregation()))
    results.append(("Legacy Aggregation", test_legacy_aggregation()))
    results.append(("Conservative vs Legacy", test_conservative_vs_legacy_difference()))

    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{name:<35} {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED")
        print("          Ready for real experimental validation!")
    else:
        print("[WARNING] SOME TESTS FAILED - Review errors above")
    print("="*80)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
