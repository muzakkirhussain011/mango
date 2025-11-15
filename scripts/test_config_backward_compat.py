#!/usr/bin/env python3
"""
Quick test to validate backward compatibility of FairCare-FL configuration system
"""

import sys
sys.path.insert(0, '.')

import torch
from faircare.algos.faircare_fl import FairCareFLAggregator
from faircare.config import FairCareFLConfig, get_conservative_preset, get_balanced_preset

def test_legacy_dict_config():
    """Test that old dict config still works."""
    print("\n" + "="*80)
    print("TEST 1: Legacy Dict Config (Backward Compatibility)")
    print("="*80)

    # Old-style dict config
    old_config = {
        'algorithm': 'faircare_fl',
        'some_param': 'value'
    }

    try:
        aggregator = FairCareFLAggregator(old_config, device='cpu')
        print(f"[PASS] Legacy dict config works")
        print(f"   Version: {aggregator.version}")
        print(f"   Legacy mode: {aggregator.enh_config.legacy_mode}")
        print(f"   Server momentum: {aggregator.server_momentum}")
        print(f"   Tau (temperature): {aggregator.tau}")
        print(f"   Weight floor: {aggregator.weight_floor}")
        print(f"   Weight cap: {aggregator.weight_cap}")
        return True
    except Exception as e:
        print(f"[FAIL] Legacy dict config FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_faircare_config_legacy():
    """Test new FairCareFLConfig in legacy mode."""
    print("\n" + "="*80)
    print("TEST 2: New FairCareFLConfig (Legacy Mode)")
    print("="*80)

    try:
        config = FairCareFLConfig.create_legacy()
        aggregator = FairCareFLAggregator(config, device='cpu')
        print(f"[PASS] FairCareFLConfig (legacy) works")
        print(f"   Version: {aggregator.version}")
        print(f"   Legacy mode: {aggregator.enh_config.legacy_mode}")
        print(f"   Server momentum: {aggregator.server_momentum}")
        print(f"   Tau (temperature): {aggregator.tau}")

        # Verify enhancements are disabled
        print(f"   Aggregate enhancements enabled: {aggregator.enh_config.aggregate.enable}")
        print(f"   Selection enhancements enabled: {aggregator.enh_config.selection.enable}")
        print(f"   Local fairness enabled: {aggregator.enh_config.local.enable}")
        return True
    except Exception as e:
        print(f"[FAIL] FairCareFLConfig (legacy) FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conservative_preset():
    """Test conservative preset configuration."""
    print("\n" + "="*80)
    print("TEST 3: Conservative Preset (Enhancements Enabled)")
    print("="*80)

    try:
        config = get_conservative_preset()
        aggregator = FairCareFLAggregator(config, device='cpu')
        print(f"[PASS] Conservative preset works")
        print(f"   Version: {aggregator.version}")
        print(f"   Legacy mode: {aggregator.enh_config.legacy_mode}")
        print(f"   Server momentum: {aggregator.server_momentum}")
        print(f"   Tau (temperature): {aggregator.tau}")
        print(f"   Weight floor: {aggregator.weight_floor}")
        print(f"   Weight cap: {aggregator.weight_cap}")

        # Verify conservative settings
        print(f"   Aggregate enabled: {aggregator.enh_config.aggregate.enable}")
        print(f"   Softmin temperature: {aggregator.enh_config.aggregate.softmin_temperature}")
        print(f"   Selection enabled: {aggregator.enh_config.selection.enable}")
        print(f"   Local fairness enabled: {aggregator.enh_config.local.enable}")
        return True
    except Exception as e:
        print(f"[FAIL] Conservative preset FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_balanced_preset():
    """Test balanced preset configuration."""
    print("\n" + "="*80)
    print("TEST 4: Balanced Preset (Full Features)")
    print("="*80)

    try:
        config = get_balanced_preset()
        aggregator = FairCareFLAggregator(config, device='cpu')
        print(f"[PASS] Balanced preset works")
        print(f"   Version: {aggregator.version}")
        print(f"   Legacy mode: {aggregator.enh_config.legacy_mode}")
        print(f"   Server momentum: {aggregator.server_momentum}")
        print(f"   Tau (temperature): {aggregator.tau}")

        # Verify balanced settings
        print(f"   Aggregate enabled: {aggregator.enh_config.aggregate.enable}")
        print(f"   Softmin temperature: {aggregator.enh_config.aggregate.softmin_temperature}")
        print(f"   FedNova: {aggregator.enh_config.aggregate.fednova_enable}")
        print(f"   Selection enabled: {aggregator.enh_config.selection.enable}")
        print(f"   Local fairness enabled: {aggregator.enh_config.local.enable}")
        print(f"   IRM enabled: {aggregator.enh_config.local.irm_enable}")
        print(f"   Bias policy enabled: {aggregator.enh_config.policy.enable}")
        return True
    except Exception as e:
        print(f"[FAIL] Balanced preset FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("FairCare-FL Configuration Backward Compatibility Tests")
    print("="*80)

    results = []

    # Run tests
    results.append(("Legacy Dict Config", test_legacy_dict_config()))
    results.append(("FairCareFLConfig (Legacy)", test_new_faircare_config_legacy()))
    results.append(("Conservative Preset", test_conservative_preset()))
    results.append(("Balanced Preset", test_balanced_preset()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED - Backward compatibility maintained!")
    else:
        print("[WARNING]  SOME TESTS FAILED - Review errors above")
    print("="*80)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
