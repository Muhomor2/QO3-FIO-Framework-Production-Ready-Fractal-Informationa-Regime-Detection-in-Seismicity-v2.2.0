#!/usr/bin/env python3
"""
QO3/FIO Framework - Basic Usage Example
========================================

This example demonstrates the core functionality of the framework.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from src.qo3_fio import (
    QO3Config,
    SeismicEvent,
    TimeSeriesPoint,
    Provenance,
    GardnerKnopoffDeclustering,
    FIOEstimators,
    run_pipeline,
    TestSuite,
)


def generate_synthetic_catalog(n_events: int = 500, seed: int = 42) -> list:
    """Generate synthetic seismic catalog for demonstration."""
    np.random.seed(seed)
    
    events = []
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    
    for i in range(n_events):
        # Random time progression
        dt_hours = np.random.exponential(24)  # ~1 event per day
        base_time += pd.Timedelta(hours=dt_hours)
        
        # Gutenberg-Richter magnitude distribution
        mag = 2.5 + np.random.exponential(1.0)  # b ≈ 1.0
        mag = min(mag, 7.0)  # Cap at M7
        
        # Random location (roughly Japan)
        lat = 35.0 + np.random.normal(0, 2)
        lon = 139.0 + np.random.normal(0, 2)
        depth = np.random.exponential(20)
        
        events.append(SeismicEvent(
            time=base_time,
            lat=lat,
            lon=lon,
            depth_km=depth,
            mag=mag,
            event_id=f"SYN{i:05d}",
            prov=Provenance(source="SYNTHETIC", notes="Demo data")
        ))
    
    return events


def main():
    print("=" * 60)
    print("QO3/FIO Framework - Basic Usage Example")
    print("=" * 60)
    
    # 1. Run unit tests first
    print("\n1. Running unit tests...")
    TestSuite.run_all()
    
    # 2. Generate synthetic data
    print("\n2. Generating synthetic catalog...")
    events = generate_synthetic_catalog(n_events=500)
    print(f"   Generated {len(events)} events")
    
    # 3. Demonstrate declustering
    print("\n3. Declustering...")
    declustered = GardnerKnopoffDeclustering.decluster(events, use_spatial_index=False)
    n_main = sum(1 for e in declustered if e.is_mainshock)
    n_after = len(declustered) - n_main
    print(f"   Mainshocks: {n_main}, Aftershocks: {n_after}")
    
    # 4. Demonstrate FIO estimators
    print("\n4. Computing FIO invariants...")
    mags = np.array([e.mag for e in events if e.mag >= 2.5])
    
    b, b_err = FIOEstimators.b_value_tinti_mulargia(mags, Mc=2.5)
    print(f"   b-value: {b:.3f} ± {b_err:.3f}")
    
    times = np.array([e.time for e in events])
    cv = FIOEstimators.cv_robust(times)
    print(f"   CV (robust): {cv:.3f}")
    
    entropy = FIOEstimators.magnitude_entropy(mags)
    print(f"   Entropy: {entropy:.3f} bits")
    
    # 5. Run full pipeline (if enough data)
    print("\n5. Running full pipeline...")
    cfg = QO3Config(
        Mc=2.5,
        M_star=5.0,
        horizon_days=7,
        use_declustering=True,
        bootstrap_B=500,  # Reduced for demo speed
    )
    
    try:
        results = run_pipeline("Synthetic/Demo", cfg, events, series=[])
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Region: Synthetic/Demo")
        print(f"Training days: {results['results']['n_train']}")
        print(f"Testing days: {results['results']['n_test']}")
        print(f"Positive samples: {results['results']['n_pos']}")
        print(f"PR-AUC: {results['results']['pr_auc']:.3f} "
              f"[{results['results']['pr_auc_ci'][0]:.3f}, "
              f"{results['results']['pr_auc_ci'][1]:.3f}]")
        print(f"BSS: {results['results']['brier_skill_score']:.3f}")
        print(f"ECE: {results['results']['ece']:.3f}")
        
        print("\nTop 5 features:")
        fi = results['results']['feature_importance'].head()
        for _, row in fi.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"   Pipeline failed (expected with synthetic data): {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
