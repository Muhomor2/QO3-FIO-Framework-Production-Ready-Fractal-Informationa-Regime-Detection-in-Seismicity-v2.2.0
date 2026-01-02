#!/usr/bin/env python3
"""
QO3/FIO Framework - Pytest Test Suite
======================================

Run with: pytest tests/test_core.py -v
"""

import sys
sys.path.insert(0, '..')

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.qo3_fio import (
    to_utc_timestamp,
    SeismicEvent,
    Provenance,
    GardnerKnopoffDeclustering,
    FIOEstimators,
    BlockedBootstrap,
    CalibrationMetrics,
    QO3Config,
)


class TestTimestampHandling:
    """Tests for TZ-safe timestamp conversion."""
    
    def test_naive_timestamp(self):
        """Naive datetime should be localized to UTC."""
        t = to_utc_timestamp(datetime(2025, 1, 1, 12, 0, 0))
        assert t.tzinfo is not None
        assert str(t.tzinfo) == "UTC"
        assert t.hour == 12
    
    def test_aware_timestamp(self):
        """Aware timestamp should be converted to UTC."""
        t = to_utc_timestamp(pd.Timestamp("2025-01-01 12:00:00+03:00"))
        assert t.hour == 9  # 12:00 +03:00 = 09:00 UTC
    
    def test_none_timestamp(self):
        """None should return NaT."""
        t = to_utc_timestamp(None)
        assert pd.isna(t)
    
    def test_string_timestamp(self):
        """String should be parsed correctly."""
        t = to_utc_timestamp("2025-01-01T12:00:00Z")
        assert t.hour == 12


class TestDeclustering:
    """Tests for Gardner-Knopoff declustering."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        return [
            SeismicEvent(
                time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                lat=35.0, lon=139.0, depth_km=10, mag=6.0, event_id="main"
            ),
            SeismicEvent(
                time=pd.Timestamp("2024-12-31 23:59:00", tz="UTC"),
                lat=35.0, lon=139.0, depth_km=10, mag=4.0, event_id="foreshock"
            ),
            SeismicEvent(
                time=pd.Timestamp("2025-01-01 00:30:00", tz="UTC"),
                lat=35.01, lon=139.01, depth_km=10, mag=4.0, event_id="aftershock"
            ),
        ]
    
    def test_causality_preservation(self, sample_events):
        """Foreshocks should NOT be classified as aftershocks."""
        declustered = GardnerKnopoffDeclustering.decluster(
            sample_events, use_spatial_index=False
        )
        
        foreshock = next(e for e in declustered if e.event_id == "foreshock")
        aftershock = next(e for e in declustered if e.event_id == "aftershock")
        
        assert foreshock.is_mainshock == True, "Foreshock should remain mainshock"
        assert aftershock.is_mainshock == False, "Aftershock should be removed"
    
    def test_empty_catalog(self):
        """Empty catalog should return empty list."""
        result = GardnerKnopoffDeclustering.decluster([])
        assert result == []
    
    def test_window_calculation(self):
        """Window parameters should follow GK formula."""
        d, t = GardnerKnopoffDeclustering.window_params_gk74(5.0)
        
        # For M=5: log10(D) = 0.1238*5 + 0.983 = 1.602, D ≈ 40 km
        assert 30 < d < 50
        # For M=5: log10(T) = 0.5409*5 - 0.547 = 2.158, T ≈ 144 days
        assert 100 < t < 200


class TestFIOEstimators:
    """Tests for FIO statistical estimators."""
    
    @pytest.fixture
    def sample_magnitudes(self):
        """Generate sample magnitudes."""
        np.random.seed(42)
        return np.random.exponential(1, 100) + 2.5
    
    def test_b_value_finite(self, sample_magnitudes):
        """b-value should be finite for valid input."""
        b, sigma = FIOEstimators.b_value_tinti_mulargia(
            sample_magnitudes, Mc=2.5, min_n=30
        )
        
        assert np.isfinite(b)
        assert np.isfinite(sigma)
        assert 0.3 < b < 3.0, "b-value should be in reasonable range"
    
    def test_b_value_insufficient_data(self):
        """b-value should be NaN for insufficient data."""
        mags = np.array([3.0, 3.5, 4.0])  # Only 3 events
        b, sigma = FIOEstimators.b_value_tinti_mulargia(mags, Mc=2.5, min_n=30)
        
        assert np.isnan(b)
        assert np.isnan(sigma)
    
    def test_entropy_positive(self, sample_magnitudes):
        """Entropy should be positive for valid input."""
        entropy = FIOEstimators.magnitude_entropy(sample_magnitudes, bins=10)
        
        assert np.isfinite(entropy)
        assert entropy > 0
    
    def test_kl_divergence_nonnegative(self):
        """KL divergence should be non-negative."""
        counts_p = np.array([10, 20, 30, 40])
        counts_q = np.array([25, 25, 25, 25])
        
        kl = FIOEstimators.kl_divergence_discrete(counts_p, counts_q)
        
        assert np.isfinite(kl)
        assert kl >= 0
    
    def test_kl_divergence_zero_for_identical(self):
        """KL divergence should be ~0 for identical distributions."""
        counts = np.array([25, 25, 25, 25])
        
        kl = FIOEstimators.kl_divergence_discrete(counts, counts)
        
        assert abs(kl) < 1e-6


class TestBlockedBootstrap:
    """Tests for blocked bootstrap CI."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions."""
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, 200)
        s = y * 0.6 + np.random.uniform(0, 0.4, 200)
        return y, s
    
    def test_ci_contains_point_estimate(self, sample_predictions):
        """CI should contain the point estimate."""
        y, s = sample_predictions
        
        result = BlockedBootstrap.bootstrap_pr_auc(y, s, B=100, seed=42)
        
        assert np.isfinite(result["pr_auc"])
        assert result["ci_low"] <= result["pr_auc"] <= result["ci_high"]
    
    def test_block_length_heuristic(self):
        """Block length should be reasonable."""
        L = BlockedBootstrap.optimal_block_length_heuristic(n=1000, autocorr_lag1=0.5)
        
        assert 3 <= L <= 333  # Should be between minimum and n/3


class TestCalibrationMetrics:
    """Tests for calibration assessment."""
    
    def test_ece_perfect_calibration(self):
        """ECE should be ~0 for perfect calibration."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        
        ece = CalibrationMetrics.expected_calibration_error(y, p, n_bins=2)
        
        # Should be close to 0 for well-calibrated predictions
        assert ece < 0.2
    
    def test_bss_skill(self):
        """BSS should be positive for skillful model."""
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, 100)
        p_good = y * 0.7 + (1 - y) * 0.1 + np.random.uniform(-0.1, 0.1, 100)
        p_good = np.clip(p_good, 0, 1)
        
        bss = CalibrationMetrics.brier_skill_score(y, p_good)
        
        # Skillful model should have positive BSS
        assert bss > 0


class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        cfg = QO3Config()
        
        assert cfg.Mc == 2.5
        assert cfg.M_star == 5.0
        assert cfg.horizon_days == 7
        assert cfg.use_declustering == True
        assert cfg.bootstrap_B == 2000
    
    def test_custom_config(self):
        """Custom config should override defaults."""
        cfg = QO3Config(Mc=3.0, M_star=6.0, horizon_days=14)
        
        assert cfg.Mc == 3.0
        assert cfg.M_star == 6.0
        assert cfg.horizon_days == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
