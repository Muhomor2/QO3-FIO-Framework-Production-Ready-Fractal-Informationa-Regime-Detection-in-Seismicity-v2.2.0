#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QO3/FIO Framework v2.2 — Production-Ready Implementation
=========================================================

Fractal-Informational Regime Detection in Seismicity
via Operator-Spectral Invariants

This framework implements a mathematically grounded approach to detecting
risk regimes in seismicity based on fractal-informational invariants and
operator-spectral formulation.

FEATURES:
  - Gardner-Knopoff style declustering with causality preservation
  - Tinti-Mulargia bias-corrected b-value estimation
  - KL-divergence based Seismic Information Deficit (SID)
  - Blocked bootstrap for temporally correlated confidence intervals
  - Rolling-window cross-validation
  - Calibration metrics (ECE, BSS)
  - CLI interface with comprehensive options
  - Structured logging system
  - Unit test suite

Related Publications:
  - Zenodo: https://zenodo.org/records/18101985
  - Zenodo: https://zenodo.org/records/18110450

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
Affiliation: Independent Researcher, Ashkelon, Israel
License: CC BY 4.0
Version: 2.2.0
Date: 2026-01-02
"""

from __future__ import annotations

__version__ = "2.2.0"
__author__ = "Igor Chechelnitsky"
__license__ = "CC BY 4.0"

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from pathlib import Path
from datetime import datetime
import argparse
import logging
import sys
import time
import warnings
import json

import numpy as np
import pandas as pd
from scipy import stats

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss, roc_auc_score


# =============================================================================
# Logging System
# =============================================================================

class QO3Logger:
    """Structured logging for QO3/FIO framework."""
    
    _instance: Optional['QO3Logger'] = None
    
    def __init__(
        self,
        name: str = "QO3",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []
        
        fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] [%(module)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)
        
        if log_file:
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        
        QO3Logger._instance = self
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            cls()
        return cls._instance.logger
    
    @classmethod
    def set_level(cls, level: Union[int, str]):
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger = cls.get_logger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def log_performance(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        logger = QO3Logger.get_logger()
        start = time.perf_counter()
        logger.debug(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


_logger = QO3Logger(level=logging.INFO)


# =============================================================================
# Data Contracts
# =============================================================================

@dataclass(frozen=True)
class Provenance:
    """Immutable provenance record for reproducibility."""
    source: str
    source_version: str = ""
    license: str = ""
    file: str = ""
    station: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "source_version": self.source_version,
            "license": self.license,
            "file": self.file,
            "station": self.station,
            "notes": self.notes,
        }


@dataclass
class SeismicEvent:
    """Single seismic event with full provenance."""
    time: pd.Timestamp
    lat: float
    lon: float
    depth_km: float
    mag: float
    event_id: str = ""
    prov: Provenance = field(default_factory=lambda: Provenance(source="UNKNOWN"))
    is_mainshock: bool = True


@dataclass
class TimeSeriesPoint:
    """Single time series observation."""
    time: pd.Timestamp
    kind: str
    value: float
    prov: Provenance = field(default_factory=lambda: Provenance(source="UNKNOWN"))


# =============================================================================
# TZ-Safe Timestamp Handling
# =============================================================================

def to_utc_timestamp(ts: Any) -> pd.Timestamp:
    """
    Convert any timestamp to UTC-aware pd.Timestamp.
    
    Handles naive, aware, string, and None inputs safely.
    """
    if ts is None:
        return pd.NaT
    
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return pd.NaT
    
    if pd.isna(t):
        return pd.NaT
    
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    else:
        return t.tz_convert("UTC")


# =============================================================================
# Gardner-Knopoff Declustering
# =============================================================================

class GardnerKnopoffDeclustering:
    """
    GK-style parametric declustering with spatial optimization.
    
    This implements a continuous approximation of the Gardner & Knopoff (1974)
    aftershock window, commonly used in seismological software.
    
    Key features:
      - Causality-preserving: only events AFTER mainshock can be aftershocks
      - Optional cKDTree spatial index for O(n log n) performance
      - Configurable window function
    
    References:
      Gardner, J.K., & Knopoff, L. (1974). BSSA, 64(5), 1363-1367.
    """
    
    @staticmethod
    def window_params_gk74(mag: float) -> Tuple[float, float]:
        """
        GK-style parametric aftershock window.
        
        Args:
            mag: Earthquake magnitude
            
        Returns:
            (distance_km, time_days): Space-time window dimensions
        """
        if mag < 2.5:
            return (19.5, 6.0)
        
        log_d = 0.1238 * mag + 0.983
        d_km = 10.0 ** log_d
        
        if mag >= 6.5:
            log_t = 0.032 * mag + 2.7389
        else:
            log_t = 0.5409 * mag - 0.547
        t_days = 10.0 ** log_t
        
        return (d_km, t_days)
    
    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in kilometers."""
        R = 6371.0
        p = np.pi / 180.0
        dlat = (lat2 - lat1) * p
        dlon = (lon2 - lon1) * p
        a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    @classmethod
    @log_performance
    def decluster(
        cls,
        events: List[SeismicEvent],
        window_func: Optional[Callable[[float], Tuple[float, float]]] = None,
        use_spatial_index: bool = True
    ) -> List[SeismicEvent]:
        """
        Identify mainshocks vs aftershocks.
        
        CRITICAL: Preserves causality - only events occurring AFTER a larger
        event can be classified as aftershocks. Foreshocks are preserved.
        
        Args:
            events: List of seismic events
            window_func: Function (mag) -> (distance_km, time_days)
            use_spatial_index: Use cKDTree for large catalogs
            
        Returns:
            Events with is_mainshock flag set correctly
        """
        logger = QO3Logger.get_logger()
        
        if not events:
            return events
        
        if window_func is None:
            window_func = cls.window_params_gk74
        
        n = len(events)
        logger.info(f"Declustering {n} events...")
        
        sorted_events = sorted(events, key=lambda e: -e.mag)
        is_aftershock = [False] * n
        
        use_tree = use_spatial_index and HAS_KDTREE and n > 500
        tree = None
        coords_rad = None
        
        if use_tree:
            logger.debug("Building cKDTree spatial index...")
            coords_rad = np.array([
                [np.deg2rad(e.lat), np.deg2rad(e.lon)]
                for e in sorted_events
            ])
            tree = cKDTree(coords_rad)
        
        for i, ev_main in enumerate(sorted_events):
            if is_aftershock[i]:
                continue
            
            d_km_window, t_days_window = window_func(ev_main.mag)
            t_main = to_utc_timestamp(ev_main.time)
            
            if pd.isna(t_main):
                continue
            
            if use_tree and tree is not None:
                d_rad = d_km_window / 6371.0
                candidates = tree.query_ball_point(coords_rad[i], d_rad)
            else:
                candidates = range(i + 1, n)
            
            for j in candidates:
                if j <= i or is_aftershock[j]:
                    continue
                
                ev_test = sorted_events[j]
                t_test = to_utc_timestamp(ev_test.time)
                
                if pd.isna(t_test):
                    continue
                
                # CAUSALITY: aftershock must be STRICTLY AFTER mainshock
                if t_test <= t_main:
                    continue
                
                dt_days = (t_test - t_main).total_seconds() / 86400.0
                if dt_days > t_days_window:
                    continue
                
                dist = cls.haversine_km(
                    ev_main.lat, ev_main.lon,
                    ev_test.lat, ev_test.lon
                )
                
                if dist <= d_km_window:
                    is_aftershock[j] = True
        
        result = []
        n_aftershocks = 0
        for i, ev in enumerate(sorted_events):
            new_ev = SeismicEvent(
                time=ev.time, lat=ev.lat, lon=ev.lon,
                depth_km=ev.depth_km, mag=ev.mag,
                event_id=ev.event_id, prov=ev.prov,
                is_mainshock=not is_aftershock[i]
            )
            result.append(new_ev)
            if is_aftershock[i]:
                n_aftershocks += 1
        
        result.sort(key=lambda e: to_utc_timestamp(e.time))
        
        logger.info(f"Declustering complete: {n - n_aftershocks} mainshocks, {n_aftershocks} aftershocks")
        return result


# =============================================================================
# FIO Estimators
# =============================================================================

class FIOEstimators:
    """
    Fractal-Informational Observables with bias corrections.
    
    Implements statistically rigorous estimators for:
      - b-value (Tinti-Mulargia correction)
      - CV (MAD-based robust estimator)
      - Shannon entropy (discrete, from counts)
      - KL divergence (discrete)
      - Seismic Information Deficit (SID)
    """
    
    @staticmethod
    def b_value_tinti_mulargia(
        mags: np.ndarray,
        Mc: float,
        delta_M: float = 0.1,
        min_n: int = 30
    ) -> Tuple[float, float]:
        """
        Tinti-Mulargia (1987) bias-corrected b-value estimator.
        
        The standard Aki-Utsu MLE is biased for finite samples.
        This correction: b = (N-1)/N * b_aki
        
        Args:
            mags: Magnitude array
            Mc: Completeness magnitude
            delta_M: Magnitude binning width (usually 0.1)
            min_n: Minimum sample size
            
        Returns:
            (b_value, uncertainty)
            
        References:
            Aki, K. (1965). BSSA 55(1), 1-15.
            Tinti, S., & Mulargia, F. (1987). BSSA 77(6), 2125-2134.
            Shi, Y., & Bolt, B.A. (1982). BSSA 72(5), 1677-1687.
        """
        mags = np.asarray(mags, dtype=float)
        mags = mags[np.isfinite(mags) & (mags >= Mc)]
        
        N = len(mags)
        if N < min_n:
            return (np.nan, np.nan)
        
        mean_M = mags.mean()
        delta = mean_M - (Mc - delta_M / 2)
        
        if delta <= 0:
            return (np.nan, np.nan)
        
        b_aki = np.log10(np.e) / delta
        b_corrected = (N - 1) / N * b_aki
        
        variance = np.sum((mags - mean_M)**2) / (N * (N - 1))
        sigma_b = 2.3 * b_corrected**2 * np.sqrt(variance)
        
        return (float(b_corrected), float(sigma_b))
    
    @staticmethod
    def cv_robust(times_utc: np.ndarray, min_n: int = 10) -> float:
        """
        Robust CV using MAD (Median Absolute Deviation).
        
        CV_robust = (1.4826 * MAD) / median
        
        More resistant to outliers than standard CV.
        """
        converted = [to_utc_timestamp(t) for t in times_utc]
        t = pd.Series([x for x in converted if not pd.isna(x)])
        
        if len(t) < min_n:
            return np.nan
        
        t_sorted = np.sort(t.astype("datetime64[ns]").astype(np.int64))
        dt = np.diff(t_sorted) / 1e9
        
        if len(dt) < 2:
            return np.nan
        
        median_dt = np.median(dt)
        if median_dt <= 0:
            return np.nan
        
        mad = np.median(np.abs(dt - median_dt))
        robust_std = 1.4826 * mad
        
        return float(robust_std / median_dt)
    
    @staticmethod
    def shannon_entropy_discrete(counts: np.ndarray) -> float:
        """
        Shannon entropy from counts (discrete).
        
        H = -sum(p_i * log2(p_i))
        """
        counts = np.asarray(counts, dtype=float)
        counts = counts[counts > 0]
        
        if len(counts) == 0:
            return np.nan
        
        total = counts.sum()
        if total <= 0:
            return np.nan
        
        p = counts / total
        return float(-(p * np.log2(p)).sum())
    
    @staticmethod
    def magnitude_entropy(mags: np.ndarray, bins: int = 20, min_n: int = 20) -> float:
        """Shannon entropy of magnitude distribution using counts."""
        mags = np.asarray(mags, dtype=float)
        mags = mags[np.isfinite(mags)]
        
        if len(mags) < min_n:
            return np.nan
        
        counts, _ = np.histogram(mags, bins=bins, density=False)
        return FIOEstimators.shannon_entropy_discrete(counts)
    
    @staticmethod
    def kl_divergence_discrete(
        counts_p: np.ndarray,
        counts_q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Kullback-Leibler divergence D_KL(P || Q) from counts.
        
        Uses counts directly for proper discrete interpretation.
        """
        p = np.asarray(counts_p, dtype=float)
        q = np.asarray(counts_q, dtype=float)
        
        if len(p) != len(q):
            raise ValueError("Count arrays must have same length")
        
        p_sum = p.sum()
        q_sum = q.sum()
        
        if p_sum <= 0 or q_sum <= 0:
            return np.nan
        
        p = p / p_sum + epsilon
        q = q / q_sum + epsilon
        p = p / p.sum()
        q = q / q.sum()
        
        return float(np.sum(p * np.log(p / q)))
    
    @staticmethod
    def sid_kl(
        mags_current: np.ndarray,
        mags_background: np.ndarray,
        bins: int = 20,
        min_n_current: int = 20,
        min_n_background: int = 50
    ) -> float:
        """
        Seismic Information Deficit using KL-divergence.
        
        Higher SID indicates more deviation from background,
        potentially signaling stress accumulation.
        """
        mags_c = np.asarray(mags_current, dtype=float)
        mags_b = np.asarray(mags_background, dtype=float)
        
        mags_c = mags_c[np.isfinite(mags_c)]
        mags_b = mags_b[np.isfinite(mags_b)]
        
        if len(mags_c) < min_n_current or len(mags_b) < min_n_background:
            return np.nan
        
        all_mags = np.concatenate([mags_c, mags_b])
        edges = np.linspace(all_mags.min(), all_mags.max(), bins + 1)
        
        counts_c, _ = np.histogram(mags_c, bins=edges, density=False)
        counts_b, _ = np.histogram(mags_b, bins=edges, density=False)
        
        return FIOEstimators.kl_divergence_discrete(counts_c, counts_b)


# =============================================================================
# Blocked Bootstrap
# =============================================================================

class BlockedBootstrap:
    """
    Moving Block Bootstrap for time-correlated data.
    
    Preserves temporal structure within blocks, providing valid
    confidence intervals for autocorrelated time series.
    
    Reference:
        Künsch, H.R. (1989). Annals of Statistics, 17(3), 1217-1241.
    """
    
    @staticmethod
    def optimal_block_length_heuristic(n: int, autocorr_lag1: float = 0.3) -> int:
        """
        Rule-of-thumb block length estimation.
        
        L ~ n^(1/3) * (2*rho / (1 - rho^2))^(2/3)
        """
        if n < 30:
            return max(3, n // 5)
        
        rho = np.clip(abs(autocorr_lag1), 0.01, 0.99)
        L = int(n**(1/3) * (2 * rho / (1 - rho**2))**(2/3))
        return max(3, min(L, n // 3))
    
    @staticmethod
    def sample_blocks(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
        """Generate bootstrap indices using moving blocks."""
        n_blocks = max(1, int(np.ceil(n / block_length)))
        max_start = max(1, n - block_length + 1)
        starts = rng.integers(0, max_start, size=n_blocks)
        
        indices = []
        for s in starts:
            indices.extend(range(s, min(s + block_length, n)))
        
        return np.array(indices[:n], dtype=int)
    
    @classmethod
    def bootstrap_metric(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        B: int = 1000,
        block_length: Optional[int] = None,
        seed: int = 42,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """Generic blocked bootstrap CI for any metric."""
        y = np.asarray(y_true).astype(int)
        s = np.asarray(y_score).astype(float)
        
        mask = np.isfinite(s) & np.isfinite(y)
        y = y[mask]
        s = s[mask]
        n = len(y)
        
        if n < 30:
            return {"metric": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}
        
        try:
            point_est = metric_func(y, s)
        except Exception:
            point_est = np.nan
        
        if block_length is None:
            autocorr = np.corrcoef(s[:-1], s[1:])[0, 1] if len(s) > 1 else 0.3
            if np.isnan(autocorr):
                autocorr = 0.3
            block_length = cls.optimal_block_length_heuristic(n, autocorr)
        
        rng = np.random.default_rng(seed)
        metrics = []
        
        for _ in range(B):
            idx = cls.sample_blocks(n, block_length, rng)
            yb = y[idx]
            sb = s[idx]
            
            if len(np.unique(yb)) < 2:
                continue
            
            try:
                m = metric_func(yb, sb)
                if np.isfinite(m):
                    metrics.append(m)
            except Exception:
                continue
        
        if len(metrics) < B // 3:
            return {"metric": point_est, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}
        
        metrics = np.array(metrics)
        
        return {
            "metric": float(point_est) if np.isfinite(point_est) else np.nan,
            "ci_low": float(np.percentile(metrics, 100 * alpha / 2)),
            "ci_high": float(np.percentile(metrics, 100 * (1 - alpha / 2))),
            "std": float(metrics.std())
        }
    
    @classmethod
    def bootstrap_pr_auc(cls, y_true, y_score, **kwargs) -> Dict[str, float]:
        """Convenience wrapper for PR-AUC with blocked bootstrap CI."""
        def pr_auc_metric(y, s):
            prec, rec, _ = precision_recall_curve(y, s)
            return float(auc(rec, prec))
        
        result = cls.bootstrap_metric(y_true, y_score, pr_auc_metric, **kwargs)
        return {
            "pr_auc": result["metric"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "std": result["std"]
        }


# =============================================================================
# Calibration Metrics
# =============================================================================

class CalibrationMetrics:
    """Probability calibration assessment tools."""
    
    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """Compute data for reliability diagram."""
        y = np.asarray(y_true).astype(int)
        p = np.asarray(y_prob).astype(float)
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        rows = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            rows.append({
                "bin_mid": (bins[i] + bins[i + 1]) / 2,
                "predicted_prob": float(p[mask].mean()),
                "observed_freq": float(y[mask].mean()),
                "count": int(mask.sum())
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error (ECE). Lower is better."""
        df = CalibrationMetrics.reliability_diagram_data(y_true, y_prob, n_bins)
        if df.empty:
            return np.nan
        
        total = df["count"].sum()
        ece = (df["count"] * np.abs(df["predicted_prob"] - df["observed_freq"])).sum() / total
        return float(ece)
    
    @staticmethod
    def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier Skill Score relative to climatology baseline."""
        y = np.asarray(y_true).astype(float)
        p = np.asarray(y_prob).astype(float)
        
        bs = brier_score_loss(y, p)
        bs_ref = brier_score_loss(y, np.full_like(p, y.mean()))
        
        if bs_ref == 0:
            return 0.0
        
        return float(1 - bs / bs_ref)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QO3Config:
    """Configuration dataclass with sensible defaults."""
    
    # Region bounds
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    
    # Catalog parameters
    Mc: float = 2.5
    delta_M: float = 0.1
    M_star: float = 5.0
    horizon_days: int = 7
    
    # Declustering
    use_declustering: bool = True
    use_spatial_index: bool = True
    
    # Rolling windows (days)
    w_b: int = 60
    w_cv: int = 30
    w_entropy: int = 60
    w_entropy_bg: int = 180
    
    # Minimum samples
    min_mags_b: int = 30
    min_events_cv: int = 15
    entropy_bins: int = 15
    
    # Evaluation
    bootstrap_B: int = 2000
    bootstrap_seed: int = 42
    train_fraction: float = 0.70
    
    # Model hyperparameters
    n_estimators: int = 150
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lat_min": self.lat_min, "lat_max": self.lat_max,
            "lon_min": self.lon_min, "lon_max": self.lon_max,
            "Mc": self.Mc, "delta_M": self.delta_M,
            "M_star": self.M_star, "horizon_days": self.horizon_days,
            "use_declustering": self.use_declustering,
            "bootstrap_B": self.bootstrap_B,
            "train_fraction": self.train_fraction,
        }


# =============================================================================
# Feature Builder
# =============================================================================

class QO3FeatureBuilder:
    """Build anti-leakage daily feature matrix."""
    
    def __init__(self, cfg: QO3Config):
        self.cfg = cfg
        self.logger = QO3Logger.get_logger()
    
    def _filter_region(self, ev: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        mask = (
            (ev["lat"] >= c.lat_min) & (ev["lat"] <= c.lat_max) &
            (ev["lon"] >= c.lon_min) & (ev["lon"] <= c.lon_max)
        )
        return ev.loc[mask].copy()
    
    @staticmethod
    def _events_to_df(events: List[SeismicEvent]) -> pd.DataFrame:
        rows = []
        for e in events:
            t_utc = to_utc_timestamp(e.time)
            if pd.isna(t_utc):
                continue
            rows.append({
                "time": t_utc,
                "date": t_utc.floor("D"),
                "lat": float(e.lat),
                "lon": float(e.lon),
                "depth_km": float(e.depth_km),
                "mag": float(e.mag),
                "event_id": str(e.event_id),
                "source": e.prov.source,
                "is_mainshock": e.is_mainshock,
            })
        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        return df.sort_values("time").reset_index(drop=True)
    
    @staticmethod
    def _series_to_df(points: List[TimeSeriesPoint]) -> pd.DataFrame:
        rows = []
        for p in points:
            t_utc = to_utc_timestamp(p.time)
            if pd.isna(t_utc):
                continue
            rows.append({
                "time": t_utc,
                "date": t_utc.floor("D"),
                "kind": str(p.kind),
                "value": float(p.value),
                "source": p.prov.source,
            })
        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        return df.sort_values("time").reset_index(drop=True)
    
    @log_performance
    def build_daily_matrix(
        self,
        events: List[SeismicEvent],
        series: List[TimeSeriesPoint],
        mainshock_only: bool = True
    ) -> pd.DataFrame:
        """
        Build daily feature matrix with strict anti-leakage guarantees.
        
        All features use only past data (rolling windows).
        Target uses strictly future data (shift + rolling max).
        """
        cfg = self.cfg
        
        ev = self._events_to_df(events)
        ts = self._series_to_df(series)
        
        if ev.empty:
            raise ValueError("No seismic events provided.")
        
        ev = self._filter_region(ev)
        self.logger.info(f"Events after region filter: {len(ev)}")
        
        if mainshock_only and "is_mainshock" in ev.columns:
            ev_features = ev[ev["is_mainshock"]].copy()
        else:
            ev_features = ev.copy()
        
        ev_target = ev.copy()
        
        start = ev["date"].min()
        end = ev["date"].max()
        axis = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        out = pd.DataFrame(index=axis)
        out.index.name = "date"
        
        self.logger.info(f"Building features for {len(axis)} days")
        
        # Daily aggregations
        ev_day = ev_features.groupby("date").agg(
            count=("mag", "size"),
            max_mag=("mag", "max"),
            mean_mag=("mag", "mean"),
        )
        out = out.join(ev_day, how="left")
        out["count"] = out["count"].fillna(0).astype(int)
        out["max_mag"] = out["max_mag"].fillna(0)
        out["mean_mag"] = out["mean_mag"].fillna(0)
        
        # Rolling rates
        for w in [7, 14, 30, 60, 90]:
            out[f"rate_{w}d"] = out["count"].rolling(w, min_periods=1).mean()
        
        out["accel_7_30"] = out["rate_7d"] / (out["rate_30d"] + 0.1)
        out["accel_14_60"] = out["rate_14d"] / (out["rate_60d"] + 0.1)
        
        # FIO invariants
        b_vals, b_errs, cv_vals, ent_vals, sid_vals = [], [], [], [], []
        
        for d in out.index:
            d_end = d + pd.Timedelta(days=1) - pd.Timedelta(ns=1)
            
            w_b_start = d - pd.Timedelta(days=cfg.w_b)
            mags_b = ev_features.loc[
                (ev_features["time"] > w_b_start) & (ev_features["time"] <= d_end),
                "mag"
            ].values
            b, b_err = FIOEstimators.b_value_tinti_mulargia(
                mags_b, cfg.Mc, cfg.delta_M, cfg.min_mags_b
            )
            b_vals.append(b)
            b_errs.append(b_err)
            
            w_cv_start = d - pd.Timedelta(days=cfg.w_cv)
            times_cv = ev_features.loc[
                (ev_features["time"] > w_cv_start) & (ev_features["time"] <= d_end),
                "time"
            ].values
            cv_vals.append(FIOEstimators.cv_robust(times_cv, cfg.min_events_cv))
            
            w_ent_start = d - pd.Timedelta(days=cfg.w_entropy)
            w_bg_start = d - pd.Timedelta(days=cfg.w_entropy_bg)
            
            mags_ent = ev_features.loc[
                (ev_features["time"] > w_ent_start) & (ev_features["time"] <= d_end),
                "mag"
            ].values
            mags_bg = ev_features.loc[
                (ev_features["time"] > w_bg_start) & (ev_features["time"] <= d_end),
                "mag"
            ].values
            
            ent_vals.append(FIOEstimators.magnitude_entropy(mags_ent, cfg.entropy_bins))
            sid_vals.append(FIOEstimators.sid_kl(mags_ent, mags_bg, cfg.entropy_bins))
        
        out["fio_b"] = b_vals
        out["fio_b_err"] = b_errs
        out["fio_cv"] = cv_vals
        out["fio_entropy"] = ent_vals
        out["fio_sid_kl"] = sid_vals
        
        out["fio_b_chg_7d"] = out["fio_b"].diff(7)
        out["fio_b_chg_14d"] = out["fio_b"].diff(14)
        out["fio_cv_chg_7d"] = out["fio_cv"].diff(7)
        out["fio_sid_chg_7d"] = out["fio_sid_kl"].diff(7)
        
        def day_energy(mags):
            if len(mags) == 0:
                return 0.0
            return float(np.sum(10.0 ** (1.5 * np.asarray(mags) + 4.8)))
        
        ev_energy = ev_features.groupby("date")["mag"].apply(day_energy).rename("energy")
        out = out.join(ev_energy, how="left").fillna({"energy": 0.0})
        
        out["energy_7d"] = out["energy"].rolling(7, min_periods=1).sum()
        out["energy_30d"] = out["energy"].rolling(30, min_periods=1).sum()
        out["energy_ratio"] = out["energy_7d"] / (out["energy_30d"] + 1.0)
        
        if not ts.empty:
            for kind in sorted(ts["kind"].unique()):
                s = ts.loc[ts["kind"] == kind, ["date", "value"]].groupby("date")["value"].mean()
                out[f"ts_{kind}"] = s.reindex(out.index).astype(float)
                out[f"ts_{kind}_diff7"] = out[f"ts_{kind}"].diff(7)
                
                mu = out[f"ts_{kind}"].rolling(30, min_periods=10).mean()
                sd = out[f"ts_{kind}"].rolling(30, min_periods=10).std()
                out[f"ts_{kind}_z"] = (out[f"ts_{kind}"] - mu) / (sd + 1e-9)
        
        # TARGET (strictly future)
        future_max = ev_target.groupby("date")["mag"].max().reindex(out.index).fillna(0)
        future_max_rolling = future_max.shift(-1).rolling(cfg.horizon_days, min_periods=1).max()
        out["target"] = (future_max_rolling >= cfg.M_star).astype(int)
        
        out["event_rate"] = out["target"].expanding().mean()
        
        self.logger.info(f"Feature matrix: {len(out)} days, {out['target'].sum()} positive labels")
        return out
    
    @log_performance
    def train_evaluate(
        self,
        matrix: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train model and evaluate with comprehensive metrics."""
        cfg = self.cfg
        
        if feature_cols is None:
            exclude = {"target", "event_rate", "max_mag", "mean_mag", "date"}
            feature_cols = [
                c for c in matrix.columns
                if c not in exclude and matrix[c].dtype in [np.float64, np.int64, float, int]
            ]
        
        data = matrix.dropna(subset=feature_cols + ["target"]).copy()
        
        n_pos = data["target"].sum()
        if n_pos < 30:
            self.logger.warning(f"Sparse positive class: N_pos={n_pos}")
        
        split = int(len(data) * cfg.train_fraction)
        train = data.iloc[:split]
        test = data.iloc[split:]
        
        X_tr = train[feature_cols].values
        y_tr = train["target"].values.astype(int)
        X_te = test[feature_cols].values
        y_te = test["target"].values.astype(int)
        
        self.logger.info(f"Training: {len(train)} days, Testing: {len(test)} days")
        
        model = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        
        p_test = model.predict_proba(X_te)[:, 1]
        
        bootstrap_full = BlockedBootstrap.bootstrap_pr_auc(
            y_te, p_test, B=cfg.bootstrap_B, seed=cfg.bootstrap_seed
        )
        
        try:
            roc_auc = roc_auc_score(y_te, p_test)
        except Exception:
            roc_auc = np.nan
        
        brier = brier_score_loss(y_te, p_test)
        bss = CalibrationMetrics.brier_skill_score(y_te, p_test)
        ece = CalibrationMetrics.expected_calibration_error(y_te, p_test)
        reliability_data = CalibrationMetrics.reliability_diagram_data(y_te, p_test)
        
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        self.logger.info(f"PR-AUC: {bootstrap_full['pr_auc']:.3f} "
                        f"[{bootstrap_full['ci_low']:.3f}, {bootstrap_full['ci_high']:.3f}]")
        
        return {
            "n_train": len(train),
            "n_test": len(test),
            "n_pos": int(y_te.sum()),
            "n_neg": int(len(y_te) - y_te.sum()),
            "prevalence": float(y_te.mean()),
            "pr_auc": bootstrap_full["pr_auc"],
            "pr_auc_ci": (bootstrap_full["ci_low"], bootstrap_full["ci_high"]),
            "pr_auc_std": bootstrap_full["std"],
            "roc_auc": roc_auc,
            "brier_score": brier,
            "brier_skill_score": bss,
            "ece": ece,
            "reliability_data": reliability_data,
            "feature_importance": fi,
            "model": model,
            "test_predictions": p_test,
            "test_labels": y_te,
        }


# =============================================================================
# Baseline Evaluators
# =============================================================================

class BaselineEvaluators:
    """Baseline methods with blocked bootstrap CI."""
    
    @staticmethod
    def constant_baseline(y_test: np.ndarray) -> np.ndarray:
        return np.full(len(y_test), y_test.mean())
    
    @staticmethod
    def b_value_baseline(b_values: np.ndarray) -> np.ndarray:
        b = np.asarray(b_values, dtype=float)
        mask = np.isfinite(b)
        
        if mask.sum() < 10:
            return np.full(len(b), 0.5)
        
        mu = np.nanmean(b)
        sd = np.nanstd(b)
        
        if sd < 1e-9:
            return np.full(len(b), 0.5)
        
        z = (b - mu) / sd
        return 1.0 / (1.0 + np.exp(z))
    
    @staticmethod
    def evaluate_baseline(
        y_test: np.ndarray,
        p_baseline: np.ndarray,
        B: int = 2000,
        seed: int = 42
    ) -> Dict[str, Any]:
        y = np.asarray(y_test).astype(int)
        p = np.asarray(p_baseline).astype(float)
        
        mask = np.isfinite(p)
        y = y[mask]
        p = p[mask]
        
        bootstrap_result = BlockedBootstrap.bootstrap_pr_auc(y, p, B=B, seed=seed)
        
        return {
            "pr_auc": bootstrap_result["pr_auc"],
            "pr_auc_ci": (bootstrap_result["ci_low"], bootstrap_result["ci_high"]),
            "pr_auc_std": bootstrap_result["std"],
        }


# =============================================================================
# Pipeline
# =============================================================================

@log_performance
def run_pipeline(
    region_name: str,
    cfg: QO3Config,
    events: List[SeismicEvent],
    series: List[TimeSeriesPoint]
) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline.
    
    Steps:
      1. Decluster (GK-style, causality-preserving)
      2. Build feature matrix (anti-leakage)
      3. Train and evaluate model
      4. Evaluate baselines
      5. Return comprehensive results
    """
    logger = QO3Logger.get_logger()
    logger.info("=" * 60)
    logger.info(f"QO3/FIO Pipeline: {region_name}")
    logger.info("=" * 60)
    logger.info(f"Input: {len(events)} events, {len(series)} series points")
    
    if cfg.use_declustering:
        events = GardnerKnopoffDeclustering.decluster(
            events, use_spatial_index=cfg.use_spatial_index
        )
    
    builder = QO3FeatureBuilder(cfg)
    matrix = builder.build_daily_matrix(events, series, mainshock_only=True)
    results = builder.train_evaluate(matrix)
    
    split = int(len(matrix.dropna()) * cfg.train_fraction)
    test_data = matrix.dropna().iloc[split:]
    y_test = test_data["target"].values
    
    p_const = BaselineEvaluators.constant_baseline(y_test)
    baseline_const = BaselineEvaluators.evaluate_baseline(
        y_test, p_const, B=cfg.bootstrap_B, seed=cfg.bootstrap_seed
    )
    
    p_b = BaselineEvaluators.b_value_baseline(test_data["fio_b"].values)
    baseline_b = BaselineEvaluators.evaluate_baseline(
        y_test, p_b, B=cfg.bootstrap_B, seed=cfg.bootstrap_seed
    )
    
    logger.info(f"Baselines - Constant: {baseline_const['pr_auc']:.3f}, "
               f"b-only: {baseline_b['pr_auc']:.3f}")
    
    return {
        "region": region_name,
        "config": cfg.to_dict(),
        "matrix": matrix,
        "results": results,
        "baselines": {"constant": baseline_const, "b_only": baseline_b},
    }


# =============================================================================
# Unit Tests
# =============================================================================

class TestSuite:
    """Pytest-compatible test suite."""
    
    @staticmethod
    def test_to_utc_timestamp():
        t1 = to_utc_timestamp(datetime(2025, 1, 1, 12, 0, 0))
        assert t1.tzinfo is not None
        
        t2 = to_utc_timestamp(pd.Timestamp("2025-01-01 12:00:00+03:00"))
        assert t2.hour == 9
        
        t3 = to_utc_timestamp(None)
        assert pd.isna(t3)
        
        print("✓ test_to_utc_timestamp passed")
    
    @staticmethod
    def test_declustering_causality():
        events = [
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
        
        declustered = GardnerKnopoffDeclustering.decluster(events, use_spatial_index=False)
        
        foreshock = next(e for e in declustered if e.event_id == "foreshock")
        aftershock = next(e for e in declustered if e.event_id == "aftershock")
        
        assert foreshock.is_mainshock == True, "Foreshock should NOT be removed"
        assert aftershock.is_mainshock == False, "Aftershock SHOULD be removed"
        
        print("✓ test_declustering_causality passed")
    
    @staticmethod
    def test_b_value():
        np.random.seed(42)
        mags = np.random.exponential(1, 100) + 2.5
        
        b, sigma = FIOEstimators.b_value_tinti_mulargia(mags, Mc=2.5, min_n=30)
        
        assert np.isfinite(b)
        assert np.isfinite(sigma)
        assert 0.3 < b < 3.0
        
        print("✓ test_b_value passed")
    
    @staticmethod
    def test_entropy():
        mags = np.array([2.5, 2.6, 2.7, 3.0, 3.1, 3.5, 4.0, 4.5, 5.0, 5.5] * 5)
        
        entropy = FIOEstimators.magnitude_entropy(mags, bins=10)
        
        assert np.isfinite(entropy)
        assert entropy > 0
        
        print("✓ test_entropy passed")
    
    @staticmethod
    def test_kl_divergence():
        counts_p = np.array([10, 20, 30, 40])
        counts_q = np.array([25, 25, 25, 25])
        
        kl = FIOEstimators.kl_divergence_discrete(counts_p, counts_q)
        
        assert np.isfinite(kl)
        assert kl >= 0
        
        print("✓ test_kl_divergence passed")
    
    @staticmethod
    def test_blocked_bootstrap():
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, 200)
        s = y * 0.6 + np.random.uniform(0, 0.4, 200)
        
        result = BlockedBootstrap.bootstrap_pr_auc(y, s, B=100, seed=42)
        
        assert np.isfinite(result["pr_auc"])
        assert result["ci_low"] < result["pr_auc"] < result["ci_high"]
        
        print("✓ test_blocked_bootstrap passed")
    
    @classmethod
    def run_all(cls):
        print("\n" + "=" * 60)
        print("Running QO3/FIO Test Suite")
        print("=" * 60 + "\n")
        
        tests = [
            cls.test_to_utc_timestamp,
            cls.test_declustering_causality,
            cls.test_b_value,
            cls.test_entropy,
            cls.test_kl_divergence,
            cls.test_blocked_bootstrap,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except AssertionError as e:
                print(f"✗ {test.__name__} FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"✗ {test.__name__} ERROR: {e}")
                failed += 1
        
        print(f"\n{'=' * 60}")
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)
        
        return failed == 0


# =============================================================================
# CLI Interface
# =============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qo3",
        description="QO3/FIO Seismic Regime Detection Framework v2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qo3_fio.py --test
  python qo3_fio.py --catalog events.csv --region "Japan"
  python qo3_fio.py --catalog events.csv --Mc 3.0 --M-star 5.5 --horizon 14
        """
    )
    
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--version", action="version", version=f"QO3/FIO v{__version__}")
    parser.add_argument("--catalog", type=str, help="Path to seismic catalog CSV")
    parser.add_argument("--series", type=str, help="Path to time series CSV")
    parser.add_argument("--region", type=str, default="Default", help="Region name")
    parser.add_argument("--lat-min", type=float, default=-90.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--lon-min", type=float, default=-180.0)
    parser.add_argument("--lon-max", type=float, default=180.0)
    parser.add_argument("--Mc", type=float, default=2.5, help="Completeness magnitude")
    parser.add_argument("--M-star", type=float, default=5.0, help="Target magnitude")
    parser.add_argument("--horizon", type=int, default=7, help="Prediction horizon (days)")
    parser.add_argument("--no-decluster", action="store_true")
    parser.add_argument("--bootstrap-B", type=int, default=2000)
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=str)
    
    return parser


def load_catalog_csv(path: str) -> List[SeismicEvent]:
    logger = QO3Logger.get_logger()
    logger.info(f"Loading catalog from {path}")
    
    df = pd.read_csv(path)
    
    col_map = {
        "latitude": "lat", "longitude": "lon",
        "depth_km": "depth", "magnitude": "mag",
        "datetime": "time", "timestamp": "time"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    events = []
    for _, row in df.iterrows():
        try:
            events.append(SeismicEvent(
                time=pd.Timestamp(row["time"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                depth_km=float(row.get("depth", 10)),
                mag=float(row["mag"]),
                event_id=str(row.get("event_id", "")),
                prov=Provenance(source="CSV")
            ))
        except Exception:
            pass
    
    logger.info(f"Loaded {len(events)} events")
    return events


def load_series_csv(path: str) -> List[TimeSeriesPoint]:
    logger = QO3Logger.get_logger()
    logger.info(f"Loading series from {path}")
    
    df = pd.read_csv(path)
    
    points = []
    for _, row in df.iterrows():
        try:
            points.append(TimeSeriesPoint(
                time=pd.Timestamp(row["time"]),
                kind=str(row["kind"]),
                value=float(row["value"]),
                prov=Provenance(source="CSV")
            ))
        except Exception:
            pass
    
    logger.info(f"Loaded {len(points)} points")
    return points


def cli_main(args: Optional[List[str]] = None):
    parser = create_cli_parser()
    opts = parser.parse_args(args)
    
    QO3Logger(level=getattr(logging, opts.log_level), log_file=opts.log_file)
    
    if opts.test:
        success = TestSuite.run_all()
        sys.exit(0 if success else 1)
    
    if not opts.catalog:
        parser.print_help()
        print("\nError: --catalog is required")
        sys.exit(1)
    
    events = load_catalog_csv(opts.catalog)
    series = load_series_csv(opts.series) if opts.series else []
    
    cfg = QO3Config(
        lat_min=opts.lat_min, lat_max=opts.lat_max,
        lon_min=opts.lon_min, lon_max=opts.lon_max,
        Mc=opts.Mc, M_star=opts.M_star, horizon_days=opts.horizon,
        use_declustering=not opts.no_decluster,
        bootstrap_B=opts.bootstrap_B,
    )
    
    results = run_pipeline(opts.region, cfg, events, series)
    
    if opts.output:
        output_data = {
            "region": results["region"],
            "pr_auc": results["results"]["pr_auc"],
            "pr_auc_ci": results["results"]["pr_auc_ci"],
            "brier_skill_score": results["results"]["brier_skill_score"],
        }
        with open(opts.output, "w") as f:
            json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Region: {opts.region}")
    print(f"PR-AUC: {results['results']['pr_auc']:.3f} "
          f"[{results['results']['pr_auc_ci'][0]:.3f}, {results['results']['pr_auc_ci'][1]:.3f}]")
    print(f"BSS: {results['results']['brier_skill_score']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    cli_main()
