"""
QO3/FIO Framework - Fractal-Informational Regime Detection in Seismicity
=========================================================================

A mathematically grounded framework for detecting risk regimes in seismicity
based on fractal-informational invariants and operator-spectral formulation.

Author: Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)
License: CC BY 4.0
"""

from .qo3_fio import (
    __version__,
    __author__,
    __license__,
    # Data contracts
    Provenance,
    SeismicEvent,
    TimeSeriesPoint,
    # Configuration
    QO3Config,
    # Core classes
    GardnerKnopoffDeclustering,
    FIOEstimators,
    BlockedBootstrap,
    CalibrationMetrics,
    QO3FeatureBuilder,
    BaselineEvaluators,
    # Utilities
    to_utc_timestamp,
    QO3Logger,
    # Pipeline
    run_pipeline,
    # Tests
    TestSuite,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Provenance",
    "SeismicEvent",
    "TimeSeriesPoint",
    "QO3Config",
    "GardnerKnopoffDeclustering",
    "FIOEstimators",
    "BlockedBootstrap",
    "CalibrationMetrics",
    "QO3FeatureBuilder",
    "BaselineEvaluators",
    "to_utc_timestamp",
    "QO3Logger",
    "run_pipeline",
    "TestSuite",
]
