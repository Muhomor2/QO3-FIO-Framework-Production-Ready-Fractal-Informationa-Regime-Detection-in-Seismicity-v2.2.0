# QO3/FIO Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Fractal-Informational Regime Detection in Seismicity via Operator-Spectral Invariants**

A mathematically rigorous framework for detecting risk regimes in seismicity based on fractal-informational observables and operator-spectral formulation.

## Overview

The QO3/FIO framework identifies transitions between stochastic regimes by monitoring:

- **b-value**: Gutenberg-Richter slope with Tinti-Mulargia bias correction
- **CV**: Coefficient of variation (MAD-robust) for inter-event times
- **SID**: Seismic Information Deficit via KL-divergence

These observables are embedded into a covariance operator whose spectral properties encode regime changes.

## Key Features

### Statistical Rigor
- ✅ Causality-preserving Gardner-Knopoff declustering (foreshocks NOT removed)
- ✅ Tinti-Mulargia bias-corrected b-value estimation
- ✅ KL-divergence from counts (proper discrete measure)
- ✅ Blocked bootstrap CI for temporally correlated data
- ✅ Anti-leakage feature engineering

### Production Ready
- ✅ CLI interface with comprehensive options
- ✅ Structured logging system
- ✅ Unit test suite (pytest-compatible)
- ✅ cKDTree spatial optimization for large catalogs

### Honest Classification
All theoretical claims are classified by proof status:
- **Theorem**: Rigorously proven
- **Proposition**: Model-based with empirical evidence
- **Conjecture**: Hypothesis requiring proof

## Installation

```bash
git clone https://github.com/username/QO3-FIO-Framework.git
cd QO3-FIO-Framework
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from src.qo3_fio import QO3Config, run_pipeline, SeismicEvent, Provenance
import pandas as pd

# Create events
events = [
    SeismicEvent(
        time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        lat=35.0, lon=139.0, depth_km=10, mag=4.5,
        prov=Provenance(source="JMA")
    ),
    # ... more events
]

# Configure
cfg = QO3Config(
    Mc=2.5,           # Completeness magnitude
    M_star=5.0,       # Target magnitude
    horizon_days=7,   # Prediction horizon
    use_declustering=True,
    bootstrap_B=2000
)

# Run pipeline
results = run_pipeline("Japan/Kanto", cfg, events, series=[])

# Results
print(f"PR-AUC: {results['results']['pr_auc']:.3f}")
print(f"95% CI: {results['results']['pr_auc_ci']}")
print(f"BSS: {results['results']['brier_skill_score']:.3f}")
```

### CLI

```bash
# Run unit tests
python src/qo3_fio.py --test

# Process catalog
python src/qo3_fio.py --catalog data/events.csv --region "Japan" --output results.json

# Custom parameters
python src/qo3_fio.py --catalog data/events.csv --Mc 3.0 --M-star 5.5 --horizon 14

# Debug mode
python src/qo3_fio.py --catalog data/events.csv --log-level DEBUG --log-file qo3.log
```

## Input Format

### Seismic Catalog (CSV)
```csv
time,lat,lon,depth,mag,event_id
2025-01-01T00:00:00,35.5,139.2,10.5,4.2,ev001
2025-01-02T12:30:00,35.6,139.3,15.0,3.8,ev002
```

### Time Series (CSV, optional)
```csv
time,kind,value
2025-01-01,radon,150.5
2025-01-01,kp,3.2
```

## Key Patches (v2.2)

| Patch | Issue | Fix |
|-------|-------|-----|
| PATCH-1 | `abs(dt)` removed foreshocks | `t_test > t_main` strictly |
| PATCH-2 | Crash on naive timestamps | `tz_localize`/`tz_convert` |
| PATCH-3 | `density=True` + renorm | Counts only |
| PATCH-4 | Baseline without CI | Blocked bootstrap for all |
| PATCH-5 | "GK(1974)" as canonical | "GK-style parametric" |

## Mathematical Foundation

See `paper/main.tex` for full derivations. Key results:

### Theorem (Operator Continuity)
Under Lipschitz dynamics, eigenvalues of the covariance operator are continuous, with spectral gaps stable under perturbations.

### Proposition (Regime Tightening)
If SID↑, b↓, CV↓ over interval [t₀, t₁], then effective dimension d_eff↓.

### Conjecture (Universality)
Critical coupling satisfies κγ* = √(π/e) ≈ 1.075.

## Project Structure

```
QO3-FIO-Framework/
├── src/
│   ├── __init__.py
│   └── qo3_fio.py          # Main module
├── paper/
    ├──QO3_FIO_Framework_Paper.pdf
│   └── main.tex            # LaTeX paper
├── examples/
│   └── basic_usage.py
├── tests/
│   └── test_core.py
├── data/
│   └── sample_catalog.csv
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
└── CHANGELOG.md
```

## Related Work

- [Zenodo Record 18101985](https://zenodo.org/records/18101985) - QADMON Theory
- [Zenodo Record 18110450](https://zenodo.org/records/18110450) - FIO Extensions

## Citation

```bibtex
@software{chechelnitsky2026qo3fio,
  author       = {Chechelnitsky, Igor},
  title        = {{QO3/FIO Framework: Fractal-Informational Regime 
                   Detection in Seismicity}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.2.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/username/QO3-FIO-Framework}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Igor Chechelnitsky**  
ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)  
Independent Researcher, Ashkelon, Israel
