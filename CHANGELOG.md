# Changelog

All notable changes to the QO3/FIO Framework are documented here.

## [2.2.0] - 2026-01-02

### Added
- CLI interface with argparse (`--test`, `--catalog`, `--region`, etc.)
- Structured logging system (`QO3Logger`)
- cKDTree spatial optimization for declustering (10x speedup on large catalogs)
- Unit test suite (pytest-compatible)
- Comprehensive documentation and examples
- GitHub + Zenodo integration files

### Changed
- Improved code organization and module structure
- Enhanced docstrings with references

## [2.1.0] - 2026-01-01

### Fixed
- **PATCH-1**: Declustering causality — now uses `t_test > t_main` strictly (foreshocks preserved)
- **PATCH-2**: TZ-safe timestamps — handles naive/aware timestamps correctly
- **PATCH-3**: Entropy/KL computation — uses counts instead of density
- **PATCH-4**: Baseline CI — blocked bootstrap for all methods
- **PATCH-5**: Documentation — "GK-style parametric window" terminology

## [2.0.0] - 2025-12-31

### Added
- Gardner-Knopoff style declustering
- Tinti-Mulargia bias-corrected b-value estimation
- KL-divergence based Seismic Information Deficit (SID)
- Blocked bootstrap for temporally correlated CI
- Rolling-window cross-validation
- Calibration metrics (ECE, BSS)
- LaTeX table generators

### Changed
- Complete rewrite with anti-leakage guarantees
- Strict temporal train-test splitting
- Feature matrix with provenance tracking

## [1.0.0] - 2025-12-15

### Added
- Initial implementation
- Basic b-value and CV computation
- Simple feature engineering
- GradientBoosting classifier

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)
