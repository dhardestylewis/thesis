# Analysis

Computational analysis, modeling, and figure generation for the zoning opposition thesis.

## Directory Structure

```
Analysis/
├── Notebooks/                 # Jupyter notebooks for exploration
├── Scripts/
│   ├── Pipeline/              # Data processing and panel construction
│   │   ├── build_panel.py           # Original panel build (Steps 1–7)
│   │   ├── rebuild_panel_v3.py      # Panel v3 with fixed parcel universe
│   │   ├── extract_centroids.py     # Centroid extraction from LUI 2024 geometry
│   │   ├── enrich_panel.py          # Census, LDB, LUI forward-fill enrichment
│   │   ├── merge_census.py          # ACS time-series merge
│   │   └── download_coa_datasets.py # Austin Open Data downloads
│   ├── Modeling/              # ML model training and evaluation
│   │   ├── backtest_naive.py          # Expanding window logistic regression
│   │   ├── backtest_generative.py     # Multi-horizon generative backtest
│   │   ├── cvae_benchmark.py          # CVAE generative model
│   │   ├── diffusion_benchmark.py     # Diffusion benchmark (v1, SMOTE-style)
│   │   └── diffusion_v2_diagnostic.py # Conditional DDPM (v2, state transitions)
│   ├── Visualization/        # Interactive HTML visualizations
│   │   ├── build_timelapse.py         # LogReg backtest timelapse map
│   │   ├── build_generative_timelapse.py  # LogReg + Diffusion timelapse
│   │   ├── build_heatmap.py           # Static protest risk heatmap
│   │   └── build_benchmark_dashboard.py   # Model comparison dashboard
│   └── Diagnostics/          # Data quality and model debugging
│       └── diagnose_shift.py          # ID mismatch and forecast diagnostics
├── Results/
│   ├── Backtests/             # Per-year backtest outputs (CSV, logs)
│   ├── Benchmarks/            # CVAE/Diffusion benchmark results
│   ├── Diffusion_v2/          # Conditional DDPM diagnostics and metrics
│   ├── Visualizations/        # Generated HTML maps and dashboards
│   ├── protest_timelapse.html         # Interactive backtest timelapse (LogReg)
│   └── generative_timelapse.html      # Interactive timelapse (LogReg + Diffusion)
└── README.md
```

## Workflow

1. **Data Pipeline** (`Scripts/Pipeline/`): Build panel from EARS, zoning cases, protest petitions
2. **Modeling** (`Scripts/Modeling/`): Train and evaluate expanding-window backtests
3. **Visualization** (`Scripts/Visualization/`): Generate interactive HTML maps and dashboards

## Models

| Model | Script | Description |
|-------|--------|-------------|
| LogReg | `backtest_naive.py` | Expanding window logistic regression with `class_weight="balanced"` |
| Diffusion v1 | `diffusion_benchmark.py` | Diffusion-as-SMOTE oversampling (deprecated) |
| Diffusion v2 | `diffusion_v2_diagnostic.py` | Conditional DDPM modeling P(state_{t+1} \| state_t) |
| CVAE | `cvae_benchmark.py` | Conditional VAE for minority generation |

## Dependencies

- `pandas`, `numpy`, `csv` — data manipulation
- `scikit-learn` — logistic regression, metrics
- `torch` — diffusion and CVAE models
- `leaflet.js`, `leaflet.heat` — interactive map visualizations (loaded via CDN)

## Related

- Data sources: `../Data/`
- Thesis writing: `../Thesis_Draft/`
- Reference materials: `../References/`
