# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                # Install/update dependencies
uv run pyopt                           # Run app via CLI entry point
uv run streamlit run streamlit_app.py  # Run app directly
```

When changing dependencies, keep both files in sync:
```bash
# After editing pyproject.toml:
uv lock
uv export --format requirements-txt --no-emit-project > requirements.txt
```

## Architecture

Single-file Streamlit application (`streamlit_app.py`) with a thin CLI wrapper (`pyopt_cli.py`).

**Data flow:** User sidebar inputs → vnstock API (live Vietnamese market data) → pandas price matrix → PyPortfolioOpt optimization → visualization (matplotlib/altair) + export (riskfolio-lib Excel reports)

**Three optimization strategies** are computed upfront on every run:
- Max Sharpe Ratio (`EfficientFrontier.max_sharpe`)
- Min Volatility (`EfficientFrontier.min_volatility`)
- Max Utility (`EfficientFrontier.max_quadratic_utility`)

A shared radio button (`portfolio_strategy_master`) controls which strategy is used across the Dollars Allocation, Report, and Risk Analysis tabs.

**Five tabs:**
1. Efficient Frontier & Weights — scatter plot of 5,000 random portfolios + weight tables/pie charts
2. Hierarchical Risk Parity — HRPOpt with dendrogram
3. Dollars Allocation — DiscreteAllocation converting weights to VND share counts
4. Report — Riskfolio-lib Excel export to `exports/reports/`
5. Risk Analysis — riskfolio plot_table, plot_drawdown, plot_range

**Caching:** `@st.cache_data` on `load_stock_symbols()` (max 1 entry) and `fetch_portfolio_stock_data()` (keyed by symbols + date range + interval).

## Key Conventions

- Python 3.10+ with type hints on function signatures
- Earth-tone custom theme defined in `.streamlit/config.toml`
- Prices from vnstock are in thousands; multiply by 1,000 for actual VND values before discrete allocation
- The `pyopt` console script in `pyproject.toml` points to `pyopt_cli:main`, which programmatically invokes `streamlit run`
- UV is the package manager; hatchling is the build backend
- No test suite, linter, or CI pipeline configured yet
