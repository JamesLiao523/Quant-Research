# Covariance Denoising for Equity Portfolios

Optimisers invert the covariance matrix, so every estimation error in it gets amplified into the weights. Random matrix theory says how much of a sample correlation matrix pure noise would produce; denoising keeps only what rises above that and flattens the rest. This project tests whether that produces lower-risk portfolios on 90 US large-cap stocks.

**Verdict: it depends on how much data the estimate has.**

## When denoising pays

Realised volatility of an unconstrained minimum-variance portfolio, rebalanced monthly, Jan 2017 – Oct 2026:

| Estimation window | Days per stock | Sample matrix | Denoised | Ledoit-Wolf |
|---|---:|---:|---:|---:|
| 6 months (126 days) | 1.4 | 21.0% | **13.4%** | **13.3%** |
| 1 year (252 days) | 2.8 | 15.2% | 15.0% | **13.7%** |
| 2 years (504 days) | 5.6 | **14.1%** | 16.4% | **13.7%** |

With little data, denoising removes noise that would otherwise drive the portfolio, and cuts turnover from 3.5 to 0.7 per rebalance. With more data the sample matrix is already reliable, and the noise cut-off starts deleting real sector structure instead.

## Three takeaways

- **The theory checks out.** Simulated pure-noise eigenvalues land on the predicted band, and on data built from clean sector blocks denoising cuts excess portfolio risk from 41% to 10% above optimal, ahead of Ledoit-Wolf shrinkage (24%).
- **The cut-off is too strict for equities.** It keeps 7 factors; out of sample, 20–30 are needed to match the sample matrix. Real stock returns have fat tails and clustered volatility, which the noise model does not describe.
- **Use smooth shrinkage as the default.** Ledoit-Wolf gives the lowest volatility at every window. Denoising earns its place for short windows and for calmer weights (lowest turnover at every window).

## Inside the study

1. Noise benchmark against simulation
2. Spectrum of the 90-stock correlation matrix and what its top factors mean (market, defensive vs cyclical, energy)
3. What denoising and detoning do to the matrix (condition number 197 → 43)
4. Known-truth Monte Carlo test
5. Rolling out-of-sample backtest, unconstrained and long-only
6. How many factors to keep
7. Weight stability and turnover
8. Short estimation windows

## Files

- [`covariance-denoising.html`](covariance-denoising.html): research note with charts (download and open in a browser)
- [`covariance_denoising.ipynb`](covariance_denoising.ipynb): executed notebook with every test, figure and reading
- `data_utils.py`: price download with fallbacks (yfinance → Yahoo chart API → Stooq) and a local cache
- `quant_tools.py`: denoising, detoning, clustering and optimisers

## Run

Python 3.11+ with `numpy pandas scipy scikit-learn matplotlib yfinance requests osqp`. Open the notebook from this folder; prices download on the first run (about 6 minutes end to end).

*Data: daily adjusted closes, Yahoo Finance, Jan 2015 – Oct 2026 (2,973 daily returns). Today's large caps, so absolute returns carry survivorship bias; estimators are compared on the same data. Method: López de Prado (2020), Machine Learning for Asset Managers.*
