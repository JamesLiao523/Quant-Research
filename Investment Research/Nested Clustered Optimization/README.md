# Nested Clustered Optimization

*Does splitting a portfolio into groups before optimizing protect it from estimation error?*

A portfolio optimizer treats every estimated volatility and correlation as exact, so it piles weight into assets that only look safe in the sample. Nested clustered optimization (NCO) groups assets by correlation, optimizes inside each group, then optimizes across groups, so each step uses fewer, more reliable numbers. This study measures the gain where the truth is known, then tests whether it survives in real monthly-rebalanced portfolios.

## In the lab vs in the market

| | Simulated market (true risks known) | Real long-only portfolios, 2017–2026 |
|---|---|---|
| Minimum variance | Plain: 22% riskier than necessary. **NCO: 7%** | Sharpe 0.47 → 0.49 (ETFs), 0.47 → 0.49 (ETFs + stocks) |
| Max-Sharpe | Plain: 43% of the best Sharpe. **NCO: 67%** | Sharpe 0.52 → 0.53 (ETFs), 1.09 → 1.13 (ETFs + stocks) |
| Weight changes from one extra month | NCO a third of plain (0.031 vs 0.106) | NCO trades **more**: 0.133 vs 0.049 per rebalance on 112 assets |
| Is the gain larger than noise? | Yes, in every summary | No: all six 90% intervals include zero |

**Verdict.** Grouping first is a strong cure for estimation error when the optimizer is free to take extreme positions. In long-only portfolios the no-short rule already does most of that work, so NCO adds little, and its changing groups add trading. On 22 ETFs, plain equal weight (Sharpe 0.79) beat every optimizer.

## When to reach for it

- **Yes:** objectives that use expected returns, long-short or leveraged books, many assets with short histories.
- **Test first:** long-only minimum variance, against inverse variance and hierarchical risk parity.
- **Keep groups stable:** refit them yearly, or the label changes cost more than the gain.
- **Never** run minimum variance on a universe that holds a near-cash fund.

## Inside the study

1. Three allocators agree on a case with a known answer.
2. 200 simulated samples of 40 assets in 5 groups, each estimated from 10 years of monthly data and scored against the true risks.
3. Why grouping helps: error amplification falls from 132 for the full matrix to at most 35.
4. Monthly long-only backtests on 22 ETFs and on 112 ETFs and stocks, with bootstrap intervals for each Sharpe difference.
5. Risk, concentration and turnover for every allocator.

## Files

| File | Contents |
|---|---|
| [`nested-clustered-optimization.html`](nested-clustered-optimization.html) | Research note with charts (download and open in a browser) |
| [`nested_clustered_optimization.ipynb`](nested_clustered_optimization.ipynb) | Research notebook with code, outputs and explanations |
| `data_utils.py` | Price download (Yahoo Finance, with fallbacks) and a local cache in `data/` |
| `quant_tools.py` | Denoising, clustering (ONC) and portfolio optimizers |

## Running

Python 3.11+ with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `yfinance`, `osqp`. Run the notebook from this folder; prices download on the first run and are cached in `data/`. The full run takes several minutes, because the backtests re-cluster at every rebalance.

*Data: daily adjusted closes, Jan 2015 – Oct 2026, 22 multi-asset ETFs (short-term Treasury fund SHY excluded) and 90 US large-cap stocks. Stock results carry survivorship bias and are read as comparisons between allocators. Method: López de Prado (2020), Machine Learning for Asset Managers.*
