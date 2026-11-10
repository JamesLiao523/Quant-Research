# Clustering a Multi-Asset Portfolio by Co-Movement

**Asset labels say what an ETF is. Returns say how it behaves.** This project groups 23 multi-asset ETFs and 90 US stocks by how they move together, and tests whether the algorithm that picks the number of groups can be trusted.

## The short answer

The grouping works; the automatic count does not. With the market factor removed, the ETFs fall into eight groups a portfolio manager would recognise:

| Group | Members |
|---|---|
| US large cap and growth | SPY · QQQ · VUG |
| US small cap and value | IWM · VTV |
| Developed ex-US equity | EFA · VGK · EWJ |
| Emerging equity | EEM · FXI |
| Treasuries, IG and EM bonds | AGG · TLT · IEF · SHY · LQD · TIP · EMB |
| High yield and real estate | HYG · VNQ |
| Precious metals | GLD · SLV |
| Oil and commodities | USO · DBC |

Before detoning, HYG, EMB and VNQ sit in one block with every equity ETF: three income funds trade as the same bet as ten equity funds.

## Four findings

1. **Detoning reveals the structure.** Removing the market factor takes the ETFs from 4 groups to 8 and lifts agreement with asset classes from 0.30 to 0.63 (adjusted Rand index).
2. **The automatic count fails when groups differ in strength.** On simulated data with six planted groups of unequal strength, the optimal number of clusters (ONC) method returns 2 groups in every draw. A recursive repair lifts recovery to 0.67–0.95, still behind k-means or Ward with the best mean silhouette (0.87–1.00).
3. **Stocks expose the same flaw.** ONC finds 2 groups among 90 stocks; k-means told there are 10 sectors does far better (0.43 against 0.13).
4. **The count is not stable.** Across rolling 2-year windows ONC picks between 2 and 7 groups; consecutive windows agree at 0.45, against 0.84 when the count is fixed at 5.

## How it was done

```
daily prices ─► returns ─► correlation ─► distance √(½(1−ρ)) ─► remove market factor ─► ONC
                                                                                         │
            validate: simulated groups · real labels · rolling 2-year windows ◄─────────┘
```

Before any clustering, correlation was checked against mutual information on all 253 ETF pairs: they rank pairs almost identically (0.92), so correlation is a sound input for liquid assets.

## Using it in practice

Detone first, let ONC propose the groups, cross-check the count with k-means or Ward, then fix the count once it is sensible and stable, and let the data reassign members over time.

## Files

| | |
|---|---|
| [`clustering-multi-asset-portfolio.html`](clustering-multi-asset-portfolio.html) | Research note: executive summary, methodology, findings, implications |
| [`clustering_multi_asset_portfolio.ipynb`](clustering_multi_asset_portfolio.ipynb) | Executed notebook with every test and figure |
| `data_utils.py` | Price download with fallbacks (yfinance → Yahoo chart API → Stooq) and a local cache |
| `quant_tools.py` | Correlation cleaning, ONC clustering and portfolio optimisers |

**Run:** Python 3.11+ with `numpy pandas scipy scikit-learn matplotlib yfinance requests osqp`. Open the notebook from this folder; prices download on first run and are cached in `data/`.

**Data:** daily adjusted closes from Yahoo Finance, Jan 2015 – Oct 2026 (2,970 days). Method from López de Prado (2020), *Machine Learning for Asset Managers*.
