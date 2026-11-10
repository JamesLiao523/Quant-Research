# Quant Projects

Quant research projects: notebooks, research pages and write-ups.

## Investment Research

| Study | Summary |
|---|---|
| [Pricing the WTI Smile](Investment%20Research/Pricing%20the%20WTI%20Smile/) | One three-parameter model prices every WTI option strike to **1.3¢/bbl** (vs 16–24¢ for a single vol) in normal markets, 2023–25. Useful for consistent pricing and spotting mispriced strikes. No hedging or forecasting edge. |
| [Clustering a Multi-Asset Portfolio by Co-Movement](Investment%20Research/Clustering%20a%20Multi-Asset%20Portfolio%20by%20Co-Movement/) | Grouping 23 ETFs by how they move together reveals 8 behavioural groups once the market factor is removed (agreement with asset classes 0.30 → 0.63). The automatic choice of how many groups is fragile: 2 groups when strengths differ, 2 to 7 across rolling windows. |
| [Covariance Denoising for Equity Portfolios](Investment%20Research/Covariance%20Denoising%20for%20Equity%20Portfolios/) | Removing random-matrix noise from a 90-stock covariance matrix pays only when data are scarce: with 6 months of returns it cuts minimum-variance volatility from 21.0% to 13.4%; with 2 years it raises it from 14.1% to 16.4%. Ledoit-Wolf shrinkage is lowest at every window. |
| [Feature Redundancy: Why Individual Importance Can Be Misleading](Investment%20Research/Feature%20Redundancy/) | When features overlap, each looks unimportant: 30 near-copies cut 5 true drivers' measured importance to 17%. Testing correlated features as groups recovers the true drivers, and on 90 US stocks shows past prices predict how big the next move is (0.598, coin flip = 0.5), not its direction (0.503). |
| [Nested Clustered Optimization](Investment%20Research/Nested%20Clustered%20Optimization/) | Grouping assets before optimizing cuts the excess risk of minimum variance from 22% to 7% and lifts max-Sharpe from 43% to 67% of optimal in simulation. In real long-only ETF and stock portfolios (2017–2026) the gain shrinks to about 0.02 of Sharpe, within luck, and equal weight beats every optimizer on ETFs. |
| [Factor Risk Decomposition and Stress Testing](Investment%20Research/Factor%20Risk%20Decomposition%20and%20Stress%20Testing/) | A 90-stock long/short book with zero net market exposure still took 70% of its risk from factors, the largest a value tilt nobody chose. Crisis replays, scenarios, simulation and a reverse stress test show a 10% month needs only a 2.2-sd momentum reversal under crisis correlations. Ranking within sectors cut the worst drawdown from −46% to −19%. |
