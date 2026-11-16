# Factor Risk Decomposition and Stress Testing

*What is a market-neutral long/short book really betting on, and which of those bets pay?*

A long/short book with zero market exposure looks hedged, and a risk report that ranks positions by value at risk tells its manager to cut the most volatile names, which are often the ones that earn the return. Splitting this book's risk into factors told a different story. Factor bets carried 70% of its variance, and the largest was a value tilt nobody chose: long banks and energy, short technology. The intended momentum bet carried 39% of the risk while earning nothing, and a 2.2-standard-deviation momentum reversal is enough for a 10% monthly loss. Ranking stocks within sectors removed the accidental bet and cut the worst drawdown from −46% to −19%, though over 16 years no Sharpe gain clears a bootstrap interval.

## Questions

| Question | Method | Section |
|---|---|---|
| What is the book exposed to? | Nine-factor risk model; variance split by factor and by position | 3 |
| Is the book paid for those risks? | 16-year return and risk attribution by factor, stock selection and risk bucket | 4 |
| What would 2008, COVID or the 2020 momentum crash do to today's book? | Historical replays of today's book, split into factor and stock-specific loss | 5 |
| What if oil jumps 40% or rates rise 100 bp? | Conditional scenarios with normal and stress correlations | 6 |
| How bad can next month get? | Normal, bootstrap, filtered and stressed filtered historical simulation; VaR and ES | 7 |
| What would it take to lose 10%? | Reverse stress test: most plausible factor move and its probability | 8 |
| Should risk be cut, hedged, re-sized or rebuilt? | Backtest of minimum volatility, factor hedging, conviction sizing and sector-neutral rebuilds against the base book | 9 |
| What action does each risk need this month? | One-table risk report with an action for each risk source | 10 |

## Key results (Jan 2010 – Nov 2026)

- **The book loses money:** −2.5% a year at 15.8% volatility (Sharpe ratio −0.16), worst drawdown −46%.
- **Most risk is accidental:** factors explain 70% of variance. Value, an unintended sector bet (long banks and energy, short technology), carries 29%; momentum 26%.
- **The intended bets are not paid:** momentum carried 39% of variance for −0.2% a year; stock selection 34% for +0.9%. The riskiest fifth of stocks lost money and the safest fifth made it.
- **Worst days come from long banks**, and the most plausible 10% monthly loss is a momentum reversal with a market rally: a 2.2-standard-deviation month under stress correlations, about once every 5 years.
- **Removing the sector bet helps most:**

| Version | Return a year | Volatility | Sharpe | Worst drawdown | Worst episode |
|---|---|---|---|---|---|
| Base book | −2.5% | 15.8% | −0.16 | −46% | −10.8% |
| Minimum volatility | +0.5% | 11.8% | 0.04 | −38% | −9.5% |
| Hedged (keep momentum) | −1.1% | 13.3% | −0.08 | −43% | −11.2% |
| Conviction-sized | −1.3% | 14.6% | −0.09 | −40% | −7.2% |
| Sector-neutral rebuild | +0.8% | 8.4% | 0.10 | −19% | −3.5% |
| Sector-neutral and hedged | +1.1% | 8.0% | 0.14 | −22% | −4.9% |

None of the Sharpe gains clears a 90% block-bootstrap interval; the sector-neutral versions are ahead in 91–92% of resamples.

## The book

- **Universe:** 90 US large caps in 10 sectors, Jan 2007 – Oct 2026 (Yahoo Finance).
- **Signal:** half 12-1 month momentum, half stability (low residual volatility).
- **Positions:** long the top 20, short the bottom 20, inverse-volatility weights, +100% / −100% of capital.
- **Rebalance:** monthly from Jan 2010, 10 bp cost per unit traded.
- **Factors:** market, size, value, momentum, rates, credit, oil, gold and dollar, all built from tradable ETFs or the stocks themselves, so every exposure can be hedged.

## Files

| File | Contents |
|---|---|
| [`stress_testing_long_short.ipynb`](stress_testing_long_short.ipynb) | Research notebook with code, outputs and explanations |
| `risk_tools.py` | Book construction, factor risk model, VaR and ES, simulation, scenarios, reverse stress test, hedging and re-allocation |
| [`factor-risk-stress-testing.html`](factor-risk-stress-testing.html) | Research note page: summary, method, findings and risk report |
| `data_utils.py` | Price download (yfinance, then Yahoo chart API and Stooq as fallbacks) with a local cache in `data/` |

## Running

Python 3.11+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `yfinance`, `osqp`. Run the notebook from this folder. Prices download on the first run (about 110 tickers from 2007) and are cached in `data/`.

*The stock list holds today's large caps (survivorship bias), so backtest returns are flattering and some crisis losses are understated. The comparisons between risk methods are the point.*
