# Implied Vol vs. Models Through an Oil Shock

*In the 2026 oil shock, which volatility forecaster was right in each phase, the option market or the models, and can a desk switch between them in time?*

Option prices saw the 2026 oil shock coming before any model did. While the shock was building, their one-week forecast error was two to nine times smaller. Then they stayed nervous long after the peak had passed. In that fading phase GARCH-X forecast better, and selling volatility paid about twice the usual premium.

A desk that always knew the phase would cut its forecast error by about 12%. In real time that gain was out of reach. All four switching blends did worse than simply trusting option prices. The warning signals did flag shocks early, but they also fired on a quarter to more than half of ordinary days.

## Key results

| Question | Finding |
|---|---|
| Best forecaster by regime (5 sessions) | Onset: implied vol (QLIKE 0.29, 2 to 9 times better than the models). Decay: GARCH-X (0.32 against 0.42 for implied vol). Calm: scaled implied vol (0.185 against 0.200). |
| Best forecaster overall | Implied vol: 0.259 at 5 sessions, 0.148 at 20 sessions. Best blend 0.291 and 0.307. |
| Value of perfect regime timing | About 12% at 5 sessions (0.227 against 0.259); none at 20 sessions, where implied vol is already best in every regime. |
| Do real-time blends capture it? | No. Every blend and the decay rule are worse than implied vol; the hard switch is significantly worse at both horizons. |
| Do signals see the regime? | Option-market signals separate onset from calm (AUC 0.89 to 0.94) and moved weeks before the 2026 onset, but are on for 23% to 62% of other days. Blend weights barely move. |
| Risk limits | One-day 99% VaR from implied vol: 4 breaches in 228 origins. In the 2026 shock VaR reacted 15 to 20 sessions after the onset began. |
| Short volatility in the decay state | +6.3 vol points per 5 sessions with the decay state on against +3.0 off, on 12 non-overlapping periods from one episode: a hypothesis for a holdout. |

## Method

- **Forecasts of next-`h`-session realized variance** (`h` = 5, 20): implied vol matched to the horizon, scaled implied vol, EWMA, HAR, HAR-RV-IV and GARCH-X.
- **Blends:** a constant weight fixed before the test, a hard switch with hysteresis, a logistic weight and an interaction regression, all refitted causally every 21 sessions.
- **Regimes:** onset, decay and calm labels built from future realized vol, used only to score forecasts.
- **Evaluation:** QLIKE and MSE on common dates, Diebold-Mariano tests with Newey-West errors, paired block-bootstrap intervals, calibration, one-day VaR backtests with Kupiec and Christoffersen tests, onset lead times and false alarms.
- **Timing analysis:** a hindsight oracle that bounds the value of regime timing, a detector scorecard (AUC, false-alarm rate, lag) for every signal, a pre-specified decay rule, and short-variance P&L by regime state.
- **Transfer:** WTI parameters frozen and applied unchanged to natural gas.

## Data

CME WTI (CL) and natural gas (NG) futures settlements and option summaries from local files. Test period 2025-06-26 to 2026-05-22 (229 origins). The test period was inspected in earlier work, so the results are exploratory and need a later holdout.

## How to run

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib pyarrow
```

Set `COMMODITY_DATA_ROOT` to the folder holding `Futures/Price/CL.parquet` and `Options/CL_summary.parquet` (and the NG files for the transfer test), then run `regime_blend.ipynb` from the project folder. The notebook imports `src/data_checks.py`, `src/features.py`, `src/blends.py`, `src/evaluation.py` and `src/var_backtest.py`. Results are written to `results/regime_blend/` and figures to `figures/`.
