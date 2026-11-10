# Pricing the WTI Smile

> One parabola in local volatility, refitted each morning, prices every WTI option strike to about a cent.

| | |
|---|---|
| **Market** | CME WTI options on futures, four listed expiries per day |
| **Sample** | May 2023 – May 2026 · 763 days · ~290,000 option quotes |
| **Model** | Quadratic normal model (QNM): volatility = level + slope × price + curvature × price² |
| **Benchmarks** | One Black (lognormal) vol, one Bachelier (normal) vol |
| **Verdict** | Use it for pricing and relative value. It does not forecast or hedge better. |

---

### The problem

A desk that quotes each strike with its own volatility runs dozens of mutually inconsistent models. That is fine for single quotes and breaks for anything that spans strikes: spreads, digitals, risk limits. The QNM replaces them with one curve described by three numbers, so every strike and payoff is priced on the same footing.

### Scoreboard

Typical pricing error, third-listed contract, cents per barrel (1¢/bbl = $10 per contract):

| Model | 2023–25 | 2026 spike | Same fit, 1 day later | Same fit, 5 days later |
|---|---:|---:|---:|---:|
| **QNM** | **1.3** | **8.7** | **18.2** | **46.0** |
| Black | 20.5 | 59.0 | 41.3 | 70.0 |
| Bachelier | 16.3 | 120.5 | 67.7 | 95.2 |

The "days later" columns cover the full sample, 2026 included.

### What the study found

- **Shape, not just level.** Error stays within ±3¢ at every strike; a single vol misses the wings by up to 24¢ (Black puts) or 50¢ (Bachelier calls).
- **Refit daily.** A fit loses most of its edge within a week.
- **The parameters are readable.** Level, slope and curvature map to the straddle, risk reversal and strangle; the slope flagged the 2026 call-skew regime weeks before curvature moved.
- **No hedging edge.** Hedging error is 0.98× Black's in 2023–25 (90% interval 0.94–1.04); the full-sample gain comes from five 2026 expiries.
- **No forecast.** Its at-the-money digital scores a Brier of 0.2529, against 0.2500 for a coin flip.
- **Known weak spot.** The closed form under-prices options beyond the 10-delta wings.

### In this folder

- [`pricing-the-wti-smile.html`](pricing-the-wti-smile.html) · research page with charts (download and open in a browser)
- [`WTI_Smile_QNM.ipynb`](WTI_Smile_QNM.ipynb) · executed notebook: data cleaning, calibration, accuracy checks, hedging and digital tests

### Run it

Python 3.10+ with `numpy pandas scipy matplotlib pyarrow`. The option data is not included; point the notebook at a parquet file of daily CL option settlements:

```bash
export CL_OPTIONS_FILE=/path/to/CL_options.parquet   # default: data/CL_options.parquet
```

<sub>Model: Bouchouev, Johnson & Sun (2025), *Volatility Trading with the Quadratic Normal Model in the Oil Options Market*; Bouchouev (2023).</sub>
