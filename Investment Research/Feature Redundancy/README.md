# Feature Redundancy: Why Individual Importance Can Be Misleading

> When similar features compete for credit, evaluating them as a group reveals what truly drives the model.

Three people on a project who can all do the same task: remove any one of them and the others cover, so each looks unimportant. Remove all three and the project fails. Model features behave the same way. A stock's 20-day and 60-day volatility both measure how jumpy it is, so when the model is tested without one, it leans on the other, and both look minor. This study measures how badly that misleads and tests a simple fix: group features that move together and test each group as a whole.

## Test 1 · Made-up data where the answer is known

10,000 rows, 40 features: 5 true drivers, 30 near-copies of them (each moves with its driver at correlation 0.97), and 5 columns of random noise.

| Question | Answer |
|---|---|
| How much credit does a driver lose when copies exist? | It keeps **17%**: average score 0.150 alone, 0.025 with copies |
| Do the usual scores find the drivers? | They find the noise (near 0%), but drivers don't stand out from their copies |
| Does grouping by correlation find the right groups? | Yes, **6 of 6**: each driver with exactly its own copies, noise apart |
| Does testing groups give the right ranking? | Yes: same order as the drivers scored alone, noise at 0.0% |

## Test 2 · 90 US large-cap stocks, 2015–2026

Each row is one stock on one date (every 5 trading days, 48,690 rows) with 15 price features in two families: **price trend** (returns from 1 day to 1 year, distance from the 50-day average, RSI, market return) and **volatility** (5-, 20- and 60-day volatility, the volatility ratio, market volatility).

| Over the next 10 days, can the model predict… | Score (0.5 = coin flip) | What does the work |
|---|---:|---|
| …whether the stock beats the median stock? | 0.503 | Nothing; no better than random columns |
| …whether its move is bigger than the median stock's? | **0.598** | The volatility group, 13× the trend group |

Tested one at a time, only 60-day volatility looks strong. Tested together, the five volatility features score 62% more than their single scores added up.

## How to prevent it

1. Draw the features' correlation heatmap before training; dark blocks will share credit.
2. Group features that move together, using the correlations.
3. Score each group by shuffling all its members at once on unseen data.
4. Add a few columns of random numbers; any score near theirs is zero.
5. Check the model beats a coin flip on unseen data before ranking anything.

## Files

- [`feature-redundancy.html`](feature-redundancy.html): research note with charts and plain-language explanations (download and open in a browser)
- [`feature_redundancy.ipynb`](feature_redundancy.ipynb): executed notebook with every test, figure and reading
- `data_utils.py`: price download with fallbacks (yfinance → Yahoo chart API → Stooq) and a local cache
- `quant_tools.py`: the clustering method (ONC) used to group features

## Run

Python 3.11+ with `numpy pandas scipy scikit-learn matplotlib yfinance requests`. Open the notebook from this folder; prices download on the first run.

*Data: Yahoo Finance daily adjusted closes, Jan 2015 – Oct 2026. The stock list holds today's large caps (survivorship bias), which does not affect the comparison of methods. Method: López de Prado (2020), Machine Learning for Asset Managers.*
