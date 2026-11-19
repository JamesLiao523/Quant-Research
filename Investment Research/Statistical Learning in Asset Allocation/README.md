# Statistical Learning in Macro Portfolio Allocation

Macro-quantamental information has a plausible and documented bearing on the returns of all major asset classes. This study applies it to the allocation of capital across eleven asset classes and cash. A theory-guided signal combines 36 macro themes with signs set by economic reasoning. A learning-based signal comes from 35 regression models that sequentially select and weight 74 macro factors.

Both macro allocations have produced value against equal weights. Since October 2004 the theory-guided allocation has posted a Sharpe ratio of 0.61, against 0.53 for the learned allocation and 0.52 for equal weights, so statistical learning has delivered a slight improvement on equal weights. Both have cut the correlation with US equities from 0.77 to about 0.58. The return advantage has been concentrated before 2010, so the main lasting value of the macro tilt is a lower dependence on equity markets.

## Key results

| Question | Finding |
|---|---|
| Does the theory score predict next-quarter returns? | Weakly: r = 0.068 (p = 0.04) over 934 asset-class quarters. |
| Which allocation performs best? | Theory 0.61, then learning 0.53, a slight improvement on equal weights at 0.52 (Sharpe, Oct 2004 to Oct 2026). |
| Do the macro allocations cut equity risk? | Yes: correlation with US equities of 0.58 and 0.59, against 0.77. |
| Does the edge last? | Not fully: since mid-2010 theory ties equal weights at 0.47 and learning trails at 0.44. |
| What wealth does each build? | 6.9%, 5.8% and 5.6% a year compounded; learning edges equal weights and has the smallest drawdown, −21%. |

## Method

Country macro series become factors such as excess inflation, growth against trend, real rates, credit growth and carry. They are averaged into GDP-weighted global and regional composites and scored against their own history. The theory-guided signal combines scores into macro themes and one support score per asset class. The learning-based signal fits 35 PCA regression models per asset class every quarter on data available at the time. Softmax weights with a cash option turn either signal into a quarterly-rebalanced allocation.

## In this folder

- [Statistical learning in Macro portfolio allocation.ipynb](<Statistical learning in Macro portfolio allocation.ipynb>): the executed study.
- [statistical_learning_allocation_source.py](statistical_learning_allocation_source.py): the editable source in percent format.
- [allocation_data.py](allocation_data.py): refreshes the data snapshot, builds factors and asset-class returns.

## Data

Country series come from frozen BIS, OECD, Eurostat and ECB exports under `~/OneDrive/Investment/Market Data/Econ`. FRED, EIA and Yahoo data come from a gathered snapshot under `Econ/Gathered/<stamp>`, which the notebook refreshes itself: its first data cell compares the snapshot with [Macro Data Extract/requests.csv](../../Macro%20Data%20Extract/requests.csv) and runs the repository gatherer for anything missing, reading keys from the local `API Key.xlsx`. Use `ad.refresh_data(force=True)` for a fresh pull, or set `REFRESH_DATA = False` to skip it. Macro inputs are current revisions with approximate publication lags.

## Run

Open the notebook and run all cells. A first run downloads about 190 series (5 to 10 minutes); the model ensemble then takes 15 to 30 minutes. Focused data tests live in [Macro Data Extract/tests/test_allocation_data.py](../../Macro%20Data%20Extract/tests/test_allocation_data.py).
