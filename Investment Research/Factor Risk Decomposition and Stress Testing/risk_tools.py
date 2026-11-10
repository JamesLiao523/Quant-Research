"""Risk tools for an equity long/short book: factor returns, the book, a factor risk model, VaR and expected
shortfall, risk decomposition, simulation, conditional and reverse stress tests, hedging and re-allocation."""
import numpy as np
import pandas as pd
from scipy import stats

# Factors are tradable return series built from ETFs (long leg +1, short leg -1). MOM is built from the stocks.
FACTOR_LEGS = {
    "MKT": {"SPY": 1.0},
    "SIZE": {"IWM": 1.0, "SPY": -1.0},
    "VALUE": {"VTV": 1.0, "VUG": -1.0},
    "RATES": {"IEF": 1.0},
    "CREDIT": {"HYG": 1.0, "IEF": -1.0},
    "OIL": {"USO": 1.0},
    "GOLD": {"GLD": 1.0},
    "USD": {"UUP": 1.0},
}
FACTOR_NAMES = {
    "MKT": "US market", "SIZE": "Small minus large caps", "VALUE": "Value minus growth", "MOM": "Momentum (winners minus losers)",
    "RATES": "Treasury bonds (rates down)", "CREDIT": "High yield minus Treasuries", "OIL": "Crude oil", "GOLD": "Gold",
    "USD": "US dollar",
}
FACTORS = ["MKT", "SIZE", "VALUE", "MOM", "RATES", "CREDIT", "OIL", "GOLD", "USD"]


# ---------------------------------------------------------------- signals and factors
def momentum(px, date, lookback=252, skip=21):
    """12-1 month momentum: return from `lookback` to `skip` trading days before `date`."""
    hist = px.loc[:date]
    if len(hist) <= lookback:
        return pd.Series(dtype=float)
    return (hist.iloc[-1 - skip] / hist.iloc[-1 - lookback] - 1).dropna()


def residual_vol(stock_r, mkt_r, date, window=126):
    """Annualised volatility of each stock's return after removing its market beta, over `window` days."""
    R = stock_r.loc[:date].iloc[-window:]
    m = mkt_r.loc[R.index]
    out = {}
    for c in R:
        y = R[c].dropna()
        if len(y) < 0.8 * window:
            continue
        x = m.loc[y.index]
        beta = np.cov(y, x)[0, 1] / x.var()
        out[c] = (y - beta * x).std() * np.sqrt(252)
    return pd.Series(out)


def build_factor_returns(etf_r, stock_px, stock_r, hold=21, lookback=252, skip=21):
    """Daily returns of the nine factors. MOM: equal-weight top third minus bottom third of the stocks by 12-1
    momentum, re-sorted every `hold` days."""
    F = pd.DataFrame({k: sum(w * etf_r[t] for t, w in legs.items()) for k, legs in FACTOR_LEGS.items()})
    mom = pd.Series(np.nan, index=stock_r.index)
    dates = stock_r.index
    for i in range(lookback + 1, len(dates), hold):
        m = momentum(stock_px, dates[i - 1], lookback, skip)
        if len(m) < 30:
            continue
        lo, hi = m.quantile(1 / 3), m.quantile(2 / 3)
        fwd = stock_r.iloc[i:i + hold]
        mom.iloc[i:i + hold] = (fwd[m.index[m >= hi]].mean(axis=1) - fwd[m.index[m <= lo]].mean(axis=1)).values
    F["MOM"] = mom.reindex(F.index)
    return F[FACTORS].dropna()


# ---------------------------------------------------------------- the book
def zscore(s, clip=3.0):
    return ((s - s.mean()) / s.std()).clip(-clip, clip)


def book_scores(stock_px, stock_r, mkt_r, date):
    """Composite score: half 12-1 momentum, half stability (low residual volatility). Both cross-sectional z-scores."""
    mom = momentum(stock_px, date)
    rv = residual_vol(stock_r, mkt_r, date)
    names = mom.index.intersection(rv.index)
    return (0.5 * zscore(mom[names]) + 0.5 * zscore(-rv[names])).sort_values(), mom[names], rv[names]


def build_book(stock_px, stock_r, mkt_r, date, n=20, vol_window=63):
    """Dollar-neutral book: long the top `n` scores, short the bottom `n`, each side weighted by inverse volatility
    and scaled to +100% and -100% of capital (200% gross)."""
    score, _, _ = book_scores(stock_px, stock_r, mkt_r, date)
    longs, shorts = score.index[-n:], score.index[:n]
    vol = stock_r.loc[:date].iloc[-vol_window:].std()
    w = pd.Series(0.0, index=score.index)
    w[longs] = (1 / vol[longs]) / (1 / vol[longs]).sum()
    w[shorts] = -(1 / vol[shorts]) / (1 / vol[shorts]).sum()
    return w[w != 0], score


# ---------------------------------------------------------------- factor risk model
def risk_model(stock_r, F, date, window=252, min_obs=150):
    """Time-series regression of each stock on the factors over the last `window` days.
    Returns betas B (stocks x factors), factor covariance Sf (daily), specific variances D (daily), alphas."""
    R = stock_r.loc[:date].iloc[-window:]
    X = F.reindex(R.index)
    ok = X.notna().all(axis=1)
    R, X = R.loc[ok], X.loc[ok]
    Xc = np.column_stack([np.ones(len(X)), X.values])
    B, D = {}, {}
    full = R.columns[R.notna().all()]
    if len(full):
        coef, *_ = np.linalg.lstsq(Xc, R[full].values, rcond=None)
        res = R[full].values - Xc @ coef
        for j, c in enumerate(full):
            B[c], D[c] = coef[1:, j], res[:, j].var(ddof=Xc.shape[1])
    for c in R.columns.difference(full):
        y = R[c].dropna()
        if len(y) < min_obs:
            continue
        Xi = Xc[R.index.get_indexer(y.index)]
        coef, *_ = np.linalg.lstsq(Xi, y.values, rcond=None)
        B[c], D[c] = coef[1:], (y.values - Xi @ coef).var(ddof=Xi.shape[1])
    B = pd.DataFrame(B, index=X.columns).T
    return B, X.cov(), pd.Series(D)


def asset_cov(B, Sf, D, names):
    b = B.loc[names].values
    return pd.DataFrame(b @ Sf.values @ b.T + np.diag(D[names].values), index=names, columns=names)


def risk_decomposition(w, B, Sf, D):
    """Ex-ante daily variance split into factor and specific parts, and by position.
    Returns exposures b, factor variance contributions, specific variance, total variance, position contributions."""
    names = w.index
    b = B.loc[names].T @ w                                   # factor exposures of the book
    fac_contrib = b * (Sf @ b)                                # variance contribution of each factor
    spec = (w ** 2 * D[names]).sum()
    total = fac_contrib.sum() + spec
    C = asset_cov(B, Sf, D, names)
    pos_contrib = w * (C @ w)                                 # Euler contribution of each position
    return b, fac_contrib, spec, total, pos_contrib


# ---------------------------------------------------------------- VaR and expected shortfall
def var_es(pnl, alpha=0.99):
    """Loss quantile (VaR) and average loss beyond it (ES), both as positive numbers."""
    pnl = np.asarray(pnl)
    q = np.quantile(pnl, 1 - alpha)
    return -q, -pnl[pnl <= q].mean()


def holdings_history(w, stock_r, F, B, start=None):
    """Apply today's weights to past daily returns. Days before a stock existed use its factor-model return."""
    R = stock_r[w.index].loc[start:] if start else stock_r[w.index]
    Fx = F.reindex(R.index)
    proxy = pd.DataFrame(Fx.values @ B.loc[w.index].T.values, index=R.index, columns=w.index)
    return R.fillna(proxy).dropna() @ w


def component_es(w, R, alpha=0.99):
    """Each position's average P&L on the book's tail days; the parts add up to the book's ES."""
    pnl = R @ w
    tail = pnl <= np.quantile(pnl, 1 - alpha)
    return -(R.loc[tail] * w).mean()


def fhs_paths(pnl, horizon=21, n_paths=20000, lam=0.94, seed=0, pool=None):
    """Filtered historical simulation: standardise daily P&L by its EWMA volatility, rescale to today's volatility,
    then add up `horizon` resampled days. `pool` limits resampling to chosen days (e.g. stress days)."""
    rng = np.random.default_rng(seed)
    x = pd.Series(pnl).dropna()
    var = x.pow(2).ewm(alpha=1 - lam, adjust=False).mean().shift(1).bfill()
    z = x / np.sqrt(var)
    sig_now = np.sqrt(var.iloc[-1] * lam + (1 - lam) * x.iloc[-1] ** 2)
    zpool = z.loc[pool] if pool is not None else z
    draws = rng.choice(zpool.values, size=(n_paths, horizon))
    return (draws * sig_now).sum(axis=1)


def bootstrap_paths(pnl, horizon=21, n_paths=20000, block=5, seed=0):
    """Moving-block bootstrap of raw daily P&L, keeping short runs of consecutive days together."""
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(pnl).dropna())
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(x) - block, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    return x[idx].sum(axis=1)


# ---------------------------------------------------------------- scenarios
def conditional_scenario(Sf, shocks):
    """Most likely move of every factor given shocks to some of them: E[f | f_s = s] = S_os S_ss^-1 s."""
    s = pd.Series(shocks, dtype=float)
    S_ss = Sf.loc[s.index, s.index].values
    out = Sf.loc[:, s.index].values @ np.linalg.solve(S_ss, s.values)
    out = pd.Series(out, index=Sf.index)
    out[s.index] = s.values
    return out


def reverse_stress(b, Sf, loss):
    """Most likely factor move that makes the book lose `loss` (fraction of capital) through factor exposures.
    For a linear book the answer is f* = -loss * Sf b / (b'Sf b); its distance in standard deviations is loss / sigma_b."""
    Sb = Sf @ b
    var_b = float(b @ Sb)
    f_star = -loss * Sb / var_b
    sigma = np.sqrt(var_b)
    return f_star, loss / sigma, stats.norm.cdf(-loss / sigma)


def stress_days(mkt_r, window=21, q=0.9):
    """Days whose trailing 21-day market volatility is in the top 10% of the sample."""
    vol = mkt_r.rolling(window).std()
    return vol.index[vol >= vol.quantile(q)]


# ---------------------------------------------------------------- re-allocation
def hedge_overlay(b, keep=("MOM",)):
    """ETF positions that cancel the book's exposure to every factor not in `keep`.
    Each factor is itself an ETF portfolio, so holding -b_k of factor k's legs moves the exposure to factor k by
    exactly -b_k and leaves the other factors unchanged. The change in factor exposure is stored in .attrs."""
    h, change = {}, {}
    for k, legs in FACTOR_LEGS.items():
        if k in keep or k not in b.index:
            continue
        change[k] = -b[k]
        for t, s in legs.items():
            h[t] = h.get(t, 0.0) - b[k] * s
    out = pd.Series(h)
    out.attrs["factor_exposure"] = pd.Series(change)
    return out


def sector_neutral_book(score, B, Sf, D, sector, per_side=2):
    """Within every sector, long the `per_side` best scores and short the `per_side` worst, inverse-volatility
    weighted, each side scaled to +/-100%. Sector bets cancel; what is left is the ranking inside each sector."""
    names = score.index.intersection(B.index)
    vol = np.sqrt(np.diag(asset_cov(B, Sf, D, names).values))
    vol = pd.Series(vol, index=names)
    w = pd.Series(0.0, index=names)
    for _, grp in score[names].groupby(sector.reindex(names)):
        if len(grp) < 2 * per_side:
            continue
        g = grp.sort_values()
        lo, hi = g.index[:per_side], g.index[-per_side:]
        w[hi] = 1 / vol[hi]
        w[lo] = -1 / vol[lo]
    w = w[w != 0]
    w[w > 0] /= w[w > 0].sum()
    w[w < 0] /= -w[w < 0].sum()
    return w


def min_vol_weights(w, C, cap=0.15):
    """Same names and sides as `w`, re-weighted to the lowest ex-ante volatility: longs sum to +1, shorts to -1,
    no position above `cap` of capital."""
    import osqp
    import scipy.sparse as sp

    names = w.index
    n = len(names)
    lng = (w > 0).values.astype(float)
    sht = (w < 0).values.astype(float)
    A = sp.vstack([sp.csc_matrix(lng[None, :]), sp.csc_matrix(sht[None, :]), sp.identity(n, format="csc")]).tocsc()
    lo = np.r_[1.0, -1.0, np.where(lng > 0, 0.0, -cap)]
    hi = np.r_[1.0, -1.0, np.where(lng > 0, cap, 0.0)]
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(2 * C.loc[names, names].values), np.zeros(n), A, lo, hi,
               eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polish=True, verbose=False)
    return pd.Series(prob.solve().x, index=names)


def risk_budget_weights(w, C, budget, n_iter=500):
    """Same names and sides as `w`, sized so each position's share of risk equals its share of `budget`
    (e.g. signal strength). Solved on |w| by cyclical coordinate descent, then each side scaled to +/-100%."""
    names = w.index
    sgn = np.sign(w.values)
    Cs = C.loc[names, names].values * np.outer(sgn, sgn) * 1e4      # daily covariance in bp^2 for numerical scale
    bud = np.asarray(budget.loc[names], dtype=float)
    bud = bud / bud.sum()
    x = np.abs(w.values).copy()
    for _ in range(n_iter):
        for i in range(len(x)):
            a = Cs[i, i]
            c = Cs[i] @ x - a * x[i]
            x[i] = (-c + np.sqrt(c * c + 4 * a * bud[i])) / (2 * a)       # solves x_i (Cx)_i = budget_i
    out = pd.Series(sgn * x, index=names)
    out[out > 0] /= out[out > 0].sum()
    out[out < 0] /= -out[out < 0].sum()
    return out


# ---------------------------------------------------------------- backtest
def run_backtest(stock_px, stock_r, etf_r, F, variants, start="2010-01-01", hold=21, cost=0.0010, n=20):
    """Monthly rebalanced book in several variants. `variants` maps a name to a function
    f(w, B, Sf, D, score) -> (stock weights, ETF weights). Returns daily net returns, weights, exposures."""
    dates = stock_r.index
    i0 = dates.searchsorted(pd.Timestamp(start))
    rets = {v: [] for v in variants}
    hist = {v: [] for v in variants}
    prev = {v: pd.Series(dtype=float) for v in variants}
    for i in range(i0, len(dates) - 1, hold):
        d = dates[i - 1]
        w0, score = build_book(stock_px, stock_r, etf_r["SPY"], d, n=n)
        B, Sf, D = risk_model(stock_r, F, d)
        w0 = w0[w0.index.intersection(B.index)]
        w0[w0 > 0] /= w0[w0 > 0].sum()
        w0[w0 < 0] /= -w0[w0 < 0].sum()
        fwd_s = stock_r.iloc[i:i + hold].fillna(0.0)
        fwd_e = etf_r.iloc[i:i + hold].fillna(0.0)
        for v, fn in variants.items():
            ws, we = fn(w0, B, Sf, D, score)
            allw = pd.concat([ws, we.add_prefix("ETF:")])
            traded = allw.sub(prev[v], fill_value=0.0).abs().sum()
            r = fwd_s[ws.index] @ ws + (fwd_e[we.index] @ we if len(we) else 0.0)
            r.iloc[0] -= cost * traded
            rets[v].append(r)
            b = B.loc[ws.index].T @ ws
            if len(we) and "factor_exposure" in we.attrs:
                b = b.add(we.attrs["factor_exposure"], fill_value=0.0)
            ex_ante = float(b @ Sf @ b) + float((ws ** 2 * D[ws.index]).sum())
            hist[v].append({"date": dates[i], "w": ws, "etf": we, "b": b, "traded": traded,
                            "vol": np.sqrt(ex_ante * 252)})
            prev[v] = allw
    return pd.DataFrame({v: pd.concat(r) for v, r in rets.items()}), hist


def factor_attribution(r, hist, F):
    """Split daily book returns into factor parts (exposure at the last rebalance x factor return) and the rest."""
    parts = []
    for j, h in enumerate(hist):
        end = hist[j + 1]["date"] if j + 1 < len(hist) else r.index[-1] + pd.Timedelta(days=1)
        idx = r.index[(r.index >= h["date"]) & (r.index < end)]
        parts.append(F.reindex(idx).fillna(0.0) * h["b"][F.columns].values)
    P = pd.concat(parts)
    P["Stock selection"] = r.loc[P.index] - P.sum(axis=1)
    return P
