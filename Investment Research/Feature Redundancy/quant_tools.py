"""Estimators: random-matrix denoising and detoning, optimal number of clusters (ONC), portfolio optimisers and nested clustered optimisation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


# ---------- covariance helpers ----------
def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1, 1)


def corr2cov(corr, std):
    return corr * np.outer(std, std)


# ---------- random-matrix denoising ----------
def mp_pdf(var, q, pts=1000):
    """Marcenko-Pastur density for q = T/N and noise variance var."""
    lo, hi = var * (1 - (1 / q) ** 0.5) ** 2, var * (1 + (1 / q) ** 0.5) ** 2
    ev = np.linspace(lo, hi, pts)
    pdf = q / (2 * np.pi * var * ev) * np.sqrt((hi - ev) * (ev - lo))
    return pd.Series(pdf, index=ev)


def get_pca(mat):
    ev, evec = np.linalg.eigh(mat)
    idx = ev.argsort()[::-1]
    return ev[idx], evec[:, idx]


def _kde_pdf(obs, x=None, bandwidth=0.25):
    obs = np.asarray(obs, dtype=float)
    x = obs if x is None else np.asarray(x, dtype=float)
    z = (x[:, None] - obs[None, :]) / bandwidth
    dens = np.exp(-0.5 * z ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return pd.Series(dens, index=x)


def _mp_sse(var, ev, q, bandwidth, pts=1000):
    pdf_theory = mp_pdf(var[0], q, pts)
    pdf_emp = _kde_pdf(ev, pdf_theory.index.values, bandwidth)
    return float(np.sum((pdf_emp.values - pdf_theory.values) ** 2))


def fit_mp(ev, q, bandwidth=0.01):
    """Fit the noise variance by matching the KDE of eigenvalues to the MP curve. Returns (e_max, var)."""
    grid = np.linspace(0.02, 0.999, 50)
    sse = [_mp_sse([v], ev, q, bandwidth, pts=400) for v in grid]
    var = grid[int(np.argmin(sse))]
    return var * (1 + (1 / q) ** 0.5) ** 2, var


def fit_mp_trace(ev, q, n_iter=100, tol=1e-8):
    """Fixed point: the noise variance is the mean eigenvalue below the upper MP edge it implies.
    Keeps the trace of the matrix, so it works when a few large factors absorb much of the variance."""
    var = 1.0
    for _ in range(n_iter):
        e_max = var * (1 + (1 / q) ** 0.5) ** 2
        new = ev[ev <= e_max].mean()
        if abs(new - var) < tol:
            break
        var = new
    return var * (1 + (1 / q) ** 0.5) ** 2, var


def denoise_const_resid(ev, evec, n_facts):
    """Replace the noise eigenvalues by their average, keep signal eigenvalues."""
    ev = ev.copy()
    ev[n_facts:] = ev[n_facts:].sum() / float(len(ev) - n_facts)
    return cov2corr(evec @ np.diag(ev) @ evec.T)


def denoise_targeted(ev, evec, n_facts, alpha=0.0):
    """Shrink only the noise eigenvectors; alpha=0 removes all off-diagonal noise."""
    ev_s, vec_s = ev[:n_facts], evec[:, :n_facts]
    ev_n, vec_n = ev[n_facts:], evec[:, n_facts:]
    c_s = vec_s @ np.diag(ev_s) @ vec_s.T
    c_n = vec_n @ np.diag(ev_n) @ vec_n.T
    return c_s + alpha * c_n + (1 - alpha) * np.diag(np.diag(c_n))


def denoise_corr(corr, q, method="const", alpha=0.0, bandwidth=0.01):
    ev, evec = get_pca(corr)
    e_max, _ = fit_mp(ev, q, bandwidth)
    n_facts = int((ev > e_max).sum())
    if method == "const":
        return denoise_const_resid(ev, evec, n_facts), n_facts
    return cov2corr(denoise_targeted(ev, evec, n_facts, alpha)), n_facts


def detone_corr(corr, n_remove=1):
    ev, evec = get_pca(corr)
    market = evec[:, :n_remove] @ np.diag(ev[:n_remove]) @ evec[:, :n_remove].T
    return cov2corr(corr - market)


# ---------- optimal number of clusters ----------
def _corr_dist(corr):
    """Correlation distance sqrt((1 - rho) / 2); each row is used as the observation vector of one item."""
    return ((1 - pd.DataFrame(corr).fillna(0)) / 2.0) ** 0.5


def _quality(labels, sil):
    """Cluster quality: mean / std of the members' silhouettes (0 for singletons or zero spread)."""
    out = {}
    for c in np.unique(labels):
        s = sil[labels == c]
        out[c] = s.mean() / s.std() if len(s) > 1 and s.std() > 0 else 0.0
    return out


def _base_clusters(dist, max_k, n_init=10, seed=0):
    """Base ONC step: k-means for every k in 2..max_k and every initialisation; keep the partition
    with the highest silhouette t-statistic (mean / std over all items)."""
    best_stat, best = -np.inf, None
    for init in range(n_init):
        for k in range(2, min(max_k, len(dist) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed + init).fit(dist)
            sil = silhouette_samples(dist, km.labels_)
            stat = sil.mean() / sil.std() if sil.std() > 0 else -np.inf
            if stat > best_stat:
                best_stat, best = stat, (km.labels_.copy(), sil)
    return best


def onc(corr, max_k=None, n_init=10, seed=0, refine_single=False):
    """Optimal Number of Clusters.
    1. Base step: k-means on the correlation distance, k and initialisation chosen by the silhouette t-statistic.
    2. Clusters whose quality is below the average are re-clustered, recursively, with the same procedure.
    3. The re-clustering is kept only if the average quality of the new partition (all clusters, silhouettes
       recomputed on the full set) beats the average quality of the clusters that were redone.
    refine_single: also redo a lone weak cluster with at least 6 members (the plain rule needs two or more).
    Returns (labels, silhouettes) as Series indexed like corr."""
    corr = pd.DataFrame(corr)
    n = len(corr)
    max_k = min(max_k or n - 1, n - 1)
    dist = _corr_dist(corr)
    if n < 3:
        return pd.Series(np.zeros(n, dtype=int), index=corr.index), pd.Series(np.zeros(n), index=corr.index)
    labels, sil = _base_clusters(dist.values, max_k, n_init, seed)
    q = _quality(labels, sil)
    avg_q = np.mean(list(q.values()))
    weak = [c for c in q if q[c] < avg_q]
    n_weak = int(np.isin(labels, weak).sum())
    redo = len(weak) > 1 or (refine_single and len(weak) == 1 and n_weak >= 6)
    if redo and n_weak >= 3:
        keys = corr.index[np.isin(labels, weak)]
        sub_lab, _ = onc(corr.loc[keys, keys], max_k=min(max_k, len(keys) - 1), n_init=n_init, seed=seed,
                         refine_single=refine_single)
        new = labels.copy()
        new[np.isin(labels, weak)] = sub_lab.values + labels.max() + 1
        if len(np.unique(new)) > 1:
            new_sil = silhouette_samples(dist.values, new)
            new_avg = np.mean(list(_quality(new, new_sil).values()))
            if new_avg > np.mean([q[c] for c in weak]):
                labels, sil = new, new_sil
    labels = pd.factorize(labels)[0]
    return pd.Series(labels, index=corr.index), pd.Series(sil, index=corr.index)


def onc_flat(corr, max_k=None, n_init=10, seed=0, refine_single=False):
    """Earlier single-pass variant kept for comparison: one base step, weak clusters redone once with the
    base step only (no recursion), accepted if their own average quality improves."""
    corr = pd.DataFrame(corr)
    dist = _corr_dist(corr)
    max_k = max_k or len(corr) // 2
    labels, sil = _base_clusters(dist.values, max_k, n_init, seed)
    q = _quality(labels, sil)
    avg_q = np.mean(list(q.values()))
    weak = [c for c in q if q[c] < avg_q]
    n_weak = int(np.isin(labels, weak).sum())
    if len(weak) > 1 or (refine_single and len(weak) == 1 and n_weak >= 6):
        weak_idx = np.where(np.isin(labels, weak))[0]
        sub = corr.iloc[weak_idx, weak_idx]
        res = _base_clusters(_corr_dist(sub).values, len(sub) // 2, n_init, seed)
        if res is not None:
            sub_labels, sub_sil = res
            if np.mean(list(_quality(sub_labels, sub_sil).values())) > np.mean([q[c] for c in weak]):
                new = labels.copy()
                new[weak_idx] = sub_labels + labels.max() + 1
                labels = new
                sil = silhouette_samples(dist.values, labels)
    labels = pd.factorize(labels)[0]
    return pd.Series(labels, index=corr.index), pd.Series(sil, index=corr.index)


# ---------- allocation ----------
def min_var_weights(cov):
    inv = np.linalg.pinv(cov)
    ones = np.ones(len(cov))
    w = inv @ ones
    return w / w.sum()


def max_sharpe_weights(cov, mu):
    """Unconstrained tangency weights scaled to sum to 1. When sum(inv(cov) mu) < 0 the scaling flips the sign
    and returns the lowest-Sharpe portfolio; tangency_sign_ok() reports when that happens."""
    w = np.linalg.pinv(cov) @ mu
    return w / w.sum()


def tangency_sign_ok(cov, mu):
    """True when the sum-to-one scaling of inv(cov) mu keeps the tangency direction (positive denominator)."""
    return float(np.sum(np.linalg.pinv(cov) @ mu)) > 0


def min_var_long_only(cov):
    import osqp
    import scipy.sparse as sp

    cov = np.asarray(cov)
    n = len(cov)
    A = sp.vstack([sp.csc_matrix(np.ones((1, n))), sp.identity(n, format="csc")]).tocsc()
    lo = np.r_[1.0, np.zeros(n)]
    hi = np.r_[1.0, np.ones(n)]
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(2 * cov), np.zeros(n), A, lo, hi, eps_abs=1e-9, eps_rel=1e-9,
               max_iter=20000, polish=True, verbose=False)
    w = np.clip(prob.solve().x, 0, None)
    return w / w.sum()


def max_sharpe_long_only(cov, mu):
    """Long-only max-Sharpe as a QP: min y'Sy  s.t. mu'y = 1, y >= 0, then w = y / sum(y). Falls back to min-variance if no mu > 0."""
    import osqp
    import scipy.sparse as sp

    cov, mu = np.asarray(cov), np.asarray(mu)
    if not (mu > 0).any():
        return min_var_long_only(cov)
    n = len(cov)
    A = sp.vstack([sp.csc_matrix(mu.reshape(1, -1)), sp.identity(n, format="csc")]).tocsc()
    lo = np.r_[1.0, np.zeros(n)]
    hi = np.r_[1.0, np.full(n, np.inf)]
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(2 * cov), np.zeros(n), A, lo, hi, eps_abs=1e-9, eps_rel=1e-9,
               max_iter=20000, polish=True, verbose=False)
    y = np.clip(prob.solve().x, 0, None)
    return y / y.sum() if y.sum() > 0 else min_var_long_only(cov)


def nco_weights(cov, labels, solver=min_var_weights, mu=None):
    """Nested clustered optimisation: optimise inside each cluster, then across clusters."""
    cov = pd.DataFrame(cov)
    labels = pd.Series(np.asarray(labels), index=cov.index)
    clusters = sorted(labels.unique())
    intra = pd.DataFrame(0.0, index=cov.index, columns=clusters)
    for c in clusters:
        members = labels.index[labels == c]
        sub = cov.loc[members, members].values
        w = solver(sub) if mu is None else solver(sub, mu.loc[members].values)
        intra.loc[members, c] = w
    cov_r = intra.T.values @ cov.values @ intra.values
    mu_r = None if mu is None else intra.T.values @ mu.values
    inter = solver(cov_r) if mu is None else solver(cov_r, mu_r)
    return pd.Series(intra.values @ inter, index=cov.index)


# ---------- Windows guard ----------
def _guard_threadpoolctl():
    """scikit-learn asks threadpoolctl to cap BLAS threads before k-means runs. On Windows, when the
    environment sits in a folder with non-ASCII characters (e.g. OneDrive\\文件), threadpoolctl can read the
    OpenBLAS path back garbled and fail to load it. Skip any library it cannot load instead of crashing;
    the only effect is that BLAS thread counts are left at their defaults."""
    try:
        import threadpoolctl as tpc
    except ImportError:
        return
    ctrl = getattr(tpc, "ThreadpoolController", None)
    if ctrl is None or getattr(ctrl, "_path_guard", False) or not hasattr(ctrl, "_make_controller_from_path"):
        return
    original = ctrl._make_controller_from_path

    def safe(self, filepath):
        try:
            return original(self, filepath)
        except OSError:
            return None

    ctrl._make_controller_from_path = safe
    ctrl._path_guard = True


_guard_threadpoolctl()
