"""Multiple-testing corrections and data-snooping tests.

p-value adjustments: Bonferroni, Holm (1979), Benjamini-Hochberg (1995),
Benjamini-Yekutieli (2001).

Data-snooping tests on a T x K matrix ``D`` of performance differentials
d_{k,t} = f_{k,t} - f_{0,t} (strategy k minus the benchmark), all using one
shared stationary bootstrap so the dependence across strategies is kept:

* White (2000) Reality Check: H0 "no strategy beats the benchmark".
* Hansen (2005) SPA test: studentised, and less hurt by adding poor
  strategies; we report the lower, consistent and upper p-values.
* Romano and Wolf (2005) StepM: which individual strategies beat the
  benchmark, controlling the family-wise error rate.
"""
from __future__ import annotations

import numpy as np

from .bootstrap import counts, optimal_block_length


# --- p-value adjustments ---------------------------------------------------
def bonferroni(p):
    p = np.asarray(p, float)
    return np.minimum(p * len(p), 1.0)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[o] * m / np.arange(m, 0, -1))
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def benjamini_yekutieli(p):
    m = len(p)
    return np.minimum(benjamini_hochberg(p) * np.sum(1.0 / np.arange(1, m + 1)), 1.0)


ADJUST = {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg,
          "by": benjamini_yekutieli}


# --- shared bootstrap ------------------------------------------------------
class SnoopingBootstrap:
    """Bootstrap draws of the mean differentials, shared by RC/SPA/StepM."""

    def __init__(self, D: np.ndarray, reps: int = 1000, block: float | None = None,
                 seed: int = 0):
        D = np.asarray(D, float)
        if D.ndim == 1:
            D = D[:, None]
        self.D = D
        self.n, self.k = D.shape
        if block is None:
            cols = range(0, self.k, max(1, self.k // 25))
            block = float(np.median([optimal_block_length(D[:, j]) for j in cols]))
        self.block = block
        C = counts(self.n, block, reps, np.random.default_rng(seed)).astype(float)
        self.dbar = D.mean(axis=0)
        self.boot = C @ D / self.n                     # reps x k bootstrap means
        rn = np.sqrt(self.n)
        self.omega = np.maximum((rn * (self.boot - self.dbar)).std(axis=0, ddof=1), 1e-12)

    def reality_check(self) -> float:
        rn = np.sqrt(self.n)
        v = np.max(rn * self.dbar)
        v_star = np.max(rn * (self.boot - self.dbar), axis=1)
        return float((1 + np.sum(v_star >= v)) / (len(v_star) + 1))

    def spa(self) -> dict:
        rn = np.sqrt(self.n)
        t = max(float(np.max(rn * self.dbar / self.omega)), 0.0)
        thresh = -np.sqrt(self.omega ** 2 * 2 * np.log(np.log(self.n))) / rn
        out = {}
        for name, mu in (("lower", np.maximum(self.dbar, 0)),
                         ("consistent", self.dbar * (self.dbar >= thresh)),
                         ("upper", self.dbar)):
            z = rn * (self.boot - mu) / self.omega
            t_star = np.maximum(np.max(z, axis=1), 0)
            out[name] = float((1 + np.sum(t_star >= t)) / (len(t_star) + 1))
        out["stat"] = t
        return out

    def stepm(self, alpha: float = 0.05) -> dict:
        """Romano-Wolf StepM with studentised statistics.

        Returns adjusted p-values (one per strategy) and the indices of the
        strategies that beat the benchmark at level ``alpha``.
        """
        rn = np.sqrt(self.n)
        t = rn * self.dbar / self.omega
        z = rn * (self.boot - self.dbar) / self.omega   # reps x k
        order = np.argsort(-t)
        padj = np.empty(self.k)
        running = 0.0
        for j, k in enumerate(order):
            rest = order[j:]
            mx = np.max(z[:, rest], axis=1)
            p = (1 + np.sum(mx >= t[k])) / (len(mx) + 1)
            running = max(running, p)
            padj[k] = running
        return {"p_adjusted": padj, "rejected": np.where(padj <= alpha)[0].tolist(),
                "t": t}

    def single_pvalues(self, empirical: bool = False) -> np.ndarray:
        """Unadjusted one-sided p-value per strategy (H0: mean <= 0).

        By default a normal p-value from t = sqrt(n) * mean / omega, with
        omega the block-bootstrap standard error (a HAC-style z-test). This
        keeps resolution below 1 / reps, which Bonferroni and Holm need
        when there are hundreds of rules. ``empirical=True`` gives the
        bootstrap tail count instead.
        """
        if empirical:
            return (1 + np.sum(self.boot - self.dbar >= self.dbar, axis=0)) / (self.boot.shape[0] + 1)
        from scipy.stats import norm
        return norm.sf(np.sqrt(self.n) * self.dbar / self.omega)
