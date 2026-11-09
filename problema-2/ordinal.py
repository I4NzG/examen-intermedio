"""Modelo logístico ordinal acumulativo con gradiente analítico."""

from __future__ import annotations

import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.special import expit


class OrdinalLogit:
    def __init__(self, l2: float = 0.0, max_iter: int = 2000):
        self.l2 = l2
        self.max_iter = max_iter

    @staticmethod
    def _unpack(params: np.ndarray, d: int, k: int):
        w = params[:d]
        raw = params[d:]
        thresholds = np.empty(k - 1)
        thresholds[0] = raw[0]
        if k > 2:
            thresholds[1:] = raw[0] + np.cumsum(np.exp(np.clip(raw[1:], -30, 30)))
        return w, raw, thresholds

    @staticmethod
    def _to_raw_gradient(g_theta: np.ndarray) -> np.ndarray:
        g_raw = np.zeros_like(g_theta)
        g_raw[0] = np.sum(g_theta)
        for j in range(1, len(g_theta)):
            gap = max(0.0, g_theta[j:].sum())
            # derivative of exp(u_j) times the sum of gradients above this gap
            # The caller sets the multiplier using the current gap below.
            g_raw[j] = gap
        return g_raw

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.arange(int(y.max()) + 1)
        n, d = X.shape
        k = len(self.classes_)
        counts = np.bincount(y, minlength=k) + 0.5
        cum = np.cumsum(counts)[:-1] / counts.sum()
        theta0 = np.log(cum / (1 - cum))
        gaps = np.maximum(np.diff(theta0), 1e-3)
        raw0 = np.r_[theta0[0], np.log(gaps)]
        x0 = np.r_[np.zeros(d), raw0]

        def objective(params):
            w, raw, theta = self._unpack(params, d, k)
            eta = X @ w
            cdf = expit(theta[None, :] - eta[:, None])
            idx = y
            upper = np.ones(n)
            valid_upper = idx < k - 1
            upper[valid_upper] = cdf[np.where(valid_upper)[0], idx[valid_upper]]
            lower = np.zeros(n)
            valid_lower = idx > 0
            lower[valid_lower] = cdf[np.where(valid_lower)[0], idx[valid_lower] - 1]
            prob = np.maximum(upper - lower, 1e-15)
            loss = -np.mean(np.log(prob)) + 0.5 * self.l2 * np.dot(w, w)

            sigp = cdf * (1 - cdf)
            gtheta = np.zeros(k - 1)
            geta = np.zeros(n)
            for cls in range(k):
                mask = y == cls
                if not np.any(mask):
                    continue
                p = prob[mask]
                if cls < k - 1:
                    up = cdf[mask, cls]
                    up_p = sigp[mask, cls]
                    geta[mask] += up_p / p
                    gtheta[cls] -= np.sum(up_p / p)
                if cls > 0:
                    lo = cdf[mask, cls - 1]
                    lo_p = sigp[mask, cls - 1]
                    geta[mask] -= lo_p / p
                    gtheta[cls - 1] += np.sum(lo_p / p)
            grad_w = (X.T @ geta) / n + self.l2 * w
            # NLL derivatives above are sums; normalize by n.
            gtheta /= n
            grad_raw = np.zeros(k - 1)
            grad_raw[0] = gtheta.sum()
            for j in range(1, k - 1):
                grad_raw[j] = np.exp(np.clip(raw[j], -30, 30)) * gtheta[j:].sum()
            return loss, np.r_[grad_w, grad_raw]

        self.result_ = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": 1e-10})
        self.coef_, _, self.thresholds_ = self._unpack(self.result_.x, d, k)
        self.objective_ = objective
        self.gradient_check_ = float(check_grad(lambda p: objective(p)[0], lambda p: objective(p)[1], self.result_.x))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        eta = np.asarray(X, dtype=float) @ self.coef_
        cdf = expit(self.thresholds_[None, :] - eta[:, None])
        probs = np.column_stack([cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1]])
        probs = np.maximum(probs, 1e-15)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

