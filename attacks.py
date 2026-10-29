"""Evasion attacks on the forecaster input (measured PV history or a covariate feed).

Threat model: the adversary edits the last `window` hours of a series before it reaches the
forecaster (e.g. compromised telemetry / data historian / forecast-vendor feed), with an L_inf
budget eps (fraction of plant capacity). It does not know the future truth.

Constraint sets
  UC : eps-ball only (standard attack, as in prior TSFM robustness studies)
  PC : eps-ball  ∩  0 <= x <= U(t) (clear-sky envelope; U=0 at night)  ∩  |x_t - x_{t-1}| <= R
       i.e. the manipulated history stays inside the physics checks of a bad-data detector.

Objectives (computed on the raw model output)
  dev : maximise the mean squared deviation of all quantiles from the clean forecast
  up  : maximise the mean P50 forecast over the horizon (over-forecast -> reserve shortfall)
"""
from __future__ import annotations

import torch


def project(x, x0, edit, eps, lo=None, hi=None, R=None, left=None):
    """Project x onto the feasible set. edit: (B,T) bool, lo/hi: (B,T) bounds or None,
    R: (B,) ramp limit or None, left: (B,) value preceding x[:, 0] for the ramp constraint."""
    x = torch.where(edit, torch.clamp(x, x0 - eps, x0 + eps), x0)
    if lo is not None:
        lo_e = torch.maximum(x0 - eps, lo)
        hi_e = torch.minimum(x0 + eps, hi)
        # if the box is empty (clean value already violates the bounds) keep the clean value
        empty = lo_e > hi_e
        lo_e = torch.where(empty, x0, lo_e)
        hi_e = torch.where(empty, x0, hi_e)
        x = torch.where(edit, torch.minimum(torch.maximum(x, lo_e), hi_e), x0)
        if R is not None:
            cols = list(x.unbind(1))
            prev = left if left is not None else cols[0]
            for t in range(len(cols)):
                if t == 0 and left is None:
                    prev = cols[0]
                    continue
                c = torch.minimum(torch.maximum(cols[t], prev - R), prev + R)
                c = torch.minimum(torch.maximum(c, lo_e[:, t]), hi_e[:, t])
                cols[t] = torch.where(edit[:, t], c, cols[t])
                prev = cols[t]
            x = torch.stack(cols, 1)
    return x


def objective(q, q_clean, kind):
    if kind == "dev":
        return ((q - q_clean) ** 2).mean()
    if kind == "up":
        return q[:, 4].mean()
    if kind == "down":
        return -q[:, 4].mean()
    raise ValueError(kind)


def pgd(f, x0, edit, eps, kind="dev", steps=10, alpha_frac=0.25, cons=None, q_clean=None, seed=0):
    """Sign-gradient PGD with a random start inside the feasible set (the 'dev' loss has zero gradient
    at x0). f: x -> raw quantiles (B,9,H). cons: dict(lo,hi,R,left) or None (UC)."""
    cons = cons or {}
    if q_clean is None:
        with torch.no_grad():
            q_clean = f(x0)
    gen = torch.Generator().manual_seed(seed)
    x = random_noise(x0, edit, eps, cons, generator=gen)
    alpha = alpha_frac * eps
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        loss = objective(f(x), q_clean, kind)
        g, = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x = x + alpha * torch.sign(g) * edit
            x = project(x, x0, edit, eps, **cons)
    return x.detach()


def random_noise(x0, edit, eps, cons=None, generator=None):
    cons = cons or {}
    u = (torch.rand(x0.shape, generator=generator) * 2 - 1) * eps
    return project(x0 + u * edit, x0, edit, eps, **cons)


def violations(x, lo, hi, R, left=None, tol=1e-4):
    """Physics bad-data detector: True if any value is below lo, above hi, or ramps faster than R."""
    v = (x < lo - tol) | (x > hi + tol)
    d = torch.diff(x, dim=1).abs() > (R[:, None] + tol)
    out = v.any(1) | d.any(1)
    if left is not None:
        out = out | ((x[:, 0] - left).abs() > R + tol)
    return out
