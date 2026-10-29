"""Shared settings and metrics for the TSFM adversarial-robustness study.

Two ERCOT case studies from the ARPA-E PERFORM data set:
  solar : hourly PV power of the existing solar sites, normalised by site AC capacity (0..1)
  load  : hourly load of the ERCOT weather zones and the system total, normalised by the zone's 2017 peak
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
for _d in (RAW_DIR, PROC_DIR, RES_DIR, FIG_DIR, WEIGHTS_DIR):
    os.makedirs(_d, exist_ok=True)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q_MED = QUANTILES.index(0.5)
Q_LO, Q_HI = QUANTILES.index(0.1), QUANTILES.index(0.9)  # 80 % central interval


@dataclass
class ExpConfig:
    context_len: int = 384          # 16 days of hourly history (multiple of 16 and 32)
    horizon: int = 24               # next local day
    issue_hour_utc: int = 6         # 00:00 CST (UTC-6) -> day-ahead issue at local midnight
    attack_window: int = 24         # attacker edits the last 24 h of the context
    eps_list: tuple = (0.02, 0.05, 0.10)  # L_inf budgets (fraction of plant capacity)
    pgd_steps: int = 20             # converged per the steps check (5/10/20/40)
    pgd_alpha_frac: float = 0.125   # step size = alpha_frac * eps = 2.5 eps / steps
    batch_size: int = 32
    seed: int = 0
    physics_target_fpr: float = 0.01  # clean false alarms of the physics check on validation days (0.5 % per rule)
    ramp_quantile: float = 0.999
    threads: int = 2
    extra: dict = field(default_factory=dict)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- metrics
# y: (N, H) truth; q: (N, Q, H) quantile forecasts; day: (N, H) bool daytime mask

def nmae(y, q, day):
    e = np.abs(q[:, Q_MED] - y)
    return float(e[day].mean())


def nrmse(y, q, day):
    e = (q[:, Q_MED] - y) ** 2
    return float(np.sqrt(e[day].mean()))


def bias(y, q, day):
    return float((q[:, Q_MED] - y)[day].mean())


def mean_quantile_loss(y, q, day):
    """Average pinball loss over the 9 quantile levels (approximates CRPS/2)."""
    qs = np.asarray(QUANTILES)[None, :, None]
    yy = y[:, None, :]
    pin = np.maximum(qs * (yy - q), (qs - 1) * (yy - q))
    pin = pin.mean(axis=1)  # over quantiles
    return float(pin[day].mean())


def picp80(y, q, day):
    inside = (y >= q[:, Q_LO]) & (y <= q[:, Q_HI])
    return float(inside[day].mean())


def piw80(q, day):
    return float((q[:, Q_HI] - q[:, Q_LO])[day].mean())


def mape(y, q, day, floor=0.05):
    """Mean absolute percentage error of P50 (meaningful for load; hours with y < floor excluded)."""
    m = day & (y > floor)
    return float((np.abs(q[:, Q_MED] - y)[m] / y[m]).mean())


def all_metrics(y, q, day):
    return dict(nMAE=nmae(y, q, day), nRMSE=nrmse(y, q, day), bias=bias(y, q, day),
                QL=mean_quantile_loss(y, q, day), PICP80=picp80(y, q, day), PIW80=piw80(q, day),
                MAPE=mape(y, q, day))


UPPER = {"solar": 1.0, "load": None}   # physical upper clip of forecasts (normalised units)
RISK_OBJECTIVE = {"solar": "up", "load": "down"}  # both push net load to be under-forecast (reserve shortfall)


def postprocess(q: torch.Tensor, upper=1.0) -> torch.Tensor:
    """Identical post-processing for every model: clip to [0, upper] and sort quantiles."""
    q = torch.clamp(q, min=0.0, max=upper)
    q, _ = torch.sort(q, dim=1)
    return q
