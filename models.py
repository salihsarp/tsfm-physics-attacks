"""Differentiable wrappers: every forecaster maps a context tensor (B, L) in capacity units
to raw quantile forecasts (B, 9, H) for QUANTILES = 0.1..0.9. Gradients flow to the context,
so the same white-box attack code applies to every model."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from common import QUANTILES, WEIGHTS_DIR


def pick_device():
    """TSFM_DEVICE=cpu|mps|cuda overrides; otherwise Apple-GPU (mps) > cuda > cpu."""
    want = os.environ.get("TSFM_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()
torch.set_num_threads(int(os.environ.get("TSFM_THREADS", min(8, os.cpu_count() or 2))))


class OnDevice(nn.Module):
    """Runs the wrapped forecaster on `device` while callers keep CPU tensors. Autograd flows through the
    device transfers, so attacks, projections and metrics stay on the CPU."""

    def __init__(self, inner, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else DEVICE
        self.inner = inner.to(self.device).eval()
        self.name = getattr(inner, "name", type(inner).__name__)

    def forward(self, ctx, **kw):
        kw = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in kw.items()}
        return self.inner(ctx.to(self.device), **kw).to("cpu")


def _pick(model_q, wanted=QUANTILES):
    model_q = [round(float(q), 4) for q in model_q]
    return [model_q.index(round(q, 4)) for q in wanted]


# --------------------------------------------------------------------------- Chronos-Bolt
class ChronosBolt(nn.Module):
    def __init__(self, name="chronos-bolt-small", horizon=24, model=None):
        super().__init__()
        if model is None:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(os.path.join(WEIGHTS_DIR, name), device_map="cpu",
                                                       torch_dtype=torch.float32)
            model = pipe.model
        self.model = model.eval()
        self.name = name
        self.horizon = horizon
        self.idx = _pick(self.model.chronos_config.quantiles)

    def forward(self, ctx, **kw):
        out = self.model(context=ctx).quantile_preds  # (B, Q, 64)
        return out[:, self.idx, : self.horizon]


# --------------------------------------------------------------------------- Chronos-2
class Chronos2(nn.Module):
    """Univariate by default. With use_cov=True a known covariate (day-ahead NWP-based PV forecast)
    is passed in the same group: past values as a co-observed series, future values as known future."""

    def __init__(self, name="chronos-2", horizon=24, use_cov=False, model=None):
        super().__init__()
        if model is None:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(os.path.join(WEIGHTS_DIR, name), device_map="cpu",
                                                       torch_dtype=torch.float32)
            model = pipe.model
        self.model = model.eval()
        self.name = name + ("+cov" if use_cov else "")
        self.horizon = horizon
        self.use_cov = use_cov
        self.idx = _pick(self.model.chronos_config.quantiles)
        ps = self.model.chronos_config.output_patch_size
        self.n_out = int(np.ceil(horizon / ps))

    def forward(self, ctx, cov_past=None, cov_fut=None, **kw):
        B = ctx.shape[0]
        if not self.use_cov:
            out = self.model(context=ctx, num_output_patches=self.n_out).quantile_preds
            return out[:, self.idx, : self.horizon]
        context = torch.cat([ctx, cov_past], dim=0)
        gid = torch.cat([torch.arange(B, device=ctx.device), torch.arange(B, device=ctx.device)])
        fut = torch.cat([torch.full((B, self.horizon), float("nan"), device=ctx.device), cov_fut], dim=0)
        out = self.model(context=context, group_ids=gid, future_covariates=fut,
                         num_output_patches=self.n_out).quantile_preds
        return out[:B, self.idx, : self.horizon]


# --------------------------------------------------------------------------- TimesFM 2.5
class TimesFM25(nn.Module):
    """Differentiable re-implementation of TimesFM-2.5 decoding for horizon <= 128 (single prefill step)
    with the recommended flags: normalize_inputs, force_flip_invariance, continuous quantile head,
    fix_quantile_crossing, infer_is_positive."""

    def __init__(self, horizon=24, module=None):
        super().__init__()
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        if module is None:
            module = TimesFM_2p5_200M_torch_module()
            module.load_checkpoint(os.path.join(WEIGHTS_DIR, "timesfm-2.5-200m-pytorch", "model.safetensors"))
        self.m = module.eval()
        self.name = "timesfm-2.5"
        self.horizon = horizon
        assert self.m.q == 10  # [mean, q0.1 ... q0.9]

    def _prefill(self, inputs):
        from timesfm.torch.util import update_running_stats, revin
        m = self.m
        B, L = inputs.shape
        assert L % m.p == 0
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        pin = inputs.reshape(B, -1, m.p)
        pmask = masks.reshape(B, -1, m.p)
        dev = inputs.device
        n = torch.zeros(B, device=dev); mu = torch.zeros(B, device=dev); sigma = torch.zeros(B, device=dev)
        mus, sigmas = [], []
        for i in range(pin.shape[1]):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, pin[:, i], pmask[:, i])
            mus.append(mu); sigmas.append(sigma)
        cmu, csig = torch.stack(mus, 1), torch.stack(sigmas, 1)
        normed = revin(pin, cmu, csig, reverse=False)
        normed = torch.where(pmask, 0.0, normed)
        (_, _, out, spread), _ = m(normed, pmask, None)
        out = revin(out, cmu, csig, reverse=True).reshape(B, -1, m.o, m.q)[:, -1]       # (B, 128, 10)
        spread = revin(spread, cmu, csig, reverse=True).reshape(B, -1, m.os, m.q)[:, -1]  # (B, 1024, 10)
        return out, spread

    def forward(self, ctx, **kw):
        from timesfm.torch.util import revin
        H = self.horizon
        mu = ctx.mean(-1, keepdim=True)
        sd = ctx.std(-1, keepdim=True)
        x = revin(ctx, mu, sd, reverse=False)
        f, s = self._prefill(x)
        ff, fs = self._prefill(-x)
        flip = lambda z: torch.cat([z[..., :1], torch.flip(z[..., 1:], dims=(-1,))], dim=-1)
        f = (f - flip(ff)) / 2
        s = (s - flip(fs)) / 2
        f = f[:, :H]
        s = s[:, :H]
        cols = [f[..., 0]]
        for qi in range(1, 10):
            cols.append(f[..., 5] if qi == 5 else s[..., qi] - s[..., 5] + f[..., 5])
        full = torch.stack(cols, -1)
        # fix quantile crossing (same rule as the reference implementation)
        c = list(full.unbind(-1))
        for i in [4, 3, 2, 1]:
            c[i] = torch.minimum(c[i], c[i + 1])
        for i in [6, 7, 8, 9]:
            c[i] = torch.maximum(c[i], c[i - 1])
        full = torch.stack(c, -1)
        full = revin(full, mu, sd, reverse=True)
        return full[..., 1:].permute(0, 2, 1)  # (B, 9, H)


# --------------------------------------------------------------------------- LSTM baseline
class QuantileLSTM(nn.Module):
    """Global sequence-to-vector LSTM trained from scratch (pinball loss) on 2017 data."""

    def __init__(self, lookback=168, horizon=24, hidden=64, layers=2):
        super().__init__()
        self.name = "lstm"
        self.lookback, self.horizon = lookback, horizon
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 9 * horizon))

    def forward(self, ctx, **kw):
        x = ctx[:, -self.lookback:]
        hod = torch.arange(self.lookback, dtype=x.dtype, device=x.device) % 24 / 23.0
        z = torch.stack([x, hod.expand_as(x)], -1)
        h, _ = self.lstm(z)
        out = self.head(h[:, -1]).view(-1, 9, self.horizon)
        base = out[:, 4:5]
        inc = torch.nn.functional.softplus(out[:, 5:])
        dec = torch.nn.functional.softplus(out[:, :4].flip(1))
        upper = base + torch.cumsum(inc, 1)
        lower = (base - torch.cumsum(dec, 1)).flip(1)
        return torch.cat([lower, base, upper], 1)


def pinball(qpred, y):
    qs = torch.tensor(QUANTILES).view(1, -1, 1)
    d = y[:, None, :] - qpred
    return torch.maximum(qs * d, (qs - 1) * d).mean()


def train_lstm(X, Y, seed=0, epochs=40, lr=2e-3, val_frac=0.15, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(X)
    perm = np.random.permutation(n)
    nv = int(n * val_frac)
    va, tr = perm[:nv], perm[nv:]
    net = QuantileLSTM()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = map(torch.tensor, (X, Y))
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        np.random.shuffle(tr)
        for b in range(0, len(tr), 64):
            j = tr[b:b + 64]
            loss = pinball(net(Xt[j]), Yt[j])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(pinball(net(Xt[va]), Yt[va]))
        log(f"  lstm seed={seed} ep={ep} val_pinball={vl:.5f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    return net.eval(), best


class Persistence(nn.Module):
    """Smart-persistence reference: tomorrow = the last 24 h (point forecast in all quantiles)."""

    def __init__(self, horizon=24):
        super().__init__()
        self.name = "persistence"
        self.horizon = horizon

    def forward(self, ctx, **kw):
        last = ctx[:, -24:][:, : self.horizon]
        return last[:, None, :].repeat(1, 9, 1)
