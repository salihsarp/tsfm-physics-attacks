"""Experiment driver. Stages (run in order; each saves its outputs under results/):

  prep      build hourly panel, physics envelope, ramp limits and forecast windows
  lstm      train the quantile-LSTM baseline (3 seeds) on 2017 windows
  clean     clean accuracy of all models on the full 2018 test set and on the attack subset
  attack    white-box grid for one model: --model M   (random, PGD-UC, PGD-PC; dev & risk objectives)
  transfer  black-box transfer of saved adversarial inputs to every other model and to an ensemble
  cov       Chronos-2 with the day-ahead forecast as a known covariate: attacks on target vs covariate
  detect    physics bad-data detector and forecast-residual detector on every saved attack
  defense   input sanitisation and split-conformal recalibration under attack

Every stage takes --domain solar|load (ERCOT solar sites or ERCOT weather-zone load).
"risk" objective = push the forecast in the direction that makes net load under-forecast:
raise PV forecasts (solar) or lower load forecasts (load).
Use --synthetic to exercise the code with random-weight models and synthetic data (never reported).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from common import (ExpConfig, PROC_DIR, RES_DIR, WEIGHTS_DIR, QUANTILES, UPPER, RISK_OBJECTIVE, all_metrics,
                    postprocess, set_seed)
import attacks as A
import data as D
import models as M

CFG = ExpConfig()
SYN = False
DOMAIN = "solar"
ATTACK_DAYS = (8, 22)          # attack subset: 2 issue days per month of 2018
MODELS = ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(os.path.join(RES_DIR, "log.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def pfx():
    return ("syn_" if SYN else "") + DOMAIN + "_"


# ----------------------------------------------------------------------------- loading models
def load_model(name, use_cov=False):
    """All forecasters are returned wrapped in M.OnDevice (TSFMs on the GPU/MPS if present, LSTM on CPU)."""
    if name.startswith("lstm"):
        return M.OnDevice(_load_lstm(name), device="cpu")
    if SYN:
        return M.OnDevice(_synthetic_model(name, use_cov))
    if name == "chronos-bolt-small":
        return M.OnDevice(M.ChronosBolt("chronos-bolt-small", CFG.horizon))
    if name == "chronos-2":
        return M.OnDevice(M.Chronos2("chronos-2", CFG.horizon, use_cov=use_cov))
    if name == "timesfm-2.5":
        return M.OnDevice(M.TimesFM25(CFG.horizon))
    raise ValueError(name)


def _synthetic_model(name, use_cov):
    from transformers import T5Config
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    torch.manual_seed(0)
    if name == "chronos-bolt-small":
        cfg = T5Config(d_model=64, d_ff=128, num_layers=2, num_decoder_layers=2, num_heads=2, d_kv=32,
                       dense_act_fn="relu", feed_forward_proj="relu", decoder_start_token_id=0, pad_token_id=0)
        cfg.chronos_config = dict(context_length=2048, prediction_length=64, input_patch_size=16,
                                  input_patch_stride=16, quantiles=QUANTILES, use_reg_token=True)
        return M.ChronosBolt(model=ChronosBoltModelForForecasting(cfg), horizon=CFG.horizon)
    if name == "chronos-2":
        cfg = Chronos2CoreConfig(d_model=64, d_ff=128, num_layers=2, num_heads=2, d_kv=32)
        cfg.chronos_config = dict(context_length=8192, output_patch_size=16, input_patch_size=16,
                                  input_patch_stride=16, quantiles=[0.01, 0.05] + QUANTILES + [0.95, 0.99],
                                  use_reg_token=True, use_arcsinh=True, max_output_patches=64)
        return M.Chronos2(model=Chronos2Model(cfg), horizon=CFG.horizon, use_cov=use_cov)
    if name == "timesfm-2.5":
        return _SynTimes()
    raise ValueError(name)


class _SynTimes(torch.nn.Module):
    name = "timesfm-2.5"

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(CFG.context_len, 9 * CFG.horizon)

    def forward(self, ctx, **kw):
        return self.lin(ctx).view(-1, 9, CFG.horizon)


def _load_lstm(name):
    seed = int(name.split("_s")[1]) if "_s" in name else 0
    net = M.QuantileLSTM()
    net.load_state_dict(torch.load(os.path.join(RES_DIR, f"{pfx()}lstm_seed{seed}.pt"), map_location="cpu"))
    net.name = f"lstm_s{seed}"
    return net.eval()


# ----------------------------------------------------------------------------- data
def stage_prep():
    if SYN:
        panel = D.synthetic_panel() if DOMAIN == "solar" else D.synthetic_load_panel()
    else:
        panel = D.build_panel(DOMAIN)
    norm = panel["norm"]
    # physics check: tolerance tau = smallest value whose false-alarm rate on clean VALIDATION windows
    # (2018 issue days other than the attack days) is <= 1 %; attack days are never used for calibration
    comp = D.envelope_components(norm, panel["metas"], CFG, domain=DOMAIN)
    t18 = [t for t in norm.index if t.hour == CFG.issue_hour_utc and t.year == 2018]
    val_t = [t for t in t18 if t.day not in ATTACK_DAYS]
    att_t = [t for t in t18 if t.day in ATTACK_DAYS]
    # two rules, each calibrated to half the target (union <= target): level band (tau_b) and ramp limit (tau_r)
    grid = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    half = CFG.physics_target_fpr / 2
    calib = []
    for night_rule in ((True, False) if DOMAIN == "solar" else (False,)):
        c_ = dict(comp, NIGHT=comp["NIGHT"] if night_rule else comp["NIGHT"] & False)
        for tau_b in grid:
            Lo, U, R = D.band(c_, tau_b, 0.0, DOMAIN)
            f = D.physics_false_alarm(norm, Lo, U, R, val_t, rules=("level",))
            calib.append(dict(rule="level", night_rule=night_rule, tau=tau_b, fpr_validation=f))
            log("physics-check calibration", DOMAIN, calib[-1])
            if f <= half:
                break
        if calib[-1]["fpr_validation"] <= half:
            break
    night_rule, tau_b = calib[-1]["night_rule"], calib[-1]["tau"]
    comp = dict(comp, NIGHT=comp["NIGHT"] if night_rule else comp["NIGHT"] & False)
    for tau_r in grid:
        Lo, U, R = D.band(comp, tau_b, tau_r, DOMAIN)
        f = D.physics_false_alarm(norm, Lo, U, R, val_t, rules=("ramp",))
        calib.append(dict(rule="ramp", night_rule=night_rule, tau=tau_r, fpr_validation=f))
        log("physics-check calibration", DOMAIN, calib[-1])
        if f <= half:
            break
    Lo, U, R = D.band(comp, tau_b, tau_r, DOMAIN)
    fpr_val_both = D.physics_false_alarm(norm, Lo, U, R, val_t)
    tau = dict(level=tau_b, ramp=tau_r)
    fpr_attack_days = D.physics_false_alarm(norm, Lo, U, R, att_t)
    da = panel["da"] if panel["da"] else None
    # solar scoring hours: the plant produced power at that hour in the same +/-10-day window of 2017
    # (independent of the physics-check tolerance); load: every hour
    ev = (comp["HI"] > 0) if DOMAIN == "solar" else None
    tr = D.make_windows(norm, Lo, U, CFG, "2017", 1, domain=DOMAIN, evalmask=ev)
    te = D.make_windows(norm, Lo, U, CFG, "2018", 1, da=da, domain=DOMAIN, evalmask=ev)
    days = pd.DatetimeIndex(te["issue"]).day
    sub = np.isin(days, ATTACK_DAYS)
    Rarr = np.array([R[s] for s in norm.columns], "float32")
    np.savez_compressed(os.path.join(PROC_DIR, f"{pfx()}windows.npz"),
                        trX=tr["X"], trY=tr["Y"], trUX=tr["UX"], trUY=tr["UY"], trLX=tr["LX"], trLY=tr["LY"],
                        trM=tr["M"], trS=tr["site"],
                        teX=te["X"], teY=te["Y"], teUX=te["UX"], teUY=te["UY"], teLX=te["LX"], teLY=te["LY"],
                        teM=te["M"], teS=te["site"],
                        teT=te["issue"].astype("int64"), sub=sub, R=Rarr,
                        **({"teCP": te["COVP"], "teCF": te["COVF"], "teCOVOK": te["COVOK"]} if da is not None else {}))
    info = dict(domain=DOMAIN, sites=list(norm.columns), caps=panel["caps"], ramp=R,
                physics_tau=tau, physics_night_rule=bool(night_rule), physics_calibration=calib,
                physics_fpr_validation=fpr_val_both, physics_fpr_attack_days=fpr_attack_days,
                site_names={k: str(v.get("site_ids", k)) for k, v in panel["metas"].items()},
                da_columns={k: v.get("ncol") for k, v in panel["da"].items()},
                da_lag_hours=panel.get("da_lag_hours", 0),
                n_train=int(len(tr["X"])), n_test=int(len(te["X"])), n_attack_subset=int(sub.sum()),
                n_test_with_covariate=int(te["COVOK"].sum()) if da is not None else 0,
                hours=int(len(norm)), missing_frac=float(norm.isna().mean().mean()))
    json.dump(info, open(os.path.join(RES_DIR, f"{pfx()}data_info.json"), "w"), indent=2, default=str)
    CFG.save(os.path.join(RES_DIR, f"{pfx()}config.json"))
    log("prep done", {k: v for k, v in info.items() if k.startswith("n_")})


def load_windows():
    return dict(np.load(os.path.join(PROC_DIR, f"{pfx()}windows.npz")))


# ----------------------------------------------------------------------------- helpers
def predict(model, X, bs=None, **cov):
    bs = bs or CFG.batch_size
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            kw = {k: torch.tensor(v[b:b + bs]) for k, v in cov.items()}
            outs.append(postprocess(model(torch.tensor(X[b:b + bs]), **kw), UPPER[DOMAIN]).numpy())
    return np.concatenate(outs)


def metrics_row(y, q, mask, **tags):
    return dict(**tags, **all_metrics(y, q, mask.astype(bool)))


def objective_name(obj):
    return RISK_OBJECTIVE[DOMAIN] if obj == "risk" else obj


def attack_batch(model, Xb, LXb, UXb, Rb, eps, kind, constraint, objective, gen=None, cov=None):
    """Attack the last CFG.attack_window hours of the context. Returns the full adversarial context."""
    W = CFG.attack_window
    x_fix, x_win = Xb[:, :-W], Xb[:, -W:]
    edit = torch.ones_like(x_win, dtype=torch.bool)
    cons = None
    if constraint == "PC":
        cons = dict(lo=LXb[:, -W:], hi=UXb[:, -W:], R=Rb, left=x_fix[:, -1] if x_fix.shape[1] else None)
    cov = cov or {}
    f = lambda xw: model(torch.cat([x_fix, xw], 1), **cov)
    if kind == "random":
        xw = A.random_noise(x_win, edit, eps, cons, generator=gen)
    else:
        xw = A.pgd(f, x_win, edit, eps, kind=objective_name(objective), steps=CFG.pgd_steps,
                   alpha_frac=CFG.pgd_alpha_frac, cons=cons)
    return torch.cat([x_fix, xw], 1)


def run_attack_grid(model, W, idx, grid, tag, save_adv=True, n_random_seeds=3):
    rows = []
    X, Y, UX, LX, MK = W["teX"][idx], W["teY"][idx], W["teUX"][idx], W["teLX"][idx], W["teM"][idx]
    R = W["R"][W["teS"][idx]]
    q_clean = predict(model, X)
    np.save(os.path.join(RES_DIR, "fc", f"{pfx()}{tag}__clean_subset.npy"), q_clean.astype("float32"))
    rows.append(metrics_row(Y, q_clean, MK, model=tag, attack="clean", constraint="-", objective="-", eps=0.0, seed=0))
    for (kind, constraint, objective, eps) in grid:
        seeds = range(n_random_seeds) if kind == "random" else [0]
        for sd in seeds:
            t0 = time.time()
            gen = torch.Generator().manual_seed(sd)
            adv = []
            for b in range(0, len(X), CFG.batch_size):
                sl = slice(b, b + CFG.batch_size)
                adv.append(attack_batch(model, torch.tensor(X[sl]), torch.tensor(LX[sl]), torch.tensor(UX[sl]),
                                        torch.tensor(R[sl]), eps, kind, constraint, objective, gen=gen).numpy())
            adv = np.concatenate(adv)
            q_adv = predict(model, adv)
            sw = min(CFG.attack_window + 1, adv.shape[1])
            viol = A.violations(torch.tensor(adv[:, -sw:]), torch.tensor(LX[:, -sw:]),
                                torch.tensor(UX[:, -sw:]), torch.tensor(R)).float().mean().item()
            row = metrics_row(Y, q_adv, MK, model=tag, attack=kind, constraint=constraint,
                              objective=objective if kind != "random" else "-", eps=eps, seed=sd)
            row["dev_from_clean"] = float(np.abs(q_adv[:, 4] - q_clean[:, 4])[MK].mean())
            row["frac_flagged_physics"] = viol
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            log(tag, kind, constraint, row["objective"], eps, sd,
                f"nMAE={row['nMAE']:.4f} bias={row['bias']:+.4f} PICP={row['PICP80']:.3f} "
                f"flag={viol:.2f} t={row['seconds']}s")
            if save_adv and kind == "random" and sd == 0:
                np.save(os.path.join(RES_DIR, "fc", f"{pfx()}{tag}__random_{constraint}_-_{eps}.npy"),
                        q_adv.astype("float32"))
            if save_adv and kind != "random":
                np.save(os.path.join(RES_DIR, "adv", f"{pfx()}{tag}__{kind}_{constraint}_{objective}_{eps}.npy"),
                        adv[:, -CFG.attack_window:].astype("float32"))
                np.save(os.path.join(RES_DIR, "fc", f"{pfx()}{tag}__{kind}_{constraint}_{objective}_{eps}.npy"),
                        q_adv.astype("float32"))
    return rows


# ----------------------------------------------------------------------------- stages
def stage_lstm(seeds=(0, 1, 2)):
    W = load_windows()
    res = []
    for sd in seeds:
        net, vl = M.train_lstm(W["trX"], W["trY"], seed=sd, epochs=3 if SYN else 40, log=log)
        torch.save(net.state_dict(), os.path.join(RES_DIR, f"{pfx()}lstm_seed{sd}.pt"))
        res.append(dict(seed=sd, val_pinball=vl))
    pd.DataFrame(res).to_csv(os.path.join(RES_DIR, f"{pfx()}lstm_training.csv"), index=False)


def stage_clean():
    W = load_windows()
    rows = []
    sub = W["sub"]
    names = ["persistence"] + MODELS[:-1] + ["lstm_s0", "lstm_s1", "lstm_s2"]
    for name in names:
        model = M.Persistence() if name == "persistence" else load_model(name)
        t0 = time.time()
        q = predict(model, W["teX"])
        np.save(os.path.join(RES_DIR, "fc", f"{pfx()}clean_{name}.npy"), q.astype("float32"))
        for split, m in (("full", slice(None)), ("attack_subset", sub)):
            rows.append(metrics_row(W["teY"][m], q[m], W["teM"][m], model=name, split=split))
        log("clean", name, f"{time.time() - t0:.0f}s", rows[-2])
    if "teCF" in W and not SYN:
        ok = W["teCOVOK"]
        # all models again on the covariate-complete windows, so the +cov comparison is like for like
        for name in ["chronos-2", "chronos-bolt-small", "timesfm-2.5", "lstm_s0"]:
            q = np.load(os.path.join(RES_DIR, "fc", f"{pfx()}clean_{name}.npy"))
            rows.append(metrics_row(W["teY"][ok], q[ok], W["teM"][ok], model=name, split="full_with_covariate"))
        model = load_model("chronos-2", use_cov=True)
        q = np.full((len(W["teX"]), 9, CFG.horizon), np.nan, "float32")
        q[ok] = predict(model, W["teX"][ok], cov_past=W["teCP"][ok], cov_fut=W["teCF"][ok])
        np.save(os.path.join(RES_DIR, "fc", f"{pfx()}clean_chronos-2+cov.npy"), q.astype("float32"))
        # the PERFORM day-ahead forecast itself (its P50) as a reference
        qd = np.repeat(np.clip(np.nan_to_num(W["teCF"]), 0, UPPER[DOMAIN])[:, None, :], 9, axis=1)
        for nm, qq in (("chronos-2+cov", q), ("PERFORM day-ahead P50", qd)):
            for split, m in (("full_with_covariate", ok), ("attack_subset_with_covariate", sub & ok)):
                rows.append(metrics_row(W["teY"][m], qq[m], W["teM"][m], model=nm, split=split))
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}clean_metrics.csv"), index=False)


def main_grid(full_up=False):
    g = []
    for c in ("UC", "PC"):
        for e in CFG.eps_list:
            g.append(("random", c, "-", e))
            g.append(("pgd", c, "dev", e))
        for e in (CFG.eps_list if full_up else (0.05,)):
            g.append(("pgd", c, "risk", e))
    return g


def stage_attack(name):
    W = load_windows()
    idx = np.where(W["sub"])[0]
    if name == "lstm":
        rows = []
        for sd in (0, 1, 2):
            rows += run_attack_grid(load_model(f"lstm_s{sd}"), W, idx, main_grid(), f"lstm_s{sd}", save_adv=(sd == 0))
    else:
        rows = run_attack_grid(load_model(name), W, idx, main_grid(), name)
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}attack_{name}.csv"), index=False)


def stage_transfer(eps=0.05):
    W = load_windows()
    idx = np.where(W["sub"])[0]
    X, Y, UY = W["teX"][idx], W["teY"][idx], W["teM"][idx]
    targets = ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0"]
    models = {t: load_model(t) for t in targets}
    clean = {t: predict(m, X) for t, m in models.items()}
    rows = []
    for c in ("UC", "PC"):
        for src in targets:
            f = os.path.join(RES_DIR, "adv", f"{pfx()}{src}__pgd_{c}_dev_{eps}.npy")
            if not os.path.exists(f):
                continue
            adv = X.copy()
            adv[:, -CFG.attack_window:] = np.load(f)
            qs = {}
            for t, m in models.items():
                qs[t] = predict(m, adv)
                rows.append(metrics_row(Y, qs[t], UY, source=src, target=t, constraint=c, eps=eps))
            ens_adv = np.median(np.stack([qs[t] for t in targets[:3]]), 0)
            rows.append(metrics_row(Y, ens_adv, UY, source=src, target="ensemble(3 TSFM)", constraint=c, eps=eps))
    ens_clean = np.median(np.stack([clean[t] for t in targets[:3]]), 0)
    rows.append(metrics_row(Y, ens_clean, UY, source="clean", target="ensemble(3 TSFM)", constraint="-", eps=0.0))
    for t in targets:
        rows.append(metrics_row(Y, clean[t], UY, source="clean", target=t, constraint="-", eps=0.0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, f"{pfx()}transfer.csv"), index=False)


def stage_cov():
    """Chronos-2 with the PERFORM day-ahead (NWP-based) forecast as past+future covariate."""
    W = load_windows()
    idx = np.where(W["sub"] & W["teCOVOK"])[0]
    X, Y, UX, UY = W["teX"][idx], W["teY"][idx], W["teUX"][idx], W["teUY"][idx]
    LX, LY, MK = W["teLX"][idx], W["teLY"][idx], W["teM"][idx]
    CP, CF, R = W["teCP"][idx], W["teCF"][idx], W["R"][W["teS"][idx]]
    model = load_model("chronos-2", use_cov=True)
    q_clean = predict(model, X, cov_past=CP, cov_fut=CF)
    rows = [metrics_row(Y, q_clean, MK, target_of_attack="clean", constraint="-", eps=0.0)]
    for c in ("UC", "PC"):
        for eps in CFG.eps_list:
            for where in ("history", "cov_future"):
                t0 = time.time()
                adv_X, adv_CF = X.copy(), CF.copy()
                for b in range(0, len(X), CFG.batch_size):
                    sl = slice(b, b + CFG.batch_size)
                    Xb, CPb, CFb = map(torch.tensor, (X[sl], CP[sl], CF[sl]))
                    Rb, UXb, UYb = torch.tensor(R[sl]), torch.tensor(UX[sl]), torch.tensor(UY[sl])
                    LXb, LYb = torch.tensor(LX[sl]), torch.tensor(LY[sl])
                    if where == "history":
                        adv = attack_batch(model, Xb, LXb, UXb, Rb, eps, "pgd", c, "dev",
                                           cov=dict(cov_past=CPb, cov_fut=CFb))
                        adv_X[sl] = adv.numpy()
                    else:
                        edit = torch.ones_like(CFb, dtype=torch.bool)
                        cons = dict(lo=LYb, hi=UYb, R=Rb, left=CPb[:, -1]) if c == "PC" else None
                        f = lambda cf: model(Xb, cov_past=CPb, cov_fut=cf)
                        adv_CF[sl] = A.pgd(f, CFb, edit, eps, "dev", CFG.pgd_steps, CFG.pgd_alpha_frac, cons).numpy()
                q = predict(model, adv_X, cov_past=CP, cov_fut=adv_CF)
                row = metrics_row(Y, q, MK, target_of_attack=where, constraint=c, eps=eps)
                row["dev_from_clean"] = float(np.abs(q[:, 4] - q_clean[:, 4])[MK].mean())
                rows.append(row)
                log("cov", where, c, eps, f"nMAE={row['nMAE']:.4f} t={time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}cov_attack.csv"), index=False)


def stage_detect():
    """(i) physics bad-data detector (band calibrated in prep), (ii) forecast-residual detector: Chronos-Bolt-small
    forecasts the last 24 h of the context from the preceding hours; score = MAE over daylight hours (solar) or
    all hours (load); threshold = 95th percentile of clean validation scores (2018 days outside the attack set)."""
    W = load_windows()
    Wn = CFG.attack_window
    det = load_model("chronos-bolt-small")

    def score(Xc, UXc):
        q = predict(det, Xc[:, :-Wn])
        day = (UXc[:, -Wn:] > 0) if DOMAIN == "solar" else np.ones_like(UXc[:, -Wn:], dtype=bool)
        e = np.abs(q[:, 4, :Wn] - Xc[:, -Wn:]) * day
        return e.sum(1) / np.maximum(day.sum(1), 1)

    val = ~W["sub"]                       # 2018 issue days not used for attacks
    s_val = score(W["teX"][val], W["teUX"][val])
    thr = float(np.quantile(s_val, 0.95))
    idx = np.where(W["sub"])[0]
    X, UX, LX, R = W["teX"][idx], W["teUX"][idx], W["teLX"][idx], W["R"][W["teS"][idx]]

    def physics(Xc):
        seg = torch.tensor(Xc[:, -Wn - 1:])
        return A.violations(seg, torch.tensor(LX[:, -Wn - 1:]), torch.tensor(UX[:, -Wn - 1:]),
                            torch.tensor(R)).numpy()

    rows = [dict(file="clean_2018", physics=float(physics(X).mean()), residual=float((score(X, UX) > thr).mean()),
                 residual_threshold=thr)]
    for f in sorted(glob.glob(os.path.join(RES_DIR, "adv", f"{pfx()}*.npy"))):
        adv = X.copy()
        adv[:, -Wn:] = np.load(f)
        rows.append(dict(file=os.path.basename(f)[:-4], physics=float(physics(adv).mean()),
                         residual=float((score(adv, UX) > thr).mean()), residual_threshold=thr))
        log("detect", rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}detection.csv"), index=False)


def stage_defense():
    """(a) sanitisation: project the received context onto the physical constraint set before forecasting;
    (b) split-conformal (CQR-style) widening of the 80 % interval, calibrated per lead hour on clean 2017."""
    W = load_windows()
    Wn = CFG.attack_window
    idx = np.where(W["sub"])[0]
    X, Y, UX, LX, MK = W["teX"][idx], W["teY"][idx], W["teUX"][idx], W["teLX"][idx], W["teM"][idx]
    R = W["R"][W["teS"][idx]]
    up = UPPER[DOMAIN] if UPPER[DOMAIN] is not None else np.inf
    rows = []
    for name in ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0"]:
        model = load_model(name)
        # conformal calibration on 2017 (clean)
        qtr = predict(model, W["trX"])
        day_tr = W["trM"]
        e = np.maximum(qtr[:, 0] - W["trY"], W["trY"] - qtr[:, 8])  # CQR score with q0.1/q0.9
        e = np.where(day_tr, e, np.nan)
        qhat = np.nanquantile(e, 0.8, axis=0, method="higher")       # per lead hour
        src = name if name != "lstm_s0" else "lstm_s0"
        for c in ("UC", "PC"):
            f = os.path.join(RES_DIR, "adv", f"{pfx()}{src}__pgd_{c}_dev_0.05.npy")
            variants = {"clean": X}
            if os.path.exists(f):
                adv = X.copy(); adv[:, -Wn:] = np.load(f)
                variants[f"pgd_{c}"] = adv
            for vname, Xv in variants.items():
                if vname == "clean" and c == "PC":
                    continue
                for sanit in (False, True):
                    Xin = Xv
                    if sanit:
                        seg = torch.tensor(Xv[:, -Wn:])
                        lo = torch.tensor(LX[:, -Wn:])
                        hi = torch.tensor(UX[:, -Wn:])
                        san = A.project(seg, seg, torch.ones_like(seg, dtype=torch.bool), 10.0, lo, hi,
                                        torch.tensor(R), torch.tensor(Xv[:, -Wn - 1]))
                        Xin = Xv.copy(); Xin[:, -Wn:] = san.numpy()
                    q = predict(model, Xin)
                    r = metrics_row(Y, q, MK, model=name, input=vname, sanitised=sanit, conformal=False)
                    qc = q.copy()
                    qc[:, 0] = np.clip(q[:, 0] - qhat, 0, up)
                    qc[:, 8] = np.clip(q[:, 8] + qhat, 0, up)
                    rc = metrics_row(Y, qc, MK, model=name, input=vname, sanitised=sanit, conformal=True)
                    rows += [r, rc]
                    log("defense", name, vname, sanit, f"nMAE={r['nMAE']:.4f} PICP={r['PICP80']:.3f} "
                                                      f"PICP_cqr={rc['PICP80']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}defense.csv"), index=False)


def stage_steps(eps=0.05):
    """Robustness check of the attack itself: PC-PGD (dev) with 5, 10, 20 and 40 steps (step 2.5 eps / steps)."""
    W = load_windows()
    idx = np.where(W["sub"])[0]
    rows = []
    for name in ["chronos-bolt-small", "timesfm-2.5", "lstm_s0"]:
        model = load_model(name)
        for steps in (5, 10, 20, 40):
            af = 2.5 / steps
            CFG.pgd_steps, CFG.pgd_alpha_frac = steps, af
            r = run_attack_grid(model, W, idx, [("pgd", "PC", "dev", eps)], name, save_adv=False)[-1]
            r["steps"] = steps
            rows.append(r)
    CFG.pgd_steps, CFG.pgd_alpha_frac = ExpConfig().pgd_steps, ExpConfig().pgd_alpha_frac
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}pgd_steps.csv"), index=False)


def stage_window(eps=0.05):
    """Ablation: how many hours of history the attacker edits (24, 96 or all 384)."""
    W = load_windows()
    idx = np.where(W["sub"])[0]
    rows = []
    for name in ["chronos-bolt-small", "chronos-2", "lstm_s0"]:
        model = load_model(name)
        for win in (24, 96, 384):
            CFG.attack_window = win
            for c in ("UC", "PC"):
                r = run_attack_grid(model, W, idx, [("pgd", c, "dev", eps)], name, save_adv=False)[-1]
                r["attack_window"] = win
                rows.append(r)
    CFG.attack_window = 24
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, f"{pfx()}attack_window.csv"), index=False)


def stage_verify(n=64):
    """Check that the differentiable wrappers reproduce the official inference APIs on n real test windows."""
    W = load_windows()
    X = W["teX"][:n]
    rows = []
    from chronos import BaseChronosPipeline
    for name in ("chronos-bolt-small", "chronos-2"):
        ours = predict(load_model(name), X)
        pipe = BaseChronosPipeline.from_pretrained(os.path.join(WEIGHTS_DIR, name), device_map="cpu",
                                                   torch_dtype=torch.float32)
        if name == "chronos-2":
            qs, _ = pipe.predict_quantiles(X[:, None, :], prediction_length=CFG.horizon, quantile_levels=QUANTILES)
            ref = torch.stack([q[0] for q in qs]).permute(0, 2, 1)
        else:
            qs, _ = pipe.predict_quantiles(torch.tensor(X), prediction_length=CFG.horizon, quantile_levels=QUANTILES)
            ref = qs.permute(0, 2, 1)
        ref = postprocess(ref.float(), UPPER[DOMAIN]).numpy()
        rows.append(dict(model=name, max_abs_diff=float(np.abs(ours - ref).max())))
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.load_checkpoint(os.path.join(WEIGHTS_DIR, "timesfm-2.5-200m-pytorch", "model.safetensors"),
                        torch_compile=False)
    tfm.compile(timesfm.ForecastConfig(max_context=CFG.context_len, max_horizon=128, per_core_batch_size=32,
                                       normalize_inputs=True, use_continuous_quantile_head=True,
                                       force_flip_invariance=True, infer_is_positive=True,
                                       fix_quantile_crossing=True))
    _, q = tfm.forecast(horizon=CFG.horizon, inputs=[x.astype("float64") for x in X])
    ref = postprocess(torch.tensor(q[..., 1:], dtype=torch.float32).permute(0, 2, 1), UPPER[DOMAIN]).numpy()
    ours = predict(load_model("timesfm-2.5"), X)
    rows.append(dict(model="timesfm-2.5", max_abs_diff=float(np.abs(ours - ref).max())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, f"{pfx()}verify_wrappers.csv"), index=False)
    log("verify", df.to_dict("records"))
    bad = df[df.max_abs_diff > 2e-3]
    if len(bad):
        raise SystemExit(f"Wrapper mismatch vs official API: {bad.to_dict('records')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--model", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--domain", default="solar", choices=["solar", "load"])
    args = ap.parse_args()
    SYN = args.synthetic
    DOMAIN = args.domain
    if args.steps:
        CFG.pgd_steps = args.steps
    os.makedirs(os.path.join(RES_DIR, "adv"), exist_ok=True)
    os.makedirs(os.path.join(RES_DIR, "fc"), exist_ok=True)
    set_seed(CFG.seed)
    log(f"stage={args.stage} domain={DOMAIN} model={args.model} device={M.DEVICE} threads={torch.get_num_threads()}")
    {"prep": stage_prep, "lstm": stage_lstm, "clean": stage_clean, "transfer": stage_transfer,
     "cov": stage_cov, "detect": stage_detect, "defense": stage_defense, "verify": stage_verify,
     "steps": stage_steps, "window": stage_window,
     "attack": lambda: stage_attack(args.model)}[args.stage]()
