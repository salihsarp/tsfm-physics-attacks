"""Figures and tables from results/*.csv. Journal style: Times New Roman 8 pt, 300 dpi,
single column 8.5 cm, double column 17.5 cm. Model identity uses colour + marker + line style
(never colour alone). Usage: python3 make_figures.py [--synthetic]"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RES_DIR, FIG_DIR, PROC_DIR

CM = 1 / 2.54
COL1, COL2 = 8.5 * CM, 17.5 * CM
MODELS = ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm"]
LABEL = {"chronos-bolt-small": "Chronos-Bolt (small)", "chronos-2": "Chronos-2", "timesfm-2.5": "TimesFM 2.5",
         "lstm": "LSTM (3 seeds)", "lstm_s0": "LSTM", "persistence": "Persistence",
         "ensemble(3 TSFM)": "TSFM ensemble"}
COLOR = {"chronos-bolt-small": "#2a78d6", "chronos-2": "#eb6834", "timesfm-2.5": "#1baf7a", "lstm": "#eda100",
         "lstm_s0": "#eda100", "ensemble(3 TSFM)": "#555555"}
MARK = {"chronos-bolt-small": "o", "chronos-2": "s", "timesfm-2.5": "^", "lstm": "D", "lstm_s0": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3",
    "grid.linewidth": 0.5, "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "stix",
})


def P(name):
    return os.path.join(RES_DIR, PFX + name)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{PFX}{name}.{ext}"))
    plt.close(fig)


def load_attack():
    dfs = []
    for m in MODELS:
        f = P(f"attack_{m}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["family"] = m
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    # LSTM: average over the three trained seeds; random noise: average over noise seeds
    num = ["nMAE", "nRMSE", "bias", "QL", "PICP80", "PIW80", "dev_from_clean", "frac_flagged_physics"]
    keys = ["family", "attack", "constraint", "objective", "eps"]
    g = df.groupby(keys)[[c for c in num if c in df]]
    out = g.mean().reset_index()
    sd = g.std().reset_index()
    return out, sd, df


# ------------------------------------------------------------------------- Fig: error vs budget
def fig_budget(att):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 5.2 * CM), sharex=True)
    panels = [("nMAE", "nMAE (fraction of capacity)"), ("bias", "Bias of P50 (fraction of capacity)"),
              ("PICP80", "80 % interval coverage")]
    for ax, (met, lab) in zip(axes, panels):
        for m in MODELS:
            d = att[att.family == m]
            if d.empty:
                continue
            clean = d[d.attack == "clean"][met].values[0]
            for cons, ls in (("UC", "-"), ("PC", "--")):
                e = d[(d.attack == "pgd") & (d.constraint == cons) & (d.objective == "dev")].sort_values("eps")
                ax.plot(np.r_[0, e.eps], np.r_[clean, e[met]], ls=ls, color=COLOR[m], marker=MARK[m], ms=3,
                        label=f"{LABEL[m]} {cons}")
            r = d[(d.attack == "random") & (d.constraint == "UC")].sort_values("eps")
            ax.plot(np.r_[0, r.eps], np.r_[clean, r[met]], ls=":", color=COLOR[m], lw=0.9)
        if met == "PICP80":
            ax.axhline(0.8, color="#888888", lw=0.6)
        ax.set_xlabel(r"Budget $\varepsilon$ (fraction of " + ("capacity)" if DOMAIN == "solar" else "2017 peak)"))
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.tight_layout()
    save(fig, "fig_budget")


# ------------------------------------------------------------------------- Fig: transfer heatmap
def fig_transfer():
    f = P("transfer.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    clean = d[d.source == "clean"].set_index("target")["nMAE"]
    for cons in ("UC", "PC"):
        t = d[(d.constraint == cons)]
        if t.empty:
            continue
        piv = t.pivot(index="source", columns="target", values="nMAE")
        piv = piv.subtract(clean[piv.columns], axis=1) / clean[piv.columns] * 100
        rows = [r for r in ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0"] if r in piv.index]
        cols = [c for c in ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0", "ensemble(3 TSFM)"]
                if c in piv.columns]
        piv = piv.loc[rows, cols]
        fig, ax = plt.subplots(figsize=(COL1, 5.6 * CM))
        vmax = max(1.0, float(np.nanmax(piv.values)))
        im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(range(len(cols)), [LABEL.get(c, c) for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)), [LABEL.get(r, r) for r in rows])
        ax.set_xlabel("Target model"); ax.set_ylabel("Surrogate (attack crafted on)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("nMAE increase (%)")
        fig.tight_layout()
        save(fig, f"fig_transfer_{cons}")


# ------------------------------------------------------------------------- Fig: detection
def fig_detection():
    f = P("detection.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    clean = d[d.file == "clean_2018"].iloc[0]
    d = d[d.file != "clean_2018"].copy()
    d["model"] = d.file.str.replace(PFX, "", regex=False).str.split("__").str[0]
    d["cfg"] = d.file.str.split("__").str[1]
    d = d[d.cfg.str.contains("dev_0.05")]
    order = [m for m in ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0"] if m in set(d.model)]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 5.0 * CM), sharey=True)
    for ax, det, title in zip(axes, ("physics", "residual"), ("Physics bad-data check", "Forecast-residual detector")):
        x = np.arange(len(order))
        for k, (cons, hatch) in enumerate((("UC", None), ("PC", "////"))):
            vals = [d[(d.model == m) & (d.cfg == f"pgd_{cons}_dev_0.05")][det].mean() for m in order]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=[COLOR[m] for m in order],
                   edgecolor="white", linewidth=0.8, hatch=hatch, label=f"PGD-{cons}")
            for xi, v in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        ax.axhline(clean[det], color="#444444", lw=0.7, ls="--")
        ax.text(len(order) - 0.5, clean[det] + 0.03, f"clean 2018: {clean[det]:.2f}", ha="right", fontsize=6)
        ax.set_xticks(x, [LABEL[m] for m in order], rotation=20, ha="right")
        ax.set_title(title); ax.set_ylim(0, 1.12)
    axes[0].set_ylabel(r"Flagged fraction of windows ($\varepsilon$ = 0.05)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_detection")


# ------------------------------------------------------------------------- Fig: example window
def fig_example(model="chronos-bolt-small", eps=0.05, pick=None):
    W = dict(np.load(os.path.join(PROC_DIR, f"{PFX}windows.npz")))
    idx = np.where(W["sub"])[0]
    X, Y, UX, LX = W["teX"][idx], W["teY"][idx], W["teUX"][idx], W["teLX"][idx]
    fdir = os.path.join(RES_DIR, "fc")
    qc = np.load(os.path.join(fdir, f"{PFX}{model}__clean_subset.npy"))
    qa = {c: np.load(os.path.join(fdir, f"{PFX}{model}__pgd_{c}_dev_{eps}.npy")) for c in ("UC", "PC")}
    xa = {c: np.load(os.path.join(RES_DIR, "adv", f"{PFX}{model}__pgd_{c}_dev_{eps}.npy")) for c in ("UC", "PC")}
    if pick is None:  # window with the median PC-induced deviation among sunny-ish days
        dev = np.abs(qa["PC"][:, 4] - qc[:, 4]).mean(1)
        pick = int(np.argsort(dev)[len(dev) // 2])
    h_hist = 72
    t_hist = np.arange(-h_hist, 0)
    t_fut = np.arange(0, 24)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 5.2 * CM), sharey=True)
    for ax, c in zip(axes, ("UC", "PC")):
        xadv = X[pick].copy(); xadv[-24:] = xa[c][pick]
        ax.fill_between(t_hist, LX[pick, -h_hist:], UX[pick, -h_hist:], color="#f1f1f1", lw=0,
                        label="Clear-sky envelope" if DOMAIN == "solar" else "Plausibility band")
        ax.plot(t_hist, X[pick, -h_hist:], color="#222222", lw=1.0, label="Measured history")
        ax.plot(t_hist[-24:], xadv[-24:], color="#d62828" if c == "UC" else "#8e44ad", lw=1.0, ls="-",
                label=f"Manipulated history ({c})")
        ax.plot(t_fut, Y[pick], color="#222222", lw=1.0, ls=":", label="Actual (next day)")
        ax.fill_between(t_fut, qc[pick, 0], qc[pick, 8], color=COLOR[model], alpha=0.18, lw=0)
        ax.plot(t_fut, qc[pick, 4], color=COLOR[model], lw=1.2, label="Forecast P50 (clean input)")
        ax.fill_between(t_fut, qa[c][pick, 0], qa[c][pick, 8], color="#999999", alpha=0.25, lw=0)
        ax.plot(t_fut, qa[c][pick, 4], color="#555555", lw=1.2, ls="--", label="Forecast P50 (attacked input)")
        ax.axvline(0, color="#888888", lw=0.6)
        ax.set_title(f"{'Unconstrained' if c == 'UC' else 'Physically constrained'} PGD, " + r"$\varepsilon$" + f" = {eps}")
        ax.set_xlabel("Hours relative to issue time")
    axes[0].set_ylabel("PV power (fraction of capacity)" if DOMAIN == "solar" else "Load (fraction of 2017 peak)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    fig.tight_layout()
    save(fig, "fig_example")
    return pick


# ------------------------------------------------------------------------- tables
def tables(att, sd):
    out = {}
    c = P("clean_metrics.csv")
    if os.path.exists(c):
        d = pd.read_csv(c)
        out["clean"] = d
    sel = att[(att.eps.isin([0.0, 0.05]))].copy()
    out["attack_eps005"] = sel.round(4)
    for k, v in out.items():
        v.to_csv(os.path.join(RES_DIR, f"{PFX}table_{k}.csv"), index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--domain", default="solar", choices=["solar", "load"])
    a = ap.parse_args()
    DOMAIN = a.domain
    PFX = ("syn_" if a.synthetic else "") + DOMAIN + "_"
    att, sd, raw = load_attack()
    fig_budget(att)
    fig_transfer()
    fig_detection()
    try:
        fig_example()
    except FileNotFoundError as e:
        print("example figure skipped:", e)
    tables(att, sd)
    print("figures written to", FIG_DIR)
