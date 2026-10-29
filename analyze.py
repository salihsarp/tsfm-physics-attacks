"""Final tables and figures for the manuscript, from a finished run (results/, data/processed/).

Confidence intervals: day-cluster bootstrap (all windows of one issue day are resampled together, because the
22 sites / 9 load series share weather), 2000 resamples, percentile 95 % intervals, paired for attack-vs-clean.
Usage: python3 analyze.py <run_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN, OUT = sys.argv[1], sys.argv[2]
RES, PROC = os.path.join(RUN, "results"), os.path.join(RUN, "data", "processed")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
B = 2000
RNG = np.random.default_rng(2026)
QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MODELS = ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm_s0"]
NAME = {"chronos-bolt-small": "Chronos-Bolt (small)", "chronos-2": "Chronos-2", "timesfm-2.5": "TimesFM 2.5",
        "lstm_s0": "LSTM", "lstm": "LSTM", "persistence": "Persistence", "chronos-2+cov": "Chronos-2 + DA covariate",
        "PERFORM day-ahead P50": "PERFORM day-ahead (P50)", "ensemble(3 TSFM)": "TSFM median ensemble"}
COLOR = {"chronos-bolt-small": "#2a78d6", "chronos-2": "#eb6834", "timesfm-2.5": "#1baf7a", "lstm_s0": "#eda100"}
MARK = {"chronos-bolt-small": "o", "chronos-2": "s", "timesfm-2.5": "^", "lstm_s0": "D"}
DOM_TITLE = {"solar": "Solar PV (22 sites)", "load": "Load (9 series)"}
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "stix", "axes.axisbelow": True})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"{name}.{ext}"))
    plt.close(fig)


# ----------------------------------------------------------------------------- metrics on arrays
def nmae_parts(y, q, m):
    """per-window sums so that clusters can be resampled: (sum |e| over masked hours, count)"""
    e = np.abs(q[:, 4] - y) * m
    return e.sum(1), m.sum(1)


def bias_parts(y, q, m):
    return ((q[:, 4] - y) * m).sum(1), m.sum(1)


def picp_parts(y, q, m):
    ins = ((y >= q[:, 0]) & (y <= q[:, 8])) * m
    return ins.sum(1), m.sum(1)


def pinball_parts(y, q, m):
    d = y[:, None, :] - q
    pin = np.maximum(QS[None, :, None] * d, (QS[None, :, None] - 1) * d).mean(1) * m
    return pin.sum(1), m.sum(1)


def cluster_index(days):
    u, inv = np.unique(days, return_inverse=True)
    return [np.where(inv == k)[0] for k in range(len(u))]


def boot(stat_fn, clusters):
    """stat_fn(idx) -> float; returns point estimate and 95 % percentile CI over cluster resamples"""
    allidx = np.concatenate(clusters)
    point = stat_fn(allidx)
    vals = []
    for _ in range(B):
        pick = RNG.integers(0, len(clusters), len(clusters))
        vals.append(stat_fn(np.concatenate([clusters[k] for k in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def ratio(parts, idx):
    s, n = parts
    return s[idx].sum() / max(n[idx].sum(), 1)


# ----------------------------------------------------------------------------- loading
def load_domain(d):
    W = dict(np.load(os.path.join(PROC, f"{d}_windows.npz")))
    info = json.load(open(os.path.join(RES, f"{d}_data_info.json")))
    fc = lambda name: np.load(os.path.join(RES, "fc", f"{d}_{name}.npy"))
    return W, info, fc


def fmt_ci(p, lo, hi, nd=3, pct=False):
    if pct:
        return f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


tables = {}
summary = {}

for d in ("solar", "load"):
    W, info, fc = load_domain(d)
    Y, M, T = W["teY"], W["teM"].astype(float), W["teT"]
    sub = W["sub"]
    ok = W.get("teCOVOK", np.ones(len(Y), bool))
    S = {}

    # ---------------- Table 1: data and physics-check summary
    caps = info["caps"]
    unit = "MW AC capacity" if d == "solar" else "MW 2017 peak"
    tau = info["physics_tau"]
    t1 = {
        "Series": f"{len(caps)} " + ("existing PV sites" if d == "solar" else "load series (8 weather zones + ERCOT total)"),
        f"Size ({unit})": f"{min(caps.values()):,.1f} – {max(caps.values()):,.1f} (sum {sum(caps.values()):,.0f})",
        "Hourly values per series": f"{info['hours']:,} (2017–2018)",
        "Training windows (2017)": f"{info['n_train']:,}",
        "Test windows (2018, all days)": f"{info['n_test']:,}",
        "Attack windows (days 8 and 22 of each month)": f"{info['n_attack_subset']:,}",
        "Test windows with day-ahead covariate": f"{info['n_test_with_covariate']:,}",
        "Physics check: level tolerance τ_b": f"{tau['level']:.3f}",
        "Physics check: ramp tolerance τ_r": f"{tau['ramp']:.3f}",
        "Ramp limit R0 + τ_r (per hour)": f"{min(info['ramp'].values()):.3f} – {max(info['ramp'].values()):.3f}",
        "Night-zero rule": "yes" if info["physics_night_rule"] else "not applicable",
        "Clean false alarms, validation days": f"{100 * info['physics_fpr_validation']:.2f} %",
        "Clean false alarms, attack days": f"{100 * info['physics_fpr_attack_days']:.2f} %",
    }
    tables[f"T1_{d}"] = pd.DataFrame({"Item": list(t1), DOM_TITLE[d]: list(t1.values())})

    # ---------------- Table 2: clean accuracy, full 2018 test set (bootstrap over 364 issue days)
    cl_all = cluster_index(T)
    cl_ok = cluster_index(T[ok])
    clean_csv = pd.read_csv(os.path.join(RES, f"{d}_clean_metrics.csv"))
    rows = []
    for name in ["persistence", "lstm_s0", "chronos-bolt-small", "chronos-2", "timesfm-2.5"]:
        q = fc(f"clean_{name}")
        p, lo, hi = boot(lambda i: ratio(nmae_parts(Y, q, M), i), cl_all)
        r = dict(Model=NAME[name], Windows="all 2018", nMAE=fmt_ci(p, lo, hi, 4),
                 nRMSE=np.sqrt(ratio((((q[:, 4] - Y) ** 2 * M).sum(1), M.sum(1)), slice(None))),
                 Bias=ratio(bias_parts(Y, q, M), slice(None)), Pinball=ratio(pinball_parts(Y, q, M), slice(None)),
                 PICP80=ratio(picp_parts(Y, q, M), slice(None)) if name != "persistence" else np.nan,
                 PIW80=float(((q[:, 8] - q[:, 0]) * M).sum() / M.sum()) if name != "persistence" else np.nan)
        if d == "load":
            r["MAPE (%)"] = 100 * float((np.abs(q[:, 4] - Y) * M / np.maximum(Y, 1e-6))[Y > 0.05].mean())
        if name == "lstm_s0":
            seeds = clean_csv[(clean_csv.model.str.startswith("lstm")) & (clean_csv.split == "full")]["nMAE"]
            r["Model"] = f"LSTM (seed 0; 3 seeds {seeds.min():.4f}–{seeds.max():.4f})"
        S[f"clean_nmae_{name}"] = p
        rows.append(r)
    for name, path in (("PERFORM day-ahead P50", None), ("chronos-2+cov", "clean_chronos-2+cov")):
        if path is None:
            qd = np.repeat(np.clip(np.nan_to_num(W["teCF"]), 0, 1 if d == "solar" else None)[:, None, :], 9, 1)
        else:
            qd = fc(path)
        yy, mm, qq = Y[ok], M[ok], qd[ok]
        p, lo, hi = boot(lambda i: ratio(nmae_parts(yy, qq, mm), i), cl_ok)
        r = dict(Model=NAME[name], Windows="with covariate", nMAE=fmt_ci(p, lo, hi, 4),
                 nRMSE=np.sqrt(ratio((((qq[:, 4] - yy) ** 2 * mm).sum(1), mm.sum(1)), slice(None))),
                 Bias=ratio(bias_parts(yy, qq, mm), slice(None)), Pinball=ratio(pinball_parts(yy, qq, mm), slice(None)),
                 PICP80=ratio(picp_parts(yy, qq, mm), slice(None)) if path else np.nan,
                 PIW80=float(((qq[:, 8] - qq[:, 0]) * mm).sum() / mm.sum()) if path else np.nan)
        if d == "load":
            r["MAPE (%)"] = 100 * float((np.abs(qq[:, 4] - yy) * mm / np.maximum(yy, 1e-6))[yy > 0.05].mean())
        S[f"clean_nmae_{name}"] = p
        rows.append(r)
    # Chronos-2 univariate on the same covariate windows, for a like-for-like comparison
    q = fc("clean_chronos-2")
    p, lo, hi = boot(lambda i: ratio(nmae_parts(Y[ok], q[ok], M[ok]), i), cl_ok)
    rows.append(dict(Model="Chronos-2 (same windows)", Windows="with covariate", nMAE=fmt_ci(p, lo, hi, 4),
                     nRMSE=np.nan, Bias=np.nan, Pinball=np.nan, PICP80=np.nan, PIW80=np.nan))
    t2 = pd.DataFrame(rows)
    tables[f"T2_{d}"] = t2

    # ---------------- Table 3: attacks at eps = 0.05 (and 0.10) on the attack subset (24 issue days)
    ys, ms, ts = Y[sub], M[sub], T[sub]
    cl_sub = cluster_index(ts)
    att = pd.concat([pd.read_csv(os.path.join(RES, f"{d}_attack_{m}.csv")) for m in
                     ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm"]])
    rows = []
    for m in MODELS:
        qc = fc(f"{m}__clean_subset")
        pc = nmae_parts(ys, qc, ms)
        base = ratio(pc, slice(None))
        r = {"Model": NAME[m], "Clean nMAE": f"{base:.4f}"}
        for eps in (0.05, 0.10):
            for lab, f in (("Random PC", f"{m}__random_PC_-_{eps}"), ("PGD-UC", f"{m}__pgd_UC_dev_{eps}"),
                           ("PGD-PC", f"{m}__pgd_PC_dev_{eps}")):
                qa = fc(f)
                pa = nmae_parts(ys, qa, ms)
                p, lo, hi = boot(lambda i: 100 * (ratio(pa, i) / ratio(pc, i) - 1), cl_sub)
                r[f"{lab} ε={eps:.2f} ΔnMAE %"] = fmt_ci(p, lo, hi, pct=True)
                S[f"dnmae_{lab}_{eps}_{m}"] = p
        # risk objective: shift of the P50 bias (fraction of capacity / peak), PC and UC
        for lab, f in (("PGD-UC risk", f"{m}__pgd_UC_risk_0.05"), ("PGD-PC risk", f"{m}__pgd_PC_risk_0.05")):
            qa = fc(f)
            ba, bc = bias_parts(ys, qa, ms), bias_parts(ys, qc, ms)
            p, lo, hi = boot(lambda i: ratio(ba, i) - ratio(bc, i), cl_sub)
            r[f"{lab} ε=0.05 Δbias"] = f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            S[f"dbias_{lab}_{m}"] = p
        qa = fc(f"{m}__pgd_PC_dev_0.05")
        r["PICP80 clean → PGD-PC ε=0.05"] = f"{ratio(picp_parts(ys, qc, ms), slice(None)):.3f} → " \
                                          f"{ratio(picp_parts(ys, qa, ms), slice(None)):.3f}"
        tag = "lstm_s0" if m == "lstm_s0" else m
        fl = att[(att.model == tag) & (att.attack == "pgd") & (att.eps == 0.05) & (att.objective == "dev")]
        r["Flagged by physics check, UC / PC (%)"] = \
            f"{100 * fl[fl.constraint == 'UC'].frac_flagged_physics.iloc[0]:.1f} / " \
            f"{100 * fl[fl.constraint == 'PC'].frac_flagged_physics.iloc[0]:.1f}"
        rows.append(r)
    tables[f"T3_{d}"] = pd.DataFrame(rows)
    # LSTM seed spread for the main attack
    ls = att[att.model.str.startswith("lstm") & (att.attack == "pgd") & (att.constraint == "PC") &
             (att.objective == "dev") & (att.eps == 0.05)]
    lc = att[att.model.str.startswith("lstm") & (att.attack == "clean")]
    S["lstm_seed_dnmae_pc005"] = list(100 * (ls.set_index("model").nMAE / lc.set_index("model").nMAE - 1))

    # risk shift in MW for the system total (load) and the fleet (solar)
    if d == "load":
        k = list(caps).index("ERCOT_total")
        sel = W["teS"][sub] == k
        for m in MODELS:
            qc, qa = fc(f"{m}__clean_subset"), fc(f"{m}__pgd_PC_risk_0.05")
            S[f"risk_MW_ERCOT_{m}"] = float(((qa[sel, 4] - qc[sel, 4]).mean()) * caps["ERCOT_total"])
    else:
        capv = np.array([caps[s] for s in caps])[W["teS"][sub]]
        for m in MODELS:
            qc, qa = fc(f"{m}__clean_subset"), fc(f"{m}__pgd_PC_risk_0.05")
            dmw = ((qa[:, 4] - qc[:, 4]) * ms).sum(1) / np.maximum(ms.sum(1), 1) * capv   # daylight mean per window
            # fleet = sum over the 22 sites of each issue day, averaged over days
            fleet = pd.Series(dmw).groupby(ts).sum().mean()
            S[f"risk_MW_fleet_{m}"] = float(fleet)

    # ---------------- Table 4: covariate attack (Chronos-2 + DA covariate)
    cov = pd.read_csv(os.path.join(RES, f"{d}_cov_attack.csv"))
    c0 = cov[cov.target_of_attack == "clean"].nMAE.iloc[0]
    cov = cov[cov.target_of_attack != "clean"].copy()
    cov["ΔnMAE %"] = 100 * (cov.nMAE / c0 - 1)
    t4 = cov.pivot_table(index=["target_of_attack", "constraint"], columns="eps", values="ΔnMAE %").round(1)
    t4.columns = [f"ε={c:.2f}" for c in t4.columns]
    t4 = t4.reset_index().rename(columns={"target_of_attack": "Attacked input", "constraint": "Constraint"})
    t4["Attacked input"] = t4["Attacked input"].map({"history": "Measured history (last 24 h)",
                                                     "cov_future": "Day-ahead forecast feed (next 24 h)"})
    # univariate Chronos-2 for comparison (history attack, PC)
    uni = att[(att.model == "chronos-2") & (att.attack == "pgd") & (att.objective == "dev")]
    uc0 = att[(att.model == "chronos-2") & (att.attack == "clean")].nMAE.iloc[0]
    add = []
    for c in ("UC", "PC"):
        rr = {"Attacked input": "Measured history, Chronos-2 without covariate", "Constraint": c}
        for e in (0.02, 0.05, 0.10):
            rr[f"ε={e:.2f}"] = round(100 * (uni[(uni.constraint == c) & (uni.eps == e)].nMAE.iloc[0] / uc0 - 1), 1)
        add.append(rr)
    t4 = pd.concat([t4, pd.DataFrame(add)], ignore_index=True)
    tables[f"T4_{d}"] = t4
    S["cov_clean_nmae"] = c0

    # ---------------- Table 5: detection and defenses at eps = 0.05 (dev objective)
    det = pd.read_csv(os.path.join(RES, f"{d}_detection.csv"))
    det["file"] = det.file.str.replace(f"{d}_", "", regex=False)
    dfn = pd.read_csv(os.path.join(RES, f"{d}_defense.csv"))
    rows = []
    clean_det = det[det.file == "clean_2018"].iloc[0]
    rows.append({"Model": "Clean data", "Input": "–", "Physics check (%)": 100 * clean_det.physics,
                 "Residual detector (%)": 100 * clean_det.residual})
    for m in MODELS:
        for c in ("UC", "PC"):
            dd = det[det.file == f"{m}__pgd_{c}_dev_0.05"].iloc[0]
            g = dfn[(dfn.model == m) & (dfn.input == f"pgd_{c}")]
            g0 = dfn[(dfn.model == m) & (dfn.input == "clean") & (~dfn.sanitised) & (~dfn.conformal)].iloc[0]
            raw = g[(~g.sanitised) & (~g.conformal)].iloc[0]
            san = g[(g.sanitised) & (~g.conformal)].iloc[0]
            cqr = g[(~g.sanitised) & (g.conformal)].iloc[0]
            cqr0 = dfn[(dfn.model == m) & (dfn.input == "clean") & (~dfn.sanitised) & (dfn.conformal)].iloc[0]
            rows.append({"Model": NAME[m], "Input": f"PGD-{c}",
                         "Physics check (%)": 100 * dd.physics, "Residual detector (%)": 100 * dd.residual,
                         "nMAE attacked": raw.nMAE, "nMAE after sanitisation": san.nMAE,
                         "PICP80 clean / attacked": f"{g0.PICP80:.3f} / {raw.PICP80:.3f}",
                         "PICP80 with CQR clean / attacked": f"{cqr0.PICP80:.3f} / {cqr.PICP80:.3f}"})
    tables[f"T5_{d}"] = pd.DataFrame(rows)

    # ---------------- appendix: PGD steps, attack window, transfer
    st = pd.read_csv(os.path.join(RES, f"{d}_pgd_steps.csv"))
    tables[f"A1_{d}"] = st.pivot_table(index="model", columns="steps", values="nMAE").rename(index=NAME).round(4) \
        .reset_index().rename(columns={"model": "Model"})
    aw = pd.read_csv(os.path.join(RES, f"{d}_attack_window.csv"))
    base = {m: att[(att.model == m) & (att.attack == "clean")].nMAE.iloc[0] for m in aw.model.unique()}
    aw["ΔnMAE %"] = [100 * (r.nMAE / base[r.model] - 1) for r in aw.itertuples()]
    t = aw.pivot_table(index=["model", "constraint"], columns="attack_window", values="ΔnMAE %").round(1)
    t.columns = [f"{c} h" for c in t.columns]
    t = t.reset_index()
    t["model"] = t.model.map(NAME)
    tables[f"A2_{d}"] = t.rename(columns={"model": "Model", "constraint": "Constraint"})
    tr = pd.read_csv(os.path.join(RES, f"{d}_transfer.csv"))
    summary[d] = S

    # ============================== figures
    # Fig: relative nMAE increase vs budget
    pass

# ----------------------------------------------------------------------------- figures (both domains)
def fig_budget():
    fig, axes = plt.subplots(2, 2, figsize=(17.5 * CM, 10.5 * CM))
    for row, d in enumerate(("solar", "load")):
        att = pd.concat([pd.read_csv(os.path.join(RES, f"{d}_attack_{m}.csv")) for m in
                         ["chronos-bolt-small", "chronos-2", "timesfm-2.5", "lstm"]])
        att = att[att.model.isin(MODELS)]
        for m in MODELS:
            a = att[att.model == m]
            c0 = a[a.attack == "clean"].iloc[0]
            for cons, ls_, lab in (("PC", "-", "PGD-PC"), ("UC", "--", "PGD-UC")):
                e = a[(a.attack == "pgd") & (a.constraint == cons) & (a.objective == "dev")].sort_values("eps")
                axes[row, 0].plot(np.r_[0, e.eps], np.r_[0, 100 * (e.nMAE / c0.nMAE - 1)], ls=ls_, color=COLOR[m],
                                  marker=MARK[m], ms=3, label=f"{NAME[m]}, {lab}")
                axes[row, 1].plot(np.r_[0, e.eps], np.r_[c0.PICP80, e.PICP80], ls=ls_, color=COLOR[m],
                                  marker=MARK[m], ms=3)
            rn = a[(a.attack == "random") & (a.constraint == "PC")].groupby("eps").nMAE.mean()
            axes[row, 0].plot(np.r_[0, rn.index], np.r_[0, 100 * (rn.values / c0.nMAE - 1)], ls=":", lw=0.9,
                              color=COLOR[m])
        axes[row, 1].axhline(0.8, color="#888888", lw=0.6)
        axes[row, 0].set_ylabel(f"{DOM_TITLE[d]}\nnMAE increase (%)")
        axes[row, 1].set_ylabel("80 % interval coverage")
        for ax in axes[row]:
            ax.set_xticks([0, 0.02, 0.05, 0.10])
    for ax in axes[1]:
        ax.set_xlabel(r"Budget $\varepsilon$ (fraction of capacity / 2017 peak)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls=":", color="#555555")); l.append("random noise (PC), each model")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "Fig2_budget")


def fig_example():
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 5.8 * CM))
    for ax, d in zip(axes, ("solar", "load")):
        W, info, fc = load_domain(d)
        sub = W["sub"]
        X, Y, UX, LX = W["teX"][sub], W["teY"][sub], W["teUX"][sub], W["teLX"][sub]
        m = "chronos-2"
        qc, qa = fc(f"{m}__clean_subset"), fc(f"{m}__pgd_PC_risk_0.05")
        xa = np.load(os.path.join(RES, "adv", f"{d}_{m}__pgd_PC_risk_0.05.npy"))
        shift = np.abs(qa[:, 4] - qc[:, 4]).mean(1)
        pick = int(np.argsort(shift)[int(0.75 * len(shift))])   # a window at the 75th percentile of the effect
        th = np.arange(-72, 0); tf = np.arange(0, 24)
        ax.fill_between(th, LX[pick, -72:], UX[pick, -72:], color="#ececec", lw=0, label="Plausibility band")
        ax.plot(th, X[pick, -72:], color="#222222", lw=1.0, label="Measured history")
        ax.plot(th[-24:], xa[pick], color="#8e44ad", lw=1.1, label="Manipulated history (PC)")
        ax.plot(tf, Y[pick], color="#222222", lw=1.0, ls=":", label="Actual next day")
        ax.fill_between(tf, qc[pick, 0], qc[pick, 8], color=COLOR[m], alpha=0.18, lw=0)
        ax.plot(tf, qc[pick, 4], color=COLOR[m], lw=1.2, label="P50, clean input (band: P10–P90)")
        ax.plot(tf, qa[pick, 4], color="#555555", lw=1.2, ls="--", label="P50, manipulated input")
        ax.axvline(0, color="#888888", lw=0.6)
        ax.set_title(f"{DOM_TITLE[d]}, Chronos-2, PGD-PC risk goal, " + r"$\varepsilon$ = 0.05")
        ax.set_xlabel("Hours from issue time (00:00 CST)")
        ax.set_ylabel("PV power (fraction of capacity)" if d == "solar" else "Load (fraction of 2017 peak)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.tight_layout()
    save(fig, "Fig3_example")


def fig_detection():
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 5.4 * CM), sharey=True)
    for ax, d in zip(axes, ("solar", "load")):
        det = pd.read_csv(os.path.join(RES, f"{d}_detection.csv"))
        det["file"] = det.file.str.replace(f"{d}_", "", regex=False)
        c0 = det[det.file == "clean_2018"].iloc[0]
        x = np.arange(len(MODELS)); w = 0.2
        specs = [("physics", "UC", None, 1.0), ("physics", "PC", "////", 1.0),
                 ("residual", "UC", None, 0.45), ("residual", "PC", "////", 0.45)]
        for k, (col, c, hatch, alpha) in enumerate(specs):
            vals = [100 * det[det.file == f"{m}__pgd_{c}_dev_0.05"][col].iloc[0] for m in MODELS]
            ax.bar(x + (k - 1.5) * w, vals, width=w * 0.92, color=[COLOR[m] for m in MODELS], alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.6)
            for xi, v in zip(x + (k - 1.5) * w, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.5)
        ax.axhline(100 * c0.physics, color="#333333", lw=0.7, ls="--")
        ax.axhline(100 * c0.residual, color="#333333", lw=0.7, ls=":")
        ax.set_xticks(x, [NAME[m] for m in MODELS], rotation=15)
        ax.set_title(DOM_TITLE[d]); ax.set_ylim(0, 110)
    axes[0].set_ylabel(r"Windows flagged (%), $\varepsilon$ = 0.05")
    from matplotlib.patches import Patch
    hl = [Patch(facecolor="#777777", label="physics check, UC"), Patch(facecolor="#777777", hatch="////", label="physics check, PC"),
          Patch(facecolor="#777777", alpha=0.45, label="residual detector, UC"),
          Patch(facecolor="#777777", alpha=0.45, hatch="////", label="residual detector, PC"),
          plt.Line2D([], [], color="#333333", ls="--", label="clean, physics"),
          plt.Line2D([], [], color="#333333", ls=":", label="clean, residual")]
    fig.legend(handles=hl, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    save(fig, "Fig4_detection")


def fig_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 6.4 * CM))
    order_t = MODELS + ["ensemble(3 TSFM)"]
    for ax, d in zip(axes, ("solar", "load")):
        tr = pd.read_csv(os.path.join(RES, f"{d}_transfer.csv"))
        clean = tr[tr.source == "clean"].set_index("target").nMAE
        p = tr[tr.constraint == "PC"].pivot(index="source", columns="target", values="nMAE")
        p = (p / clean[p.columns] - 1) * 100
        p = p.loc[MODELS, order_t]
        vmax = max(5.0, float(p.values.max()))
        im = ax.imshow(p.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(range(len(order_t)), [NAME[c] for c in order_t], rotation=30, ha="right")
        ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS])
        ax.set_title(f"{DOM_TITLE[d]}: PGD-PC, " + r"$\varepsilon$ = 0.05")
        ax.set_xlabel("Target model"); ax.set_ylabel("Surrogate (attack crafted on)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("nMAE increase (%)")
    fig.tight_layout()
    save(fig, "Fig5_transfer")


def fig_window():
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 5.4 * CM))
    for ax, d in zip(axes, ("solar", "load")):
        t = tables[f"A2_{d}"]
        t = t[t.Constraint == "PC"]
        x = np.arange(3); w = 0.26
        for k, (mdl, key) in enumerate((("Chronos-Bolt (small)", "chronos-bolt-small"), ("Chronos-2", "chronos-2"),
                                        ("LSTM", "lstm_s0"))):
            vals = t[t.Model == mdl][["24 h", "96 h", "384 h"]].values[0]
            ax.bar(x + (k - 1) * w, vals, width=w * 0.92, color=COLOR[key], label=mdl, edgecolor="white", lw=0.6)
            for xi, v in zip(x + (k - 1) * w, vals):
                ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=6)
        ax.set_xticks(x, ["last 24 h", "last 96 h", "all 384 h"])
        ax.set_title(f"{DOM_TITLE[d]}: PGD-PC, " + r"$\varepsilon$ = 0.05")
        ax.set_ylabel("nMAE increase (%)"); ax.set_xlabel("Hours of history edited")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "Fig6_window")


fig_budget(); fig_example(); fig_detection(); fig_transfer(); fig_window()

# ----------------------------------------------------------------------------- write tables
with pd.ExcelWriter(os.path.join(OUT, "tables", "all_tables.xlsx")) as xw:
    for k, v in tables.items():
        v.to_csv(os.path.join(OUT, "tables", f"{k}.csv"), index=False)
        v.to_excel(xw, sheet_name=k[:31], index=False)
json.dump(summary, open(os.path.join(OUT, "tables", "key_numbers.json"), "w"), indent=2, default=float)
print("written", sorted(tables))
