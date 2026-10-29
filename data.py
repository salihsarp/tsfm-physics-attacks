"""ARPA-E PERFORM (ERCOT) data: download, hourly aggregation, windows, physical plausibility envelopes.

Source: NREL ARPA-E PERFORM datasets, s3://arpa-e-perform (CC BY 4.0, DOI 10.25984/1891136).
Solar (site level): ERCOT/{YEAR}/Solar/Actuals/Site_level/Site_<name>_solar_actuals_{YEAR}.h5
                    ERCOT/2018/Solar/Forecasts/Day-ahead/Site_level/Site_<name>_solar_day-ahead_fcst_2018.h5
Load (8 weather zones + system):
                    ERCOT/{YEAR}/Load/Actuals/{Zone_level,BA_level}/<Zone_name|BA>_load_actuals_{YEAR}.h5
                    ERCOT/2018/Load/Forecasts/Day-ahead/{Zone_level,BA_level}/<...>_load_day-ahead_fcst_2018.h5
"""
from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
import requests

from common import RAW_DIR, PROC_DIR, ExpConfig

BUCKET = "https://arpa-e-perform.s3.amazonaws.com"


def _bucket():
    """Use the bucket's regional endpoint if S3 asks for it (x-amz-bucket-region)."""
    global BUCKET
    r = requests.get(BUCKET + "/", params={"list-type": "2", "max-keys": "1"}, timeout=60, allow_redirects=False)
    region = r.headers.get("x-amz-bucket-region")
    if r.status_code != 200 and region:
        BUCKET = f"https://arpa-e-perform.s3.{region}.amazonaws.com"
    return BUCKET


# ------------------------------------------------------------------ download
def s3_list(prefix: str) -> list[str]:
    keys, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
        if root.find("s3:IsTruncated", ns).text == "true":
            token = root.find("s3:NextContinuationToken", ns).text
        else:
            return keys


def s3_get(key: str, out_dir: str = RAW_DIR) -> str:
    path = os.path.join(out_dir, key.replace("/", "__"))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with requests.get(f"{BUCKET}/{key}", stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(tmp, path)
    return path


def _is_proposed(meta: dict) -> bool:
    """PERFORM site metadata carries a 'proposed' field ('Proposed' for interconnection-queue sites)."""
    v = meta.get("proposed", "")
    v = v.decode() if isinstance(v, bytes) else str(v)
    return "propos" in v.lower()


def download_ercot(domain="solar"):
    _bucket()
    keys = {}
    if domain == "solar":
        # only the existing (operating) sites are used; their names are read from the 2017 file metadata
        k17 = [k for k in s3_list("ERCOT/2017/Solar/Actuals/Site_level/") if k.endswith(".h5")]
        for i, k in enumerate(k17):
            s3_get(k)
            if i % 20 == 0:
                print(f"  2017 site files {i + 1}/{len(k17)}", flush=True)
        existing = set()
        for k in k17:
            meta, _, _ = read_actuals(os.path.join(RAW_DIR, k.replace("/", "__")))
            if not _is_proposed(meta.iloc[0].to_dict()):
                existing.add(series_name(k, "solar"))
        print(f"  existing solar sites: {len(existing)} of {len(k17)}", flush=True)
        keys["act2017"] = [k for k in k17 if series_name(k, "solar") in existing]
        keys["act2018"] = [k for k in s3_list("ERCOT/2018/Solar/Actuals/Site_level/")
                           if k.endswith(".h5") and series_name(k, "solar") in existing]
        keys["da2018"] = [k for k in s3_list("ERCOT/2018/Solar/Forecasts/Day-ahead/Site_level/")
                          if k.endswith(".h5") and series_name(k, "solar") in existing]
    else:
        for year in (2017, 2018):
            keys[f"act{year}"] = [k for lvl in ("Zone_level", "BA_level")
                                  for k in s3_list(f"ERCOT/{year}/Load/Actuals/{lvl}/") if k.endswith(".h5")]
        keys["da2018"] = [k for lvl in ("Zone_level", "BA_level")
                          for k in s3_list(f"ERCOT/2018/Load/Forecasts/Day-ahead/{lvl}/") if k.endswith(".h5")]
    print(domain, {k: len(v) for k, v in keys.items()}, flush=True)
    for name, group in keys.items():
        for i, k in enumerate(group):
            s3_get(k)
            if i % 10 == 0 or i == len(group) - 1:
                print(f"  {domain} {name}: {i + 1}/{len(group)} files", flush=True)
    return keys


# ------------------------------------------------------------------ parsing
def _decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def read_actuals(path):
    with h5py.File(path, "r") as f:
        meta = pd.DataFrame(f["meta"][...])
        for c in meta.columns:
            if meta[c].dtype == object:
                meta[c] = _decode(meta[c].values)
        t = pd.to_datetime(_decode(f["time_index"][...]), utc=True)
        x = f["actuals"][...]
    if x.dtype.names:            # solar site files store a structured array with one field ('df_power')
        x = x[x.dtype.names[0]]
    x = np.asarray(x, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    return meta, t, x


def read_forecast(path):
    with h5py.File(path, "r") as f:
        meta = pd.DataFrame(f["meta"][...])
        names = list(f.keys())
        issue = pd.to_datetime(_decode(f["issue_time"][...]), utc=True)
        ftime = pd.to_datetime(_decode(f["forecast_time"][...]), utc=True)
        fc = f["forecasts"][...].astype("float64")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["forecasts"].attrs.items()}
    return meta, issue, ftime, fc, names, attrs


def series_name(fn, domain):
    """Series name from a PERFORM file name, or None if the file is not part of the domain's panel."""
    b = os.path.basename(fn.replace("__", "/"))
    if domain == "solar":
        m = re.match(r"Site_(.+?)_solar_", b)
        return m.group(1) if m else None
    if b.startswith("BA_load_"):
        return "ERCOT_total"
    m = re.match(r"Zone_(.+?)_load_", b)
    return m.group(1) if m else None


# ------------------------------------------------------------------ build hourly panel
def build_panel(domain="solar"):
    """Hourly normalised actuals 2017-2018 per series, metadata, scale and DA forecasts.
    Solar is normalised by AC capacity (metadata, else observed max); load by the 2017 peak."""
    files = sorted(os.listdir(RAW_DIR))
    act = {2017: {}, 2018: {}}
    metas = {}
    key = "_solar_actuals_" if domain == "solar" else "_load_actuals_"
    for fn in files:
        if key in fn and fn.endswith(".h5") and "Existing" not in fn:
            name = series_name(fn, domain)
            if name is None:
                continue
            year = int(fn[-7:-3])
            meta, t, x = read_actuals(os.path.join(RAW_DIR, fn))
            if domain == "solar" and len(meta) and _is_proposed(meta.iloc[0].to_dict()):
                continue  # proposed (not yet built) sites are excluded
            act[year][name] = pd.Series(x[:, 0], index=t)
            metas[name] = meta.iloc[0].to_dict() if len(meta) else {}
    names = sorted(set(act[2017]) & set(act[2018]))
    hourly = {}
    for name in names:
        s_ = pd.concat([act[2017][name], act[2018][name]]).sort_index()
        s_ = s_[~s_.index.duplicated()]
        hourly[name] = s_.resample("1h", label="left", closed="left").mean()
    df = pd.DataFrame(hourly).loc["2017-01-01 00:00":"2018-12-31 23:00"]
    caps = {}
    for name in names:
        if domain == "load":
            caps[name] = float(np.nanmax(df[name].loc["2017"].values))
            continue
        m, cap = metas[name], None
        for k in m:
            if re.search(r"capacity", str(k), re.I):
                try:
                    cap = float(m[k]); break
                except (TypeError, ValueError):
                    pass
        obs_max = float(np.nanmax(df[name].values))
        caps[name] = cap if (cap and cap >= 0.5 * obs_max) else obs_max
    norm = df / pd.Series(caps)
    da = {}
    key = "_solar_day-ahead_fcst_2018" if domain == "solar" else "_load_day-ahead_fcst_2018"
    for fn in files:
        if key in fn:
            name = series_name(fn, domain)
            if name not in names:
                continue
            meta, issue, ftime, fc, _, attrs = read_forecast(os.path.join(RAW_DIR, fn))
            ncol = fc.shape[1] if fc.ndim == 2 else 1
            # PERFORM forecast columns: [actual, deterministic, P1..P99] (101) or P1..P99 (99) -> use P50
            det_col = 1 if ncol == 101 else (49 if ncol == 99 else 0)
            da[name] = dict(issue=issue, ftime=ftime, fc=fc / caps[name], det_col=det_col, ncol=ncol, attrs=attrs)
    da_lag = align_da(da, norm) if da else 0
    return dict(norm=norm, caps=caps, metas={k: metas[k] for k in names}, da=da, sites=names, da_lag_hours=da_lag)


def align_da(da, norm, lags=(-2, -1, 0, 1, 2)):
    """Time-stamp convention check: shift every day-ahead forecast by the whole-hour lag that maximises its
    mean correlation with the hourly actuals (hour-beginning means) across all series; returns the lag."""
    score = {}
    for lag in lags:
        cs = []
        for name, d in da.items():
            det = d["fc"][:, d["det_col"]] if d["fc"].ndim == 2 else d["fc"]
            ser = (pd.DataFrame({"issue": d["issue"], "ftime": d["ftime"], "v": det}).sort_values("issue")
                   .drop_duplicates("ftime", keep="last").set_index("ftime")["v"])
            ser.index = ser.index + pd.Timedelta(hours=lag)
            act = norm[name].reindex(ser.index)
            cs.append(ser.corr(act))
        score[lag] = float(np.nanmean(cs))
    best = max(score, key=score.get)
    print("  day-ahead alignment (lag h: mean corr):", {k: round(v, 4) for k, v in score.items()}, "-> lag", best,
          flush=True)
    for d in da.values():
        d["ftime"] = d["ftime"] + pd.Timedelta(hours=best)
        d["align_scores"] = score
    return best


def latlon(meta):
    lat = lon = None
    for k, v in meta.items():
        kl = str(k).lower()
        if kl in ("latitude", "lat"):
            lat = float(v)
        if kl in ("longitude", "lon", "long"):
            lon = float(v)
    return lat, lon


def solar_elevation(index_utc: pd.DatetimeIndex, lat, lon):
    import pvlib
    mid = index_utc + pd.Timedelta(minutes=30)
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
    return sp["apparent_elevation"].values


# ------------------------------------------------------------------ physics envelope
def _seasonal_table(x, how, half_window_days):
    tab = x.groupby([x.index.dayofyear, x.index.hour]).agg(how).unstack()  # doy x hour (UTC)
    tab = tab.reindex(range(1, 367)).ffill().bfill()
    ext = pd.concat([tab.iloc[-half_window_days:], tab, tab.iloc[:half_window_days]])
    roll = ext.rolling(2 * half_window_days + 1, center=True, min_periods=1)
    roll = roll.max() if how == "max" else roll.min()
    roll = roll.iloc[half_window_days:-half_window_days]
    roll.index = tab.index
    return roll.values


def envelope_components(norm: pd.DataFrame, metas: dict, cfg: ExpConfig, domain="solar", half_window_days=10):
    """Building blocks of the physics check, all from the 2017 (training) year:
    HI / LO = max / min of the same hour within +/- half_window_days; NIGHT (solar) = sun below -1 deg at both
    edges of the hour (no PV is physically possible); R0 = cfg.ramp_quantile of the 2017 absolute hourly ramps."""
    train = norm.loc["2017"]
    HI = pd.DataFrame(index=norm.index, columns=norm.columns, dtype=float)
    LO = pd.DataFrame(index=norm.index, columns=norm.columns, dtype=float)
    NIGHT = pd.DataFrame(False, index=norm.index, columns=norm.columns)
    R0 = {}
    doy, hr = norm.index.dayofyear, norm.index.hour
    for s_ in norm.columns:
        x = train[s_]
        HI[s_] = _seasonal_table(x, "max", half_window_days)[doy - 1, hr]
        LO[s_] = _seasonal_table(x, "min", half_window_days)[doy - 1, hr]
        if domain == "solar":
            lat, lon = latlon(metas[s_])
            if lat is not None:
                import pvlib
                e0 = pvlib.solarposition.get_solarposition(norm.index, lat, lon)["apparent_elevation"].values
                e1 = pvlib.solarposition.get_solarposition(norm.index + pd.Timedelta(hours=1), lat, lon)[
                    "apparent_elevation"].values
                NIGHT[s_] = (e0 < -1.0) & (e1 < -1.0)
        R0[s_] = float(np.nanquantile(np.abs(np.diff(x.values)), cfg.ramp_quantile))
    return dict(HI=HI, LO=LO, NIGHT=NIGHT, R0=R0)


def band(comp, tau_b, tau_r=None, domain="solar"):
    """Plausibility band [L, U] with level tolerance tau_b and ramp limit R0 + tau_r (fractions of capacity /
    2017 peak). solar: L = 0, U = HI + tau_b, U = 0 at night (when the night rule is on); load: L = LO - tau_b,
    U = HI + tau_b. tau_r defaults to tau_b."""
    tau_r = tau_b if tau_r is None else tau_r
    HI, LO, NIGHT = comp["HI"], comp["LO"], comp["NIGHT"]
    if domain == "solar":
        U = (HI + tau_b).clip(0.0, 1.0).mask(NIGHT, 0.0)
        Lo = pd.DataFrame(0.0, index=HI.index, columns=HI.columns)
    else:
        U = HI + tau_b
        Lo = (LO - tau_b).clip(lower=0.0)
    R = {k: v + tau_r for k, v in comp["R0"].items()}
    return Lo, U, R


def physics_false_alarm(norm, Lo, U, R, issue_times, seg=25, tol=1e-4, rules=("level", "ramp")):
    """Share of clean windows (last `seg` hours before each issue time) that fail the physics check
    (`rules` selects the level-band test, the ramp test, or both)."""
    pos = norm.index.get_indexer(pd.DatetimeIndex(issue_times))
    flags = []
    for s_ in norm.columns:
        x, u, lo = norm[s_].values, U[s_].values, Lo[s_].values
        for i in pos:
            if i - seg < 0:
                continue
            a, b = x[i - seg:i], slice(i - seg, i)
            bad = False
            if "level" in rules:
                bad |= bool((a > u[b] + tol).any() or (a < lo[b] - tol).any())
            if "ramp" in rules:
                bad |= bool((np.abs(np.diff(a)) > R[s_] + tol).any())
            flags.append(bad)
    return float(np.mean(flags))


def envelope_and_ramp(norm: pd.DataFrame, metas: dict, cfg: ExpConfig, domain="solar", tau=0.0):
    comp = envelope_components(norm, metas, cfg, domain)
    return band(comp, tau, tau, domain)


# ------------------------------------------------------------------ windows
def make_windows(norm, Lo, U, cfg: ExpConfig, year: str, day_stride: int, da: dict | None = None,
                 domain="solar", evalmask=None):
    """Day-ahead windows issued at cfg.issue_hour_utc. Returns dict of arrays. M = evaluation mask
    (daylight hours, U > 0, for solar; every hour for load)."""
    L, H = cfg.context_len, cfg.horizon
    idx = norm.index
    pos = {t: i for i, t in enumerate(idx)}
    issue_times = [t for t in idx if t.hour == cfg.issue_hour_utc and str(t.year) == year]
    issue_times = issue_times[::day_stride]
    X, Y, UX, UY, LX, LY, S, T, COV_PAST, COV_FUT, EVY = [], [], [], [], [], [], [], [], [], [], []
    for s_i, s in enumerate(norm.columns):
        v = norm[s].values
        u = U[s].values
        lo = Lo[s].values
        ev = evalmask[s].values.astype(bool) if evalmask is not None else None
        for t in issue_times:
            i = pos[t]
            if i - L < 0 or i + H > len(v):
                continue
            x, y = v[i - L:i], v[i:i + H]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            cp = cf = None
            if da is not None:
                cp, cf = da_covariate(da, s, idx[i - L:i + H], t)
            X.append(x); Y.append(y); UX.append(u[i - L:i]); UY.append(u[i:i + H])
            EVY.append(ev[i:i + H] if ev is not None else u[i:i + H] > 0)
            LX.append(lo[i - L:i]); LY.append(lo[i:i + H])
            S.append(s_i); T.append(t)
            if da is not None:
                COV_PAST.append(cp); COV_FUT.append(cf)
    UY_ = np.array(UY, "float32")
    out = dict(X=np.array(X, "float32"), Y=np.array(Y, "float32"), UX=np.array(UX, "float32"),
               UY=UY_, LX=np.array(LX, "float32"), LY=np.array(LY, "float32"),
               M=np.array(EVY, dtype=bool) if domain == "solar" else np.ones_like(UY_, dtype=bool),
               site=np.array(S), issue=np.array(T, dtype="datetime64[ns]"))
    if da is not None:
        out["COVP"] = np.array(COV_PAST, "float32")
        out["COVF"] = np.array(COV_FUT, "float32")
        # windows whose context starts before the first day-ahead forecast (early Jan 2018) lack the covariate
        out["COVOK"] = ~(np.isnan(out["COVP"]).any(1) | np.isnan(out["COVF"]).any(1))
    return out


def da_covariate(da, site, times, issue_t):
    """Deterministic day-ahead forecast available at issue time for each hour in `times`.
    Uses, for every hour, the most recent DA issuance strictly before issue_t (no look-ahead)."""
    d = da[site]
    det = d["fc"][:, d["det_col"]] if d["fc"].ndim == 2 else d["fc"]
    df = pd.DataFrame({"issue": d["issue"], "ftime": d["ftime"], "v": det})
    df = df[df["issue"] < issue_t].sort_values("issue").drop_duplicates("ftime", keep="last")
    ser = df.set_index("ftime")["v"].reindex(times)
    vals = ser.values.astype("float32")
    n_fut = 24
    return vals[:-n_fut], vals[-n_fut:]


# ------------------------------------------------------------------ synthetic (code testing ONLY)
def synthetic_panel(n_sites=3, seed=0):
    """Synthetic PV-like data used ONLY to test the code path before the real data are available.
    Never used for reported results."""
    import pvlib
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", "2018-12-31 23:00", freq="1h", tz="UTC")
    norm, metas = {}, {}
    for k in range(n_sites):
        lat, lon = 30 + rng.uniform(-2, 2), -100 + rng.uniform(-3, 3)
        loc = pvlib.location.Location(lat, lon)
        cs = loc.get_clearsky(idx + pd.Timedelta(minutes=30))["ghi"].values / 1000.0
        cloud = np.clip(1 - 0.6 * rng.beta(0.6, 1.5, size=len(idx) // 24 + 1).repeat(24)[: len(idx)]
                        + 0.05 * rng.standard_normal(len(idx)), 0, 1)
        norm[f"synth{k}"] = np.clip(cs * cloud * 0.9, 0, 1)
        metas[f"synth{k}"] = {"latitude": lat, "longitude": lon, "capacity": 1.0}
    return dict(norm=pd.DataFrame(norm, index=idx), caps={k: 1.0 for k in norm}, metas=metas, da={}, sites=list(norm))


def synthetic_load_panel(n_zones=3, seed=0):
    """Synthetic load-like data used ONLY to test the code path. Never used for reported results."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", "2018-12-31 23:00", freq="1h", tz="UTC")
    h = (idx.hour.values - 6) % 24
    doy = idx.dayofyear.values
    norm = {}
    for k in range(n_zones):
        season = 0.65 + 0.2 * np.cos(2 * np.pi * (doy - 200) / 365)
        daily = 0.12 * np.sin(2 * np.pi * (h - 9) / 24)
        weekly = np.where(idx.dayofweek.values >= 5, -0.05, 0.0)
        x = season + daily + weekly + 0.02 * rng.standard_normal(len(idx))
        norm[f"zone{k}"] = x / x[: 8760].max()
    return dict(norm=pd.DataFrame(norm, index=idx), caps={k: 1.0 for k in norm}, metas={k: {} for k in norm},
                da={}, sites=list(norm))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--domain", default="solar", choices=["solar", "load"])
    args = ap.parse_args()
    if args.download:
        download_ercot(args.domain)
