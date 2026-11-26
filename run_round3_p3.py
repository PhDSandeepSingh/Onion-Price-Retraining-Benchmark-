import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 2027; H = 4; WIN = 156; COV_MIN = 0.80; PRICE_MAX = 20000
DATA = Path("data/onion_2014_2024.parquet"); OUT = Path("results"); FIGS = Path("manuscript/figures")
COLS = ["date", "state", "district", "market_key", "variety", "arrivals_tonnes", "min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl",
        "rain", "tmax", "tmin", "month", "dayofweek", "modal_price_lag_1", "modal_price_lag_7"]
raw = pd.read_parquet(DATA, columns=COLS).sort_values(["market_key", "date"], kind="stable").reset_index(drop=True)

# Rebuild weekly panel identically to make_notebook.py
bad = (raw.modal_price_rs_qtl <= 0) | (raw.min_price_rs_qtl > raw.max_price_rs_qtl) | \
      (raw.modal_price_rs_qtl < raw.min_price_rs_qtl) | (raw.modal_price_rs_qtl > raw.max_price_rs_qtl) | \
      (raw.modal_price_rs_qtl > PRICE_MAX)
clean = raw[~bad].copy()
day = clean.groupby(["market_key", "date"]).agg(state=("state", "first"), p=("modal_price_rs_qtl", "median"),
                                                arr=("arrivals_tonnes", "sum"), rain=("rain", "mean"), tmax=("tmax", "mean")).reset_index()
day["week"] = day.date.dt.to_period("W-SUN").dt.start_time
wk = day.groupby(["market_key", "week"]).agg(state=("state", "first"), p=("p", "median"), arr=("arr", "sum"),
                                             rain=("rain", "sum"), tmax=("tmax", "mean")).reset_index()
WEEKS = pd.date_range("2015-01-05", "2024-12-30", freq="W-MON")
cov = wk[wk.week.isin(WEEKS)].groupby("market_key").week.nunique() / len(WEEKS)
MKTS = cov[cov >= COV_MIN].index.tolist()
idx = pd.MultiIndex.from_product([MKTS, WEEKS], names=["market_key", "week"])
Pn = wk.set_index(["market_key", "week"]).reindex(idx).reset_index()
Pn["state"] = Pn.groupby("market_key").state.transform(lambda s: s.ffill().bfill())
Pn["lp"] = np.log(Pn.p); Pn["lp_f"] = Pn.groupby("market_key").lp.transform(lambda s: s.ffill(limit=2))
g = Pn.groupby("market_key")
for k in (1, 2, 4, 8, 13, 26, 52): Pn[f"r{k}"] = Pn.lp_f - g.lp_f.shift(k)
Pn["dev52"] = Pn.lp_f - g.lp_f.transform(lambda s: s.rolling(52, min_periods=26).mean())
Pn["la"] = np.log1p(Pn.arr); Pn["la_f"] = g.la.transform(lambda s: s.ffill(limit=2)); Pn["da4"] = Pn.la_f - g.la_f.shift(4)
Pn["rain4"] = g.rain.transform(lambda s: s.rolling(4, min_periods=1).sum()); Pn["tmax4"] = g.tmax.transform(lambda s: s.rolling(4, min_periods=1).mean())
woy = Pn.week.dt.isocalendar().week.astype(int); Pn["ws"] = np.sin(2 * np.pi * woy / 52.18); Pn["wc"] = np.cos(2 * np.pi * woy / 52.18)
nat = Pn.groupby("week")[["r1", "r4"]].median().rename(columns={"r1": "nat_r1", "r4": "nat_r4"}); Pn = Pn.merge(nat, left_on="week", right_index=True, how="left")
Pn["state_c"] = Pn.state.astype("category").cat.codes
Pn["y"] = g.lp.shift(-H) - Pn.lp
FE = ["r1", "r2", "r4", "r8", "r13", "r26", "r52", "dev52", "la_f", "da4", "rain4", "tmax4", "ws", "wc", "nat_r1", "nat_r4", "state_c"]
Pn = Pn.sort_values(["week", "market_key"]).reset_index(drop=True)
Pn["wi"] = (Pn.week - WEEKS[0]).dt.days // 7
EVAL0 = int(np.where(WEEKS == pd.Timestamp("2018-01-01"))[0][0]); LAST = int(Pn[Pn.y.notna()].wi.max())

XF = Pn[FE].astype("float32").values; Y = Pn.y.values; WI = Pn.wi.values; lab = ~np.isnan(Y)
start = {T: int(np.searchsorted(WI, T)) for T in range(EVAL0, LAST + 1)}
def pred_of(T, rows): return np.load(OUT / "models" / f"pred_T{T}.npy", mmap_mode="r")[rows - start[T]]
weeks_rows = {w: np.arange(start[w], start[w + 1] if w + 1 in start else start[w] + (WI == w).sum()) for w in range(EVAL0, LAST + 1)}

def psi_fn(ref, cur, bins=10):
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 20: return 0.0
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    a = np.histogram(ref, q)[0] / len(ref) + 1e-4; b = np.histogram(cur, q)[0] / len(cur) + 1e-4
    return float(((b - a) * np.log(b / a)).sum())
PSI_F = [FE.index(c) for c in ("r4", "r13", "la_f", "dev52")]

class ADWIN:
    def __init__(self, delta=0.002): self.d, self.w = delta, []
    def update(self, x):
        self.w.append(x); n = len(self.w)
        for c in range(5, n - 4):
            a, b = np.array(self.w[:c]), np.array(self.w[c:]); m = 1 / (1 / len(a) + 1 / len(b)); v = np.var(self.w); dd = self.d / n
            eps = np.sqrt(2 / m * v * np.log(2 / dd)) + 2 / (3 * m) * np.log(2 / dd)
            if b.mean() - a.mean() > eps: self.w = list(b); return True
        return False

class PageHinkley:
    def __init__(self, delta, lam): self.delta, self.lam = delta, lam; self.reset()
    def reset(self): self.n, self.mean, self.cum, self.mn = 0, 0.0, 0.0, 0.0
    def update(self, x):
        self.n += 1; self.mean += (x - self.mean) / self.n; self.cum += x - self.mean - self.delta; self.mn = min(self.mn, self.cum)
        return self.cum - self.mn > self.lam

pre = (WI < EVAL0) & lab; s0 = float(pd.Series(np.abs(Y[pre])).groupby(WI[pre]).mean().std())
PH_DELTA, PH_LAMBDA = 0.1 * s0, 5.0 * s0

def simulate(policy, ph_d=PH_DELTA, ph_l=PH_LAMBDA, psi_thr=0.2):
    T = EVAL0; retr = [EVAL0]; preds = np.full(len(Y), np.nan); hist = {}
    det = ADWIN() if policy == "adwin" else PageHinkley(ph_d, ph_l) if policy == "pagehinkley" else None
    for w in range(EVAL0, LAST + 1):
        if w > EVAL0:
            go, gap = False, w - T
            if policy == "weekly": go = True
            elif policy.startswith("every"): go = gap >= int(policy[5:])
            elif gap >= 4 and policy == "psi":
                tr = (WI + H <= T) & (WI + H > T - WIN) & lab; cur = (WI > w - 4) & (WI <= w)
                go = max(psi_fn(XF[tr, j], XF[cur, j]) for j in PSI_F) > psi_thr
            elif policy in ("skill", "adwin", "pagehinkley"):
                kw = w - H
                if kw >= EVAL0 and kw in hist:
                    e_m, e_p = hist[kw]
                    if policy == "skill":
                        ks = [k for k in range(kw - 3, kw + 1) if k in hist and k >= T]
                        go = gap >= 4 and len(ks) == 4 and np.mean([hist[k][0] for k in ks]) > np.mean([hist[k][1] for k in ks])
                    elif kw >= T:
                        fired = det.update(e_m); go = fired and gap >= 4
            if go:
                T = w; retr.append(w)
                if det is not None: det = ADWIN() if policy == "adwin" else PageHinkley(ph_d, ph_l)
        rows = weeks_rows[w]; preds[rows] = pred_of(T, rows)
        lr = rows[lab[rows]]
        if len(lr): hist[w] = (np.abs(Y[lr] - preds[lr]).mean(), np.abs(Y[lr]).mean())
    return preds, retr

POL = ["static", "every52", "every13", "every4", "weekly", "psi", "skill", "adwin", "pagehinkley"]
SIM = {p: simulate(p) for p in POL}
ev = (WI >= EVAL0) & lab; P_now = np.exp(Pn.lp.values); A_true = np.exp(Pn.lp.values + Y)
RES = pd.read_csv(OUT / "table_policies.csv").set_index("policy")

REF = "every13"; wk_ids = np.arange(EVAL0, LAST + 1); n_wk = len(wk_ids); B = 1000
def wk_agg(vals):
    return pd.Series(vals[ev]).groupby(WI[ev]).agg(["sum", "count"]).reindex(wk_ids).fillna(0)

w_mae, w_mape, w_log = {}, {}, {}
for p in POL:
    lvl = P_now * np.exp(SIM[p][0]); ae = np.abs(lvl - A_true)
    w_mae[p] = wk_agg(ae); w_mape[p] = wk_agg(100.0 * ae / A_true); w_log[p] = wk_agg(np.abs(Y - SIM[p][0]))
ae_p = np.abs(P_now - A_true)
w_mae["persistence"] = wk_agg(ae_p); w_mape["persistence"] = wk_agg(100.0 * ae_p / A_true); w_log["persistence"] = wk_agg(np.abs(Y))

def circ_boot_pair(ps, pc, ref_s, ref_c, block_len, seed=SEED):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_wk / block_len))
    starts = rng.integers(0, n_wk, size=(B, nb))
    ds = []
    for s in starts:
        ix = np.concatenate([(a + np.arange(block_len)) % n_wk for a in s])[:n_wk]
        ds.append(ps[ix].sum() / pc[ix].sum() - ref_s[ix].sum() / ref_c[ix].sum())
    return float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))

def circ_boot_dict(wdict, block_len, seed=SEED):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_wk / block_len))
    starts = rng.integers(0, n_wk, size=(B, nb))
    out = {}
    ref_s = wdict[REF]["sum"].values; ref_c = wdict[REF]["count"].values
    for p in POL + ["persistence"]:
        if p == REF: continue
        ps = wdict[p]["sum"].values; pc = wdict[p]["count"].values
        ds = []
        for s in starts:
            ix = np.concatenate([(a + np.arange(block_len)) % n_wk for a in s])[:n_wk]
            ds.append(ps[ix].sum() / pc[ix].sum() - ref_s[ix].sum() / ref_c[ix].sum())
        out[p] = (float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975)))
    return out

# === Condition L4 + CR L7: Extend block-length sensitivity to ALL policies and ALL 3 metrics using PRIMARY SEED ===
sens_rows = []
for bl in (4, 13, 26, 52):
    # Use primary SEED for 13w so 13w CIs match table_bootstrap_vs_quarterly.csv identically (CR L7)
    s_use = SEED if bl == 13 else SEED + bl
    cb_mae = circ_boot_dict(w_mae, bl, seed=s_use)
    cb_mape = circ_boot_dict(w_mape, bl, seed=s_use)
    cb_log = circ_boot_dict(w_log, bl, seed=s_use)
    for p in POL + ["persistence"]:
        if p == REF: continue
        for met, cb, col in (("mae", cb_mae, "mae"), ("mape", cb_mape, "mape"), ("log_mae", cb_log, "mae_log")):
            lo, hi = cb[p]
            sens_rows.append(dict(block_weeks=bl, policy=p, metric=met,
                                  diff=float(RES.loc[p, col] - RES.loc[REF, col]),
                                  lo=lo, hi=hi, includes_zero=(lo <= 0 <= hi)))
SENS_DF = pd.DataFrame(sens_rows)
SENS_DF.to_csv(OUT / "table_bootstrap_block_sensitivity.csv", index=False)
all_inc_zero = bool(SENS_DF["includes_zero"].all())
n_sens_total = len(SENS_DF)
n_sens_inc_zero = int(SENS_DF["includes_zero"].sum())

# === Condition L5: Bootstrap all 9 Page-Hinkley grid settings (and 4 PSI) vs every13 ===
ref_mae_s, ref_mae_c = w_mae[REF]["sum"].values, w_mae[REF]["count"].values
grid_rows = []
for dm in (0.05, 0.1, 0.2):
    for lm in (3.0, 5.0, 8.0):
        pr, retr = simulate("pagehinkley", ph_d=dm * s0, ph_l=lm * s0)
        lvl = P_now * np.exp(pr); ae = np.abs(lvl - A_true)
        wa = wk_agg(ae)
        lo, hi = circ_boot_pair(wa["sum"].values, wa["count"].values, ref_mae_s, ref_mae_c, 13, seed=SEED)
        dmae = float(ae[ev].mean() - RES.loc[REF, "mae"])
        grid_rows.append(dict(detector="Page-Hinkley", param=f"d={dm}s0, l={lm}s0", retrains=len(retr),
                              mae=float(ae[ev].mean()), mape=float(100*(ae[ev]/A_true[ev]).mean()),
                              mae_log=float(np.abs(Y - pr)[ev].mean()),
                              dmae_vs_qtr=dmae, ci_lo=lo, ci_hi=hi, includes_zero=(lo <= 0 <= hi)))
for thr in (0.1, 0.2, 0.25, 0.3):
    pr, retr = simulate("psi", psi_thr=thr)
    lvl = P_now * np.exp(pr); ae = np.abs(lvl - A_true)
    wa = wk_agg(ae)
    lo, hi = circ_boot_pair(wa["sum"].values, wa["count"].values, ref_mae_s, ref_mae_c, 13, seed=SEED)
    dmae = float(ae[ev].mean() - RES.loc[REF, "mae"])
    grid_rows.append(dict(detector="PSI", param=f"thr={thr}", retrains=len(retr),
                          mae=float(ae[ev].mean()), mape=float(100*(ae[ev]/A_true[ev]).mean()),
                          mae_log=float(np.abs(Y - pr)[ev].mean()),
                          dmae_vs_qtr=dmae, ci_lo=lo, ci_hi=hi, includes_zero=(lo <= 0 <= hi)))
GRID = pd.DataFrame(grid_rows); GRID.to_csv(OUT / "table_threshold_grid.csv", index=False)

# === Condition L3: Kaplan-Meier median returning NaN ("not reached") when S(t) never falls to 0.5 ===
def km_medians(times, events):
    """Exact Kaplan-Meier estimator returning (smallest_t_median, midpoint_median)."""
    order = np.argsort(times)
    t_arr = np.asarray(times)[order]
    e_arr = np.asarray(events, dtype=int)[order]
    n_at_risk = len(t_arr)
    S = 1.0
    uniq_t = np.unique(t_arr)
    surv_t, surv_s = [], []
    for t in uniq_t:
        mask = (t_arr == t)
        d = int(e_arr[mask].sum())
        c = int((1 - e_arr[mask]).sum())
        if d > 0:
            S *= (1.0 - d / n_at_risk)
            surv_t.append(float(t))
            surv_s.append(float(S))
        n_at_risk -= (d + c)
    if not surv_s or min(surv_s) > 0.5 + 1e-12:
        return np.nan, np.nan
    # smallest t with S(t) <= 0.5
    idx_le = next(i for i, s in enumerate(surv_s) if s <= 0.5 + 1e-12)
    t_smallest = surv_t[idx_le]
    # midpoint convention: if S(t) == 0.5 exactly, average with next event time where S(t) < 0.5
    if abs(surv_s[idx_le] - 0.5) < 1e-12 and idx_le + 1 < len(surv_t):
        t_mid = 0.5 * (surv_t[idx_le] + surv_t[idx_le + 1])
    else:
        t_mid = t_smallest
    return float(t_smallest), float(t_mid)

nat_abs = pd.Series(np.abs(Y[ev])).groupby(WI[ev]).median()
SHOCK_W = set(nat_abs[nat_abs >= nat_abs.quantile(0.9)].index)
ep_raw = sorted(w for w in SHOCK_W if (w - 1) not in SHOCK_W)
ep_merged = [ep_raw[0]]
for w in ep_raw[1:]:
    if w - ep_merged[-1] > 2: ep_merged.append(w)

del_rows = []
for p in ("psi", "skill", "adwin", "pagehinkley", "every13"):
    rw = np.array(SIM[p][1])
    times, events, w8 = [], [], 0
    for o in ep_merged:
        after = rw[rw >= o]
        if len(after) > 0:
            d = int(after.min() - o)
            times.append(d); events.append(1)
            if d <= 8: w8 += 1
        else:
            times.append(int(LAST - o)); events.append(0)
    km_sm, km_mid = km_medians(times, events)
    del_rows.append(dict(policy=p, episodes=len(ep_merged), followed_by_retrain=int(sum(events)),
                         km_median_smallest=km_sm, km_median_midpoint=km_mid,
                         within_8=w8))
DEL = pd.DataFrame(del_rows); DEL.to_csv(OUT / "table_detection_delay.csv", index=False)

# === CR L10: Regenerate Fig 2 (fig_timeline.pdf) with en dash in "Page–Hinkley" legend ===
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})
nat_p = Pn[Pn.wi >= EVAL0].groupby("week")["p"].median()
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.plot(nat_p.index, nat_p.values, color="#3182bd", lw=1.0, label="national median price")
for w in sorted(SHOCK_W):
    dt = WEEKS[w]
    ax.axvspan(dt, dt + pd.Timedelta(days=7), color="#fee6ce", alpha=0.7, lw=0)
ymax = nat_p.max()
markers = [("pagehinkley", "Page\u2013Hinkley", "#d94801", "^", 1.04),
           ("skill", "skill-loss trigger", "#31a354", "v", 1.09),
           ("every13", "quarterly", "#636363", "|", 0.99)]
for p, lab, col, mk, mult in markers:
    dts = [WEEKS[w] for w in SIM[p][1] if w > EVAL0]
    ax.scatter(dts, [ymax * mult] * len(dts), color=col, marker=mk, s=16, label=lab, zorder=5)
ax.set_ylabel("modal price (Rs/quintal)"); ax.set_ylim(0, ymax * 1.16)
ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper left")
plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_timeline.pdf")
plt.savefig(OUT / "fig_timeline.png"); plt.close()

print(f"=== Block sensitivity: {n_sens_inc_zero}/{n_sens_total} CIs include zero (all={all_inc_zero}) ===")
print(SENS_DF[SENS_DF.policy == "persistence"].round(3).to_string(index=False))
print("\n=== Threshold Grid with 95% CIs ===")
print(GRID.round(3).to_string(index=False))
print("\n=== Kaplan-Meier Delays (12 merged episodes) ===")
print(DEL.to_string(index=False))
