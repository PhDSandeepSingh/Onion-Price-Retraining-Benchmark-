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

# === R2-L4: G6 lag audit BY VALUE under 3 hypotheses (market row, market-variety row, calendar day) with non-null denominators ===
g_mkt = raw.groupby("market_key")
g_mv = raw.groupby(["market_key", "variety"])
# Calendar-day lookup (median per market-date)
cal_lk = raw.groupby(["market_key", "date"])["modal_price_rs_qtl"].median()
g6_rows = []
for lag_k, col in ((1, "modal_price_lag_1"), (7, "modal_price_lag_7")):
    valid = raw[col].notna()
    n_nn = int(valid.sum())
    vals = raw.loc[valid, col].values
    # (a) k rows earlier for market
    p_mkt = g_mkt["modal_price_rs_qtl"].shift(lag_k).loc[valid].values
    m_mkt = float(np.mean(np.isclose(vals, p_mkt, rtol=1e-4, atol=1e-2, equal_nan=False)))
    # (b) k rows earlier for market-variety
    p_mv = g_mv["modal_price_rs_qtl"].shift(lag_k).loc[valid].values
    m_mv = float(np.mean(np.isclose(vals, p_mv, rtol=1e-4, atol=1e-2, equal_nan=False)))
    # (c) k calendar days earlier
    prev_dt = raw.loc[valid, "date"] - pd.Timedelta(days=lag_k)
    idx_c = pd.MultiIndex.from_arrays([raw.loc[valid, "market_key"], prev_dt])
    p_cal = cal_lk.reindex(idx_c).values
    m_cal = float(np.mean(np.isclose(vals, p_cal, rtol=1e-4, atol=1e-2, equal_nan=False)))
    # Date-based shares over non-null denominator
    prev_date = g_mkt["date"].shift(lag_k).loc[valid]
    dt_diff = (raw.loc[valid, "date"] - prev_date).dt.days
    same_dt = float((dt_diff == 0).mean()) if lag_k == 1 else np.nan
    not_k_days = float((dt_diff != lag_k).mean())
    g6_rows.append(dict(lag=col, n_nonnull=n_nn, match_mkt_row=m_mkt, match_mkt_var_row=m_mv,
                        match_cal_day=m_cal, date_same_day=same_dt, date_not_k_days=not_k_days))
G6 = pd.DataFrame(g6_rows); G6.to_csv(OUT / "table_g6_value_audit.csv", index=False)

# Fix Rs/qtl -> Rs/quintal in table_gate.csv and use non-null denominators for G6a/b/c
GATE = pd.read_csv(OUT / "table_gate.csv")
GATE["check"] = GATE["check"].str.replace("Rs/qtl", "Rs/quintal", regex=False)
for rule, lag_col, field in (("G6a", "modal_price_lag_1", "date_same_day"),
                             ("G6b", "modal_price_lag_1", "date_not_k_days"),
                             ("G6c", "modal_price_lag_7", "date_not_k_days")):
    val = G6.set_index("lag").loc[lag_col, field]
    GATE.loc[GATE.rule == rule, "share"] = val
GATE.to_csv(OUT / "table_gate.csv", index=False)

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

# Average window size & baseline log-error
win_sizes = [int(((WI + H <= T) & (WI + H > T - WIN) & lab).sum()) for T in range(EVAL0, LAST + 1)]
mean_win_rows = int(round(np.mean(win_sizes)))

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

# === R2-L1: Circular block bootstrap (fixing off-by-one!) for MAE, MAPE, Log-MAE + block sensitivity (4/13/26/52) ===
REF = "every13"; wk_ids = np.arange(EVAL0, LAST + 1); n_wk = len(wk_ids); B = 1000
def wk_agg(vals):
    return pd.Series(vals[ev]).groupby(WI[ev]).agg(["sum", "count"]).reindex(wk_ids).fillna(0)

w_mae, w_mape, w_log = {}, {}, {}
for p in POL:
    lvl = P_now * np.exp(SIM[p][0]); ae = np.abs(lvl - A_true)
    w_mae[p] = wk_agg(ae); w_mape[p] = wk_agg(100.0 * ae / A_true); w_log[p] = wk_agg(np.abs(Y - SIM[p][0]))
ae_p = np.abs(P_now - A_true)
w_mae["persistence"] = wk_agg(ae_p); w_mape["persistence"] = wk_agg(100.0 * ae_p / A_true); w_log["persistence"] = wk_agg(np.abs(Y))

def circ_boot(wdict, block_len, seed=SEED):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_wk / block_len))
    starts = rng.integers(0, n_wk, size=(B, nb))  # fixed: covers 0..n_wk-1 circularly
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

ci_mae13 = circ_boot(w_mae, 13)
ci_mape13 = circ_boot(w_mape, 13)
ci_log13 = circ_boot(w_log, 13)
bs_rows = []
for p in POL + ["persistence"]:
    if p == REF: continue
    bs_rows.append(dict(
        policy=p,
        dmae=float(RES.loc[p, "mae"] - RES.loc[REF, "mae"]), lo=ci_mae13[p][0], hi=ci_mae13[p][1],
        dmape=float(RES.loc[p, "mape"] - RES.loc[REF, "mape"]), mape_lo=ci_mape13[p][0], mape_hi=ci_mape13[p][1],
        dlog=float(RES.loc[p, "mae_log"] - RES.loc[REF, "mae_log"]), log_lo=ci_log13[p][0], log_hi=ci_log13[p][1],
    ))
BS_DF = pd.DataFrame(bs_rows); BS_DF.to_csv(OUT / "table_bootstrap_vs_quarterly.csv", index=False)

# Block-length sensitivity (4, 13, 26, 52 weeks) for MAE
sens_rows = []
for bl in (4, 13, 26, 52):
    cb = circ_boot(w_mae, bl, seed=SEED + bl)
    for p in ("static", "weekly", "pagehinkley", "persistence"):
        sens_rows.append(dict(block_weeks=bl, policy=p, dmae=float(RES.loc[p, "mae"] - RES.loc[REF, "mae"]),
                              lo=cb[p][0], hi=cb[p][1], includes_zero=(cb[p][0] <= 0 <= cb[p][1])))
pd.DataFrame(sens_rows).to_csv(OUT / "table_bootstrap_block_sensitivity.csv", index=False)

# === R2-L2: Threshold grid for Page-Hinkley (3x3) and PSI (4) on cached predictions ===
grid_rows = []
for dm in (0.05, 0.1, 0.2):
    for lm in (3.0, 5.0, 8.0):
        pr, retr = simulate("pagehinkley", ph_d=dm * s0, ph_l=lm * s0)
        lvl = P_now * np.exp(pr); ae = np.abs(lvl - A_true)
        grid_rows.append(dict(detector="Page-Hinkley", param=f"d={dm}s0, l={lm}s0", retrains=len(retr),
                              mae=float(ae[ev].mean()), mape=float(100*(ae[ev]/A_true[ev]).mean()),
                              mae_log=float(np.abs(Y - pr)[ev].mean())))
for thr in (0.1, 0.2, 0.25, 0.3):
    pr, retr = simulate("psi", psi_thr=thr)
    lvl = P_now * np.exp(pr); ae = np.abs(lvl - A_true)
    grid_rows.append(dict(detector="PSI", param=f"thr={thr}", retrains=len(retr),
                          mae=float(ae[ev].mean()), mape=float(100*(ae[ev]/A_true[ev]).mean()),
                          mae_log=float(np.abs(Y - pr)[ev].mean())))
GRID = pd.DataFrame(grid_rows); GRID.to_csv(OUT / "table_threshold_grid.csv", index=False)

# === R2-L3: Censored median delay on 12 merged episodes (<=2 weeks apart) ===
nat_abs = pd.Series(np.abs(Y[ev])).groupby(WI[ev]).median()
SHOCK_W = set(nat_abs[nat_abs >= nat_abs.quantile(0.9)].index)
ep_raw = sorted(w for w in SHOCK_W if (w - 1) not in SHOCK_W)
ep_merged = [ep_raw[0]]
for w in ep_raw[1:]:
    if w - ep_merged[-1] > 2: ep_merged.append(w)

del_rows = []
for p in ("psi", "skill", "adwin", "pagehinkley", "every13"):
    rw = np.array(SIM[p][1])
    d_cens, n_det, w8 = [], 0, 0
    for o in ep_merged:
        after = rw[rw >= o]
        if len(after) > 0:
            d = int(after.min() - o); n_det += 1
            if d <= 8: w8 += 1
            d_cens.append(d)
        else:
            d_cens.append(int(LAST - o))  # right-censored at end of evaluation
    del_rows.append(dict(policy=p, episodes=len(ep_merged), detected=n_det,
                         median_delay_censored=float(np.median(d_cens)),
                         detected_within_8=w8))
DEL = pd.DataFrame(del_rows); DEL.to_csv(OUT / "table_detection_delay.csv", index=False)

# Save Round 2 summary JSON for gen_tables.py
sh_old = json.load(open(OUT / "shocks.json"))
sh_old["merged_episodes"] = len(ep_merged)
sh_old["mean_window_rows"] = mean_win_rows
sh_old["s0_pre_eval"] = s0
sh_old["ord_worse_pct"] = float(100 * (RES.loc["every13", "mae_normal"] / RES.loc["persistence", "mae_normal"] - 1.0))
json.dump(sh_old, open(OUT / "shocks.json", "w"), indent=2)

# === R2-L7: Regenerate Fig 1 (fig_pareto.pdf) with CI bars, "ADWIN (ours)", and en-dash "Page–Hinkley" ===
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})
NAMES = {"static": "static", "every52": "yearly", "every13": "quarterly", "every4": "monthly", "weekly": "weekly",
         "psi": "PSI trigger", "skill": "skill-loss trigger", "adwin": "ADWIN (ours)", "pagehinkley": "Page\u2013Hinkley"}
fig, ax = plt.subplots(figsize=(4.8, 3.0))
R_ = RES.loc[POL].reset_index()
bs_map = BS_DF.set_index("policy")
q_mae = float(RES.loc["every13", "mae"])
for _, r_ in R_.iterrows():
    p = r_.policy
    col = "#e6550d" if p in ("psi", "skill", "adwin", "pagehinkley") else "#3182bd"
    ax.scatter(r_.retrains, r_.mae, color=col, s=20, zorder=4)
    if p != "every13":
        lo_err = max(0.0, r_.mae - (q_mae + bs_map.loc[p, "lo"]))
        hi_err = max(0.0, (q_mae + bs_map.loc[p, "hi"]) - r_.mae)
        ax.errorbar(r_.retrains, r_.mae, yerr=[[lo_err], [hi_err]], fmt="none", ecolor=col, alpha=0.45, elinewidth=1.0, capsize=2, zorder=2)
lab_df = R_.groupby(["retrains", "mae"]).policy.apply(lambda v: " = ".join(NAMES[x] for x in v)).reset_index()
OFF = {"quarterly": (-5, -10), "skill-loss trigger": (5, 3), "Page\u2013Hinkley": (5, -9)}
for _, r_ in lab_df.iterrows():
    dx, dy = OFF.get(r_.policy, (5, 3)); ha = "right" if dx < 0 else "left"
    ax.annotate(r_.policy, (r_.retrains, r_.mae), fontsize=6.5, xytext=(dx, dy), textcoords="offset points", ha=ha)
ax.axhline(RES.loc["persistence", "mae"], color="grey", ls="--", lw=.8)
ax.text(1.1, RES.loc["persistence", "mae"] + 1.0, " persistence", fontsize=6.5, va="bottom", color="grey")
ax.set_xscale("log"); ax.set_xlabel("number of models trained (log scale)"); ax.set_ylabel("MAE (Rs/quintal)")
plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_pareto.pdf")
plt.savefig(OUT / "fig_pareto.png"); plt.close()

print("=== G6 value audit ===")
print(G6.round(3).to_string(index=False))
print("\n=== Circular Bootstrap (13w) ===")
print(BS_DF.round(3).to_string(index=False))
print("\n=== Block sensitivity ===")
print(pd.DataFrame(sens_rows).round(1).to_string(index=False))
print("\n=== Threshold Grid ===")
print(GRID.round(3).to_string(index=False))
print("\n=== Censored Delays (12 merged episodes) ===")
print(DEL.to_string(index=False))
