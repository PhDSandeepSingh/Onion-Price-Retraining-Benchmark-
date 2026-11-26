import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md(r'''
# Paper 3 - When should a mandi price model be retrained? A data-validated MLOps benchmark on all-India onion markets
ICECRF-2027, Track 1, topic 8 (*AI deployment frameworks, MLOps, scalable AI systems*).
Data: *A Comprehensive Spatio-Temporal Dataset for Agricultural Price Forecasting* (Mendeley Data, DOI 10.17632/ds9jmxp9zy.1): Agmarknet onion prices and arrivals 2014-2024 joined with IMD weather.
''')
md(r'''
## 0. Fixed design decisions (pre-specified)
| Item | Decision |
|---|---|
| Validation gate | Rules G1-G6 (key uniqueness, non-positive price, min>max, modal outside [min,max], price > 20,000 Rs/qtl, lag semantics). Rows failing G2-G5 are dropped before modelling. |
| Leakage experiment | Row-level next-observation nowcast with the dataset's own engineered features vs clean market-day features; random 80/20 split vs temporal split (train <= 2021, test 2022-2024). |
| Forecasting panel | Market-week median of the market-day median modal price; markets with >= 80% weekly coverage 2015-2024. |
| Task | Predict log price 4 weeks ahead as a log return from week t; labels arrive 4 weeks later. |
| Model | HistGradientBoosting (300 iterations, learning rate 0.05, 31 leaves, L2 1.0), state as categorical. Fixed; no tuning. |
| Training window | Sliding 156 weeks of labelled data ending at the retraining week. |
| Evaluation | Every week from 2018-01-01 to the last week with a label; MAE of the implied price level (Rs/qtl); skill vs persistence. |
| Policies | static; periodic every 4 / 13 / 52 weeks; weekly; PSI input-drift trigger (max PSI > 0.2 on four features); skill-loss trigger (4-week model MAE > persistence MAE); ADWIN (delta 0.002) and Page-Hinkley on weekly model MAE. Triggered policies wait >= 4 weeks between retrains. |
| Shock weeks | Weeks whose national median absolute 4-week-ahead log return is in the top decile of the evaluation period. |
''')
code(r'''
import os, sys, json, time, glob, hashlib, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sklearn, scipy, matplotlib
SEED = 2027; H = 4; WIN = 156; COV_MIN = 0.80; PRICE_MAX = 20000; N_JOBS = int(os.environ.get("N_JOBS", 4))
MAX_SEC = float(os.environ.get("MAX_SECONDS", 1e9))
DATA = Path(os.environ.get("DATA_PATH", "data/onion_2014_2024.parquet"))   # in Colab: upload the parquet from Mendeley 10.17632/ds9jmxp9zy.1
OUT = Path("results"); (OUT / "models").mkdir(parents=True, exist_ok=True)
print({k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy).items()}, "| Python", sys.version.split()[0])
json.dump({DATA.name: hashlib.sha256(DATA.read_bytes()).hexdigest()}, open(OUT / "data_sha256.json", "w"), indent=2)
COLS = ["date", "state", "district", "market_key", "variety", "arrivals_tonnes", "min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl",
        "rain", "tmax", "tmin", "month", "dayofweek", "modal_price_lag_1", "modal_price_lag_7", "modal_price_lag_30",
        "modal_price_7d_mean", "modal_price_30d_mean", "arrivals_lag_1", "arrivals_7d_mean", "cum_rain_5m", "temp_avg_anomaly"]
raw = pd.read_parquet(DATA, columns=COLS).sort_values(["market_key", "date"], kind="stable").reset_index(drop=True)
print(raw.shape, raw.date.min().date(), raw.date.max().date(), raw.market_key.nunique(), "markets")
''')
md("## 1. Data-validation gate")
code(r'''
r = raw; n = len(r); g = r.groupby("market_key")
prev = g.date.shift(1); prev7 = g.date.shift(7)
rules = [
 ("G1", "duplicate key (market, variety, date)", r.duplicated(["market_key", "variety", "date"], keep=False)),
 ("G2", "modal price <= 0", r.modal_price_rs_qtl <= 0),
 ("G3", "min price > max price", r.min_price_rs_qtl > r.max_price_rs_qtl),
 ("G4", "modal price outside [min, max]", (r.modal_price_rs_qtl < r.min_price_rs_qtl) | (r.modal_price_rs_qtl > r.max_price_rs_qtl)),
 ("G5", f"modal price > {PRICE_MAX:,} Rs/qtl", r.modal_price_rs_qtl > PRICE_MAX),
 ("G6a", "lag_1 taken from the same date", (prev == r.date) & r.modal_price_lag_1.notna()),
 ("G6b", "lag_1 not from the previous calendar day", ((r.date - prev).dt.days != 1) & r.modal_price_lag_1.notna()),
 ("G6c", "lag_7 not from 7 calendar days earlier", ((r.date - prev7).dt.days != 7) & r.modal_price_lag_7.notna()),
]
GATE = pd.DataFrame([(c, d, int(m.sum()), m.mean()) for c, d, m in rules], columns=["rule", "check", "rows", "share"])
GATE.to_csv(OUT / "table_gate.csv", index=False)
bad = rules[1][2] | rules[2][2] | rules[3][2] | rules[4][2]
clean = r[~bad].copy(); clean["g6a"] = rules[5][2][~bad].values; print("rows dropped by G2-G5:", int(bad.sum()), f"({bad.mean():.2%})"); GATE
''')
md("## 2. Leakage experiment: offline score vs deployment-like score")
code(r'''
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
L = clean.copy(); L["y"] = np.log(L.modal_price_rs_qtl)
# clean market-day features: built only from strictly earlier dates
md_ = L.groupby(["market_key", "date"]).agg(p=("modal_price_rs_qtl", "median"), a=("arrivals_tonnes", "sum")).reset_index()
md_ = md_.sort_values(["market_key", "date"]); gg = md_.groupby("market_key")
md_["c_prev_p"] = gg.p.shift(1); md_["c_prev_gap"] = (md_.date - gg.date.shift(1)).dt.days
md_["c_mean7"] = gg.p.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()); md_["c_mean30"] = gg.p.transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
md_["c_prev_a"] = gg.a.shift(1)
L = L.merge(md_[["market_key", "date", "c_prev_p", "c_prev_gap", "c_mean7", "c_mean30", "c_prev_a"]], on=["market_key", "date"], how="left")
COMMON = ["rain", "tmax", "tmin", "month", "dayofweek", "cum_rain_5m", "temp_avg_anomaly"]
FEAT = {"provided": ["modal_price_lag_1", "modal_price_lag_7", "modal_price_lag_30", "modal_price_7d_mean", "modal_price_30d_mean", "arrivals_lag_1", "arrivals_7d_mean"] + COMMON,
        "clean": ["c_prev_p", "c_prev_gap", "c_mean7", "c_mean30", "c_prev_a"] + COMMON}
L = L.dropna(subset=FEAT["provided"] + FEAT["clean"]).reset_index(drop=True)
rng = np.random.default_rng(SEED); rnd_test = rng.random(len(L)) < 0.2; tmp_test = (L.date.dt.year >= 2022).values
LEAK = []
for fs in ("provided", "clean"):
    X = L[FEAT[fs]].astype("float32")
    for split, te in (("random", rnd_test), ("temporal", tmp_test)):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X[~te], L.y[~te])
        p = np.exp(m.predict(X[te])); a = L.modal_price_rs_qtl[te].values
        sd = L.g6a.values[te]; ape = np.abs(a - p) / a
        LEAK.append(dict(features=fs, split=split, n_test=int(te.sum()), mape=100 * ape.mean(), mae=np.abs(a - p).mean(),
                         n_same_date=int(sd.sum()), mape_same_date=100 * ape[sd].mean(), mape_other=100 * ape[~sd].mean()))
LEAK = pd.DataFrame(LEAK); LEAK.to_csv(OUT / "table_leakage.csv", index=False); LEAK.round(2)
''')
md("## 3. Weekly forecasting panel")
code(r'''
day = clean.groupby(["market_key", "date"]).agg(state=("state", "first"), p=("modal_price_rs_qtl", "median"), arr=("arrivals_tonnes", "sum"),
                                                rain=("rain", "mean"), tmax=("tmax", "mean")).reset_index()
day["week"] = day.date.dt.to_period("W-SUN").dt.start_time
wk = day.groupby(["market_key", "week"]).agg(state=("state", "first"), p=("p", "median"), arr=("arr", "sum"), rain=("rain", "sum"), tmax=("tmax", "mean")).reset_index()
WEEKS = pd.date_range("2015-01-05", "2024-12-30", freq="W-MON")
cov = wk[wk.week.isin(WEEKS)].groupby("market_key").week.nunique() / len(WEEKS)
MKTS = cov[cov >= COV_MIN].index.tolist()
idx = pd.MultiIndex.from_product([MKTS, WEEKS], names=["market_key", "week"])
Pn = wk.set_index(["market_key", "week"]).reindex(idx).reset_index()
Pn["state"] = Pn.groupby("market_key").state.transform(lambda s: s.ffill().bfill())
Pn["observed"] = Pn.p.notna()
Pn["lp"] = np.log(Pn.p); Pn["lp_f"] = Pn.groupby("market_key").lp.transform(lambda s: s.ffill(limit=2))   # features may carry forward <= 2 weeks
g = Pn.groupby("market_key")
for k in (1, 2, 4, 8, 13, 26, 52): Pn[f"r{k}"] = Pn.lp_f - g.lp_f.shift(k)
Pn["dev52"] = Pn.lp_f - g.lp_f.transform(lambda s: s.rolling(52, min_periods=26).mean())
Pn["la"] = np.log1p(Pn.arr); Pn["la_f"] = g.la.transform(lambda s: s.ffill(limit=2)); Pn["da4"] = Pn.la_f - g.la_f.shift(4)
Pn["rain4"] = g.rain.transform(lambda s: s.rolling(4, min_periods=1).sum()); Pn["tmax4"] = g.tmax.transform(lambda s: s.rolling(4, min_periods=1).mean())
woy = Pn.week.dt.isocalendar().week.astype(int); Pn["ws"] = np.sin(2 * np.pi * woy / 52.18); Pn["wc"] = np.cos(2 * np.pi * woy / 52.18)
nat = Pn.groupby("week")[["r1", "r4"]].median().rename(columns={"r1": "nat_r1", "r4": "nat_r4"}); Pn = Pn.merge(nat, left_on="week", right_index=True, how="left")
Pn["state_c"] = Pn.state.astype("category").cat.codes
Pn["y"] = g.lp.shift(-H) - Pn.lp                      # target only when both weeks are actually observed
FE = ["r1", "r2", "r4", "r8", "r13", "r26", "r52", "dev52", "la_f", "da4", "rain4", "tmax4", "ws", "wc", "nat_r1", "nat_r4", "state_c"]
Pn = Pn.sort_values(["week", "market_key"]).reset_index(drop=True)
Pn["wi"] = (Pn.week - WEEKS[0]).dt.days // 7
EVAL0 = int(np.where(WEEKS == pd.Timestamp("2018-01-01"))[0][0]); LAST = int(Pn[Pn.y.notna()].wi.max())
PANEL = {"markets": len(MKTS), "states": int(Pn.state.nunique()), "weeks": len(WEEKS), "eval_weeks": LAST - EVAL0 + 1,
         "label_rows": int(Pn.y.notna().sum()), "obs_share": float(Pn.observed.mean())}
json.dump(PANEL, open(OUT / "panel.json", "w"), indent=2); PANEL
''')
code(r'''
# One model per possible retraining week T (EVAL0..LAST): trained on labelled rows with target week <= T within the last WIN weeks,
# then used to predict every later week. Cached on disk so all policies share identical models (resumable).
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
XF = Pn[FE].astype("float32").values; Y = Pn.y.values; WI = Pn.wi.values; CAT = [FE.index("state_c")]
def fit_T(T):
    f = OUT / "models" / f"pred_T{T}.npy"
    if f.exists(): return
    tr = (WI + H <= T) & (WI + H > T - WIN) & ~np.isnan(Y) & ~np.isnan(XF).all(1)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                      categorical_features=CAT, random_state=SEED).fit(XF[tr], Y[tr])
    te = WI >= T
    np.save(f, m.predict(XF[te]).astype("float32"))
Ts = [T for T in range(EVAL0, LAST + 1) if not (OUT / "models" / f"pred_T{T}.npy").exists()]
print(len(Ts), "models to fit"); t0 = time.time()
for i in range(0, len(Ts), 4 * N_JOBS):
    if time.time() - t0 > MAX_SEC: print("time budget reached - re-run to resume"); break
    Parallel(n_jobs=N_JOBS)(delayed(fit_T)(T) for T in Ts[i:i + 4 * N_JOBS]); print(f"{min(i + 4*N_JOBS, len(Ts))}/{len(Ts)} {time.time()-t0:.0f}s", flush=True)
''')
md("## 4. Retraining policies under label delay")
code(r'''
start = {T: int(np.searchsorted(WI, T)) for T in range(EVAL0, LAST + 1)}      # rows are sorted by week
def pred_of(T, rows):  # predictions of model T for global row indices `rows` (all with week >= T)
    return np.load(OUT / "models" / f"pred_T{T}.npy", mmap_mode="r")[rows - start[T]]
weeks_rows = {w: np.arange(start[w], start[w + 1] if w + 1 in start else start[w] + (WI == w).sum()) for w in range(EVAL0, LAST + 1)}
lab = ~np.isnan(Y)

def psi(ref, cur, bins=10):
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 20: return 0.0
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    a = np.histogram(ref, q)[0] / len(ref) + 1e-4; b = np.histogram(cur, q)[0] / len(cur) + 1e-4
    return float(((b - a) * np.log(b / a)).sum())
PSI_F = [FE.index(c) for c in ("r4", "r13", "la_f", "dev52")]

class ADWIN:   # variance-based ADWIN (Bifet & Gavalda 2007); flags an increase in the mean of the stream
    def __init__(self, delta=0.002): self.d, self.w = delta, []
    def update(self, x):
        self.w.append(x); n = len(self.w)
        for c in range(5, n - 4):
            a, b = np.array(self.w[:c]), np.array(self.w[c:]); m = 1 / (1 / len(a) + 1 / len(b)); v = np.var(self.w); dd = self.d / n
            eps = np.sqrt(2 / m * v * np.log(2 / dd)) + 2 / (3 * m) * np.log(2 / dd)
            if b.mean() - a.mean() > eps: self.w = list(b); return True
        return False
class PageHinkley:   # one-sided (increase) Page-Hinkley test
    def __init__(self, delta, lam): self.delta, self.lam = delta, lam; self.reset()
    def reset(self): self.n, self.mean, self.cum, self.mn = 0, 0.0, 0.0, 0.0
    def update(self, x):
        self.n += 1; self.mean += (x - self.mean) / self.n; self.cum += x - self.mean - self.delta; self.mn = min(self.mn, self.cum)
        return self.cum - self.mn > self.lam

# scale for Page-Hinkley from the pre-evaluation period (persistence errors, no model involved)
pre = (WI < EVAL0) & lab; s0 = pd.Series(np.abs(Y[pre])).groupby(WI[pre]).mean().std()
PH_DELTA, PH_LAMBDA = 0.1 * s0, 5 * s0

def simulate(policy):
    T = EVAL0; retr = [EVAL0]; preds = np.full(len(Y), np.nan); hist = {}    # hist[w] = weekly abs error of the active model on labelled rows of week w
    det = ADWIN() if policy == "adwin" else PageHinkley(PH_DELTA, PH_LAMBDA) if policy == "pagehinkley" else None
    for w in range(EVAL0, LAST + 1):
        if w > EVAL0:   # decide whether to retrain at the start of week w, using labels known by now (target weeks <= w)
            go, gap = False, w - T
            if policy == "weekly": go = True
            elif policy.startswith("every"): go = gap >= int(policy[5:])
            elif gap >= 4 and policy == "psi":
                tr = (WI + H <= T) & (WI + H > T - WIN) & lab; cur = (WI > w - 4) & (WI <= w)
                go = max(psi(XF[tr, j], XF[cur, j]) for j in PSI_F) > 0.2
            elif policy in ("skill", "adwin", "pagehinkley"):
                kw = w - H                                 # newest week whose labels just arrived
                if kw >= EVAL0 and kw in hist:
                    e_m, e_p = hist[kw]
                    if policy == "skill":
                        ks = [k for k in range(kw - 3, kw + 1) if k in hist and k >= T]
                        go = gap >= 4 and len(ks) == 4 and np.mean([hist[k][0] for k in ks]) > np.mean([hist[k][1] for k in ks])
                    elif kw >= T:
                        fired = det.update(e_m); go = fired and gap >= 4
            if go:
                T = w; retr.append(w)
                if det is not None: det = ADWIN() if policy == "adwin" else PageHinkley(PH_DELTA, PH_LAMBDA)
        rows = weeks_rows[w]; preds[rows] = pred_of(T, rows)
        lr = rows[lab[rows]]
        if len(lr): hist[w] = (np.abs(Y[lr] - preds[lr]).mean(), np.abs(Y[lr]).mean())
    return preds, retr

POL = ["static", "every52", "every13", "every4", "weekly", "psi", "skill", "adwin", "pagehinkley"]
SIM = {p: simulate(p) for p in POL}
ev = (WI >= EVAL0) & lab; P_now = np.exp(Pn.lp.values); A_true = np.exp(Pn.lp.values + Y)
nat_abs = pd.Series(np.abs(Y[ev])).groupby(WI[ev]).median(); SHOCK_W = set(nat_abs[nat_abs >= nat_abs.quantile(0.9)].index)
shock = np.isin(WI, list(SHOCK_W)) & ev
rows = []
for p, (pr, retr) in SIM.items():
    lvl = P_now * np.exp(pr); e = np.abs(lvl - A_true); ep = np.abs(P_now - A_true)
    rows.append(dict(policy=p, retrains=len(retr), mae=e[ev].mean(), mape=100 * (e[ev] / A_true[ev]).mean(), skill=1 - e[ev].mean() / ep[ev].mean(),
                     mae_shock=e[shock].mean(), mae_normal=e[ev & ~shock].mean(), mae_log=np.abs(Y - pr)[ev].mean()))
RES = pd.DataFrame(rows); e_p = np.abs(P_now - A_true)
RES = pd.concat([RES, pd.DataFrame([dict(policy="persistence", retrains=0, mae=e_p[ev].mean(), mape=100 * (e_p[ev] / A_true[ev]).mean(), skill=0.0,
                                          mae_shock=e_p[shock].mean(), mae_normal=e_p[ev & ~shock].mean(), mae_log=np.abs(Y)[ev].mean())])], ignore_index=True)
RES.to_csv(OUT / "table_policies.csv", index=False)
json.dump({p: [int(x) for x in r_[1]] for p, r_ in SIM.items()}, open(OUT / "retrain_weeks.json", "w"))
RES.round(3)
''')
code(r'''
# Uncertainty: moving-block bootstrap over evaluation weeks (13-week blocks) for MAE differences vs the quarterly schedule
REF = "every13"; wk_ids = np.arange(EVAL0, LAST + 1); B = 1000; rng = np.random.default_rng(SEED)
werr = {p: pd.Series(np.abs(P_now * np.exp(SIM[p][0]) - A_true)[ev]).groupby(WI[ev]).agg(["sum", "count"]).reindex(wk_ids).fillna(0) for p in POL}
werr["persistence"] = pd.Series(np.abs(P_now - A_true)[ev]).groupby(WI[ev]).agg(["sum", "count"]).reindex(wk_ids).fillna(0)
nb = int(np.ceil(len(wk_ids) / 13)); starts = rng.integers(0, len(wk_ids) - 13, size=(B, nb))
BS = []
for p in POL + ["persistence"]:
    if p == REF: continue
    d = []
    for s in starts:
        ix = np.concatenate([np.arange(a, a + 13) for a in s])[:len(wk_ids)]
        mp = werr[p]["sum"].values[ix].sum() / werr[p]["count"].values[ix].sum(); mr = werr[REF]["sum"].values[ix].sum() / werr[REF]["count"].values[ix].sum()
        d.append(mp - mr)
    pt = RES.set_index("policy").loc[p, "mae"] - RES.set_index("policy").loc[REF, "mae"]
    BS.append(dict(policy=p, dmae=pt, lo=np.quantile(d, .025), hi=np.quantile(d, .975)))
BS = pd.DataFrame(BS); BS.to_csv(OUT / "table_bootstrap_vs_quarterly.csv", index=False)
# detection delay: first retrain at or after the onset of each shock episode
ep_on = sorted(w for w in SHOCK_W if (w - 1) not in SHOCK_W); DEL = []
for p in ("psi", "skill", "adwin", "pagehinkley", "every13"):
    rw = np.array(SIM[p][1]); dl = [int(rw[rw >= o].min() - o) if (rw >= o).any() else np.nan for o in ep_on]
    DEL.append(dict(policy=p, episodes=len(ep_on), median_delay=np.nanmedian(dl), detected_within_8=int(np.sum(np.array(dl) <= 8))))
DEL = pd.DataFrame(DEL); DEL.to_csv(OUT / "table_detection_delay.csv", index=False)
json.dump({"shock_weeks": len(SHOCK_W), "episodes": len(ep_on), "episode_onsets": [str(WEEKS[w].date()) for w in ep_on], "ph_delta": PH_DELTA, "ph_lambda": PH_LAMBDA},
          open(OUT / "shocks.json", "w"), indent=2)
display(BS.round(2)); DEL
''')
code(r'''
import matplotlib
if "google.colab" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})
NAMES = {"static": "static", "every52": "yearly", "every13": "quarterly", "every4": "monthly", "weekly": "weekly", "psi": "PSI trigger",
         "skill": "skill-loss trigger", "adwin": "ADWIN", "pagehinkley": "Page-Hinkley"}
fig, ax = plt.subplots(figsize=(4.6, 2.9)); R_ = RES[RES.policy.isin(POL)]
ax.scatter(R_.retrains, R_.mae, color=["#e6550d" if p in ("psi", "skill", "adwin", "pagehinkley") else "#3182bd" for p in R_.policy], s=18, zorder=3)
lab = R_.groupby(["retrains", "mae"]).policy.apply(lambda v: " = ".join(NAMES[x] for x in v)).reset_index()
OFF = {"quarterly": (-4, -9), "skill-loss trigger": (4, 2)}
for _, r_ in lab.iterrows():
    dx, dy = OFF.get(r_.policy, (4, 2)); ha = "right" if dx < 0 else "left"
    ax.annotate(r_.policy, (r_.retrains, r_.mae), fontsize=6.5, xytext=(dx, dy), textcoords="offset points", ha=ha)
ax.axhline(RES.set_index("policy").loc["persistence", "mae"], color="grey", ls="--", lw=.8); ax.text(1.1, RES.set_index("policy").loc["persistence", "mae"], " persistence", fontsize=6.5, va="bottom", color="grey")
ax.set_xscale("log"); ax.set_xlabel("number of models trained (log scale)"); ax.set_ylabel("MAE (Rs/quintal)")
plt.tight_layout(); plt.savefig(OUT / "fig_pareto.pdf"); plt.savefig(OUT / "fig_pareto.png"); plt.show()

fig, ax = plt.subplots(figsize=(4.9, 2.6)); natp = Pn.groupby("week").p.median()
ax.plot(natp.index, natp.values, color="k", lw=.8)
for w in SHOCK_W: ax.axvspan(WEEKS[w], WEEKS[w] + pd.Timedelta(days=7), color="#fdae6b", alpha=.5, lw=0)
for j, (p, col) in enumerate((("psi", "#3182bd"), ("skill", "#31a354"), ("adwin", "#756bb1"), ("pagehinkley", "#e6550d"))):
    for w in SIM[p][1][1:]: ax.plot(WEEKS[w], natp.max() * (1.02 + 0.05 * j), "|", color=col, ms=4)
    ax.text(WEEKS[0], natp.max() * (1.02 + 0.05 * j), NAMES[p], fontsize=6, color=col, va="center", ha="left")
ax.axvline(WEEKS[EVAL0], color="grey", lw=.7, ls=":"); ax.set_ylabel("median price (Rs/quintal)"); ax.set_ylim(top=natp.max() * 1.25)
plt.tight_layout(); plt.savefig(OUT / "fig_timeline.pdf"); plt.savefig(OUT / "fig_timeline.png"); plt.show()
json.dump({"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED,
           "packages": {k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy).items()}}, open(OUT / "environment.json", "w"), indent=2)
''')
nb.cells = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "Paper3_OnionRetraining.ipynb"); print("ok", len(C))
