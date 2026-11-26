import pandas as pd, numpy as np, json

R = "results/"
G = pd.read_csv(R + "table_gate.csv")
LK = pd.read_csv(R + "table_leakage.csv")
P = pd.read_csv(R + "table_policies.csv").set_index("policy")
BS = pd.read_csv(R + "table_bootstrap_vs_quarterly.csv").set_index("policy")
BSB = pd.read_csv(R + "table_bootstrap_block_sensitivity.csv")
TG = pd.read_csv(R + "table_threshold_grid.csv")
DL = pd.read_csv(R + "table_detection_delay.csv").set_index("policy")
G6V = pd.read_csv(R + "table_g6_value_audit.csv").set_index("lag")
pan = json.load(open(R + "panel.json"))
sh = json.load(open(R + "shocks.json"))
env = json.load(open(R + "environment.json"))

mac = {}
def M(k, v): mac[k] = v
f0 = lambda x: f"{x:,.0f}"
f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"

M("NRows", "965,314")
M("NMarketsRaw", "818")
g = G.set_index("rule")
RN = {"G1": "One", "G2": "Two", "G3": "Three", "G4": "Four", "G5": "Five", "G6a": "SixA", "G6b": "SixB", "G6c": "SixC"}
for r in g.index:
    M("Gate" + RN[r], f0(g.loc[r, "rows"]))
    M("GatePct" + RN[r], f1(100 * g.loc[r, "share"]))

# Value-based G6 audit (non-null denominators)
for lag_col, tag in [("modal_price_lag_1", "LagOne"), ("modal_price_lag_7", "LagSeven")]:
    row = G6V.loc[lag_col]
    M("G" + tag + "N", f0(row["n_nonnull"]))
    M("G" + tag + "Mkt", f1(100 * row["match_mkt_row"]))
    M("G" + tag + "Mv", f1(100 * row["match_mkt_var_row"]))
    M("G" + tag + "Cal", f1(100 * row["match_cal_day"]))
    M("G" + tag + "NotK", f1(100 * row["date_not_k_days"]))
    if pd.notna(row["date_same_day"]):
        M("G" + tag + "Same", f1(100 * row["date_same_day"]))

raw = pd.read_parquet("data/onion_2014_2024.parquet", columns=["modal_price_rs_qtl", "min_price_rs_qtl", "max_price_rs_qtl"])
bad = (
    (raw.modal_price_rs_qtl <= 0)
    | (raw.min_price_rs_qtl > raw.max_price_rs_qtl)
    | (raw.modal_price_rs_qtl < raw.min_price_rs_qtl)
    | (raw.modal_price_rs_qtl > raw.max_price_rs_qtl)
    | (raw.modal_price_rs_qtl > 20000)
)
M("NDropped", f0(bad.sum()))
M("PctDropped", f2(100 * bad.mean()))
M("NMarkets", str(pan["markets"]))
M("NStates", str(pan["states"]))
M("NWeeks", str(pan["weeks"]))
M("NEval", str(pan["eval_weeks"]))
M("NLabel", f0(pan["label_rows"]))
M("ObsShare", f1(100 * pan["obs_share"]))
M("NShockWeeks", str(sh["shock_weeks"]))
M("NEpisodes", str(sh["episodes"]))
M("NMergedEp", str(sh["merged_episodes"]))
M("MeanWinRows", f0(sh["mean_window_rows"]))
M("OrdWorsePct", f1(sh["ord_worse_pct"]))
M("SZeroPre", f"{sh['s0_pre_eval']:.4f}")
M("MeanLogErr", f2(P.loc["every13", "mae_log"]))

lk = LK.set_index(["features", "split"])
for fs in ("provided", "clean"):
    for sp in ("random", "temporal"):
        t = fs.capitalize() + sp.capitalize()
        M("Mape" + t, f2(lk.loc[(fs, sp), "mape"]))
        M("MapeSame" + t, f1(lk.loc[(fs, sp), "mape_same_date"]))
M("NSameTemporal", f0(lk.loc[("provided", "temporal"), "n_same_date"]))

NM = {
    "static": "Static", "every52": "Yearly", "every13": "Quarterly", "every4": "Monthly",
    "weekly": "Weekly", "psi": "PSI", "skill": "Skill", "adwin": "Adwin",
    "pagehinkley": "PH", "persistence": "Persist"
}
for p in P.index:
    t = NM[p]
    M("Mae" + t, f1(P.loc[p, "mae"]))
    M("Mods" + t, str(int(P.loc[p, "retrains"])))
    M("Skill" + t, f1(100 * P.loc[p, "skill"]))
    M("MaeShock" + t, f0(P.loc[p, "mae_shock"]))
    M("MaeNormal" + t, f0(P.loc[p, "mae_normal"]))
    M("Mape" + t + "Pol", f2(P.loc[p, "mape"]))
    M("LogMae" + t, f"{P.loc[p, 'mae_log']:.4f}")

for p in BS.index:
    t = NM[p]
    M("D" + t, f1(BS.loc[p, "dmae"]))
    M("Lo" + t, f1(BS.loc[p, "lo"]))
    M("Hi" + t, f1(BS.loc[p, "hi"]))
    M("DMape" + t, f2(BS.loc[p, "dmape"]))
    M("MapeLo" + t, f2(BS.loc[p, "mape_lo"]))
    M("MapeHi" + t, f2(BS.loc[p, "mape_hi"]))
    M("DLog" + t, f3(BS.loc[p, "dlog"]))
    M("LogLo" + t, f3(BS.loc[p, "log_lo"]))
    M("LogHi" + t, f3(BS.loc[p, "log_hi"]))

# Block-length sensitivity (all 9 policies x 3 metrics x 4 block lengths = 108 CIs)
M("SensTotal", str(len(BSB)))
M("SensIncZero", str(int(BSB["includes_zero"].sum())))
for blk, tag in [(4, "Four"), (13, "Thirteen"), (26, "TwentySix"), (52, "FiftyTwo")]:
    sub = BSB[(BSB["block_weeks"] == blk) & (BSB["policy"] == "persistence") & (BSB["metric"] == "mae")].iloc[0]
    M("BsPers" + tag + "Lo", f1(sub["lo"]))
    M("BsPers" + tag + "Hi", f1(sub["hi"]))

# Threshold grid summary macros + bootstrap CIs (Condition L5)
phg = TG[TG["detector"] == "Page-Hinkley"]
M("PHGridMinMod", str(int(phg["retrains"].min())))
M("PHGridMaxMod", str(int(phg["retrains"].max())))
M("PHGridMinMae", f1(phg["mae"].min()))
M("PHGridMaxMae", f1(phg["mae"].max()))
M("PHGridMinMape", f2(phg["mape"].min()))
M("PHGridMaxMape", f2(phg["mape"].max()))
M("PHGridTotal", str(len(phg)))
M("PHGridIncZero", str(int(phg["includes_zero"].sum())))
ph_best = phg.loc[phg["mae"].idxmin()]
M("PHBestMod", str(int(ph_best["retrains"])))
M("PHBestMae", f1(ph_best["mae"]))
M("PHBestMape", f2(ph_best["mape"]))
M("PHBestDmae", f1(ph_best["dmae_vs_qtr"]))
M("PHBestLo", f1(ph_best["ci_lo"]))
M("PHBestHi", f1(ph_best["ci_hi"]))

# Kaplan-Meier delay macros (Condition L3: returns "not reached" when S(t) never falls to 0.5)
for p in DL.index:
    t = NM[p]
    val_mid = DL.loc[p, "km_median_midpoint"]
    val_sm = DL.loc[p, "km_median_smallest"]
    M("Delay" + t, "not reached" if pd.isna(val_mid) else f1(val_mid))
    M("DelaySmall" + t, "not reached" if pd.isna(val_sm) else f1(val_sm))
    M("Det" + t, str(int(DL.loc[p, "within_8"])))
    M("DetEver" + t, str(int(DL.loc[p, "followed_by_retrain"])))

pol = P.drop(index="persistence")
M("SkillMin", f1(100 * pol.skill.min()))
M("SkillMax", f1(100 * pol.skill.max()))
M("ShockGainQuarterly", f0(100 * (1 - P.loc["every13", "mae_shock"] / P.loc["persistence", "mae_shock"])))
M("PHdelta", f"{sh['ph_delta']:.4f}")
M("PHlambda", f"{sh['ph_lambda']:.3f}")
for k in ["sklearn", "numpy", "pandas", "scipy"]:
    M("Ver" + k.capitalize(), env["packages"][k])
M("VerPython", env["python"])

open("manuscript/numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))

# Table 1: Data-validation gate (Rs/quintal in G5 and non-null denominators for G6a-c)
L = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Data-validation gate on the " + mac["NRows"] + r" raw rows. Rules G2--G5 remove invalid price records before modelling; G6a--G6c report date-alignment shares over non-null lag values ($n = " + mac["GLagOneN"] + r"$ for \texttt{lag\_1}, $n = " + mac["GLagSevenN"] + r"$ for \texttt{lag\_7}).}",
    r"\label{tab:gate}",
    r"\scriptsize",
    r"\begin{tabularx}{\textwidth}{@{}l X r r@{}}",
    r"\toprule",
    r"Rule & Check & Rows & Share (\%) \\",
    r"\midrule",
]
def texchk(c):
    return (
        c.replace("lag_1", "\\texttt{lag\\_1}")
         .replace("lag_7", "\\texttt{lag\\_7}")
         .replace("<=", "$\\le$")
         .replace(">", "$>$")
         .replace("[", "$[$")
         .replace("]", "$]$")
    )
for _, r in G.iterrows():
    if r.rule == "G6a":
        share_pct = 100 * float(G6V.loc["modal_price_lag_1", "date_same_day"])
    elif r.rule == "G6b":
        share_pct = 100 * float(G6V.loc["modal_price_lag_1", "date_not_k_days"])
    elif r.rule == "G6c":
        share_pct = 100 * float(G6V.loc["modal_price_lag_7", "date_not_k_days"])
    else:
        share_pct = 100 * float(r.share)
    L.append(f"{r.rule} & {texchk(r.check)} & {int(r.rows):,} & {share_pct:.2f} \\\\")
L += [
    r"\bottomrule",
    r"\end{tabularx}",
    r"\smallskip\raggedright\scriptsize Source: authors' audit of Shelar, R., Hajare, A., and Mangarule, S. (Mendeley Data V1, 2025, CC BY 4.0)~\cite{onion2025dataset}.",
    r"\end{table}",
]
open("manuscript/tab_gate.tex", "w").write("\n".join(L))

# Table 2: Leakage ablation
L = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Leakage ablation: next-observation nowcast MAPE (\%) with provided lag features versus strictly earlier market-day features under random and temporal splits. Same-date rows = rows whose \texttt{lag\_1} shares the same date as another variety (G6a). Both arms used early stopping (random 10\% internal validation split, \texttt{n\_iter\_no\_change=10}).}",
    r"\label{tab:leak}",
    r"\scriptsize",
    r"\begin{tabularx}{\textwidth}{@{}X r r r r@{}}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Random split} & \multicolumn{2}{c}{Temporal split} \\ \cmidrule(lr){2-3}\cmidrule(l){4-5}",
    r"Features & All rows & Same-date rows & All rows & Same-date rows \\",
    r"\midrule",
]
for fs, lab in (("provided", "Provided (pre-computed lags)"), ("clean", "Rebuilt (strictly earlier days)")):
    L.append(f"{lab} & {lk.loc[(fs,'random'),'mape']:.2f} & {lk.loc[(fs,'random'),'mape_same_date']:.1f} & {lk.loc[(fs,'temporal'),'mape']:.2f} & {lk.loc[(fs,'temporal'),'mape_same_date']:.1f} \\\\")
L += [
    r"\bottomrule",
    r"\end{tabularx}",
    r"\smallskip\raggedright\scriptsize Source: authors' ablation on Shelar, R. et al.~\cite{onion2025dataset}.",
    r"\end{table}",
]
open("manuscript/tab_leak.tex", "w").write("\n".join(L))

# Table 3: Retraining policies with circular block bootstrap CIs (MAE, MAPE, Log-MAE)
L = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Retraining policies over " + mac["NEval"] + r" evaluation weeks (" + mac["NMarkets"] + r" markets, 4-week horizon). MAE in Rs/quintal; MAPE in \%; Log-MAE on log-price scale; Shock MAE is conditional on realised top-decile 4-week moves. $\Delta$ MAE shows paired difference vs.\ quarterly schedule with 13-week circular block bootstrap 95\% CI (paired 95\% CIs for $\Delta$ MAPE and $\Delta$ Log-MAE also include zero for every policy; see Sect.~\ref{sec:results}).}",
    r"\label{tab:pol}",
    r"\scriptsize",
    r"\begin{tabularx}{\textwidth}{@{}l r r r r r r@{}}",
    r"\toprule",
    r"Policy & Models & MAE & MAPE & Log-MAE & Shock MAE & $\Delta$ MAE vs.\ Qtr [95\% CI] \\",
    r"\midrule",
]
LAB = {
    "static": "Static (never)", "every52": "Every 52 weeks", "every13": "Every 13 weeks",
    "every4": "Every 4 weeks", "weekly": "Every week", "psi": "PSI input drift$^*$",
    "skill": "Skill loss", "adwin": r"ADWIN (ours)$^\dagger$", "pagehinkley": "Page--Hinkley",
    "persistence": "Persistence (0 models)"
}
for p in ["static", "every52", "every13", "every4", "weekly", "psi", "skill", "adwin", "pagehinkley", "persistence"]:
    r = P.loc[p]
    d = "reference" if p == "every13" else f"${BS.loc[p,'dmae']:+.1f}$ [${BS.loc[p,'lo']:.1f}$, ${BS.loc[p,'hi']:.1f}$]"
    if p == "persistence":
        L.append(r"\midrule")
    L.append(f"{LAB[p]} & {int(r.retrains)} & {r.mae:.1f} & {r.mape:.2f} & {r.mae_log:.4f} & {r.mae_shock:.0f} & {d} \\\\")
L += [
    r"\bottomrule",
    r"\end{tabularx}",
    r"\smallskip\raggedright\scriptsize $^*$Identical to Every 4 weeks under this 4-vs-156-week configuration. $^\dagger$Identical to Static. Source: authors' benchmark on Shelar, R. et al.~\cite{onion2025dataset}.",
    r"\end{table}",
]
open("manuscript/tab_pol.tex", "w").write("\n".join(L))

# Table 4: Threshold sensitivity grid with Rs/quintal unit (Condition L2) and 95% CIs (Condition L5)
L = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Threshold sensitivity grid replayed on cached weekly predictions: Page--Hinkley ($\delta \in \{0.05, 0.1, 0.2\} s_0$, $\lambda \in \{3, 5, 8\} s_0$) and PSI ($\text{threshold} \in \{0.10, 0.20, 0.25, 0.30\}$). All 9 Page--Hinkley and 4 PSI 13-week circular block-bootstrap 95\% CIs for $\Delta$ MAE vs.\ quarterly include zero.}",
    r"\label{tab:grid}",
    r"\scriptsize",
    r"\begin{tabularx}{\textwidth}{@{}l X r r r r r@{}}",
    r"\toprule",
    r"Detector & Setting & Models & MAE (Rs/quintal) & MAPE (\%) & Log-MAE & $\Delta$ MAE vs.\ Qtr [95\% CI] \\",
    r"\midrule",
]
for _, r in TG.iterrows():
    det = "Page--Hinkley" if r.detector == "Page-Hinkley" else "PSI"
    prm = r.param.replace("d=", "$\\delta=$").replace("l=", "$\\lambda=$").replace("s0", "$s_0$").replace("thr=", "threshold ")
    dv = 0.0 if abs(r.dmae_vs_qtr) < 0.05 else r.dmae_vs_qtr
    dv_str = "0.0" if dv == 0.0 else f"{dv:+.1f}"
    ci = f"${dv_str}$ [${r.ci_lo:.1f}$, ${r.ci_hi:.1f}$]"
    L.append(f"{det} & {prm} & {int(r.retrains)} & {r.mae:.1f} & {r.mape:.2f} & {r.mae_log:.4f} & {ci} \\\\")
L += [
    r"\bottomrule",
    r"\end{tabularx}",
    r"\smallskip\raggedright\scriptsize Source: authors' replay on cached predictions from Shelar, R. et al.~\cite{onion2025dataset}.",
    r"\end{table}",
]
open("manuscript/tab_grid.tex", "w").write("\n".join(L))

print("P3 gen_tables.py R3 done:", mac["DelayAdwin"], mac["DelayPH"], mac["SensIncZero"], mac["PHGridIncZero"])
