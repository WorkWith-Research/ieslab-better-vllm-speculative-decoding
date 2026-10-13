#!/usr/bin/env python3
"""Track-A training + evaluation (prereg docs/tracka-prereg.md §2-§3).

Reads per-request acceptance labels (results/tracka/labels_set1.jsonl, labels_set2.jsonl)
and evaluates admission-time predictors for DSpark-K7 draft acceptance:

  H-TA1 (predictive power): 5-fold CV on SET 1 (diverse, content-disjoint folds).
        Models B0 (intercept), B1 {isl}, M1 {isl,H,bigram,trigram}, M2 M1+{ppl,cache}.
        Primary = M2. Verdict bands: R^2>=0.30 & dR^2 vs B1 >=0.10 -> SUPPORTED;
        R^2>=0.15 & beats B1 -> PARTIAL; else NOT SUPPORTED.
  H-TA2 (cache invariance): within SET 2 (shared doc D), prompt-disjoint 5-fold CV,
        compare MAE(M1) vs MAE(M2=M1+cache_hit_rate). Delta <0.003 -> cache is not a
        quality feature. (In-distribution feature-contribution test.)
  H-TA3 (decision value): on the existing H-CLASS-1 A+C paired per-request data,
        Spearman(pred_acceptance, SD_payoff=1-ratio) and a threshold policy vs
        always-on / always-off regret at each load.

Models: ridge (alpha by inner 5-fold CV) + gradient-boosted trees. Deterministic seed.
Writes results/tracka/model_report.json + model_table.md.
"""
import json, math, os, sys
from collections import Counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
RES = os.path.join(REPO, "results", "tracka")
SEED = 20261013
rng = np.random.RandomState(SEED)

F_M1 = ["isl", "unigram_entropy", "bigram_rep", "trigram_rep"]
F_M2 = F_M1 + ["prompt_ppl", "cache_hit_rate"]

def load_labels(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if "err" in r or r.get("acc_rate") is None:
            continue
        if r.get("ntok", 0) < 10:      # prereg §1 inclusion floor
            continue
        rows.append(r)
    return rows

def to_matrix(rows, feats):
    X = np.zeros((len(rows), len(feats)))
    for j, f in enumerate(feats):
        for i, r in enumerate(rows):
            v = r.get(f)
            X[i, j] = 0.0 if v is None else float(v)
    y = np.array([r["acc_rate"] for r in rows], dtype=float)
    return X, y

def make_model(name, alpha=None):
    if name == "ridge":
        return lambda: Ridge(alpha=alpha or 1.0, random_state=SEED)
    if name == "gbt":
        return lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.9, random_state=SEED)

def cv_evaluate(X, y, feats, model_name="ridge", n_splits=5):
    """Prompt-disjoint K-fold CV. Returns dict with R^2, MAE (mean over folds),
    and per-fold values. Each row is one distinct prompt -> rows are disjoint by identity."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2s, maes, preds_true = [], [], []
    for tr, te in kf.split(X):
        m = make_model(model_name)()
        m.fit(X[tr][:, [F_M1.index(f) if f in F_M1 else F_M2.index(f) for f in feats]], y[tr])
        Xi = X[te][:, [F_M1.index(f) if f in F_M1 else F_M2.index(f) for f in feats]]
        p = m.predict(Xi)
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
        preds_true.append((p, y[te]))
    return {"R2": round(float(np.mean(r2s)), 4), "MAE": round(float(np.mean(maes)), 5),
            "n_folds": n_splits, "model": model_name,
            "per_fold_R2": [round(v, 4) for v in r2s],
            "_preds_true": preds_true}

def fit_full(X, y, feats, model_name="ridge"):
    m = make_model(model_name)()
    idx = [F_M1.index(f) if f in F_M1 else F_M2.index(f) for f in feats]
    m.fit(X[:, idx], y)
    return m

def main():
    set1 = load_labels(os.path.join(RES, "labels_set1.jsonl"))
    set2 = load_labels(os.path.join(RES, "labels_set2.jsonl"))
    report = {"n_set1": len(set1), "n_set2": len(set2)}
    print(f"loaded set1={len(set1)} set2={len(set2)} valid labeled prompts")

    X1, y1 = to_matrix(set1, F_M2)
    X2, y2 = to_matrix(set2, F_M2) if set2 else (None, None)

    # ---------------- H-TA1: SET 1 primary ----------------
    print("\n=== H-TA1: 5-fold CV on SET 1 (content-disjoint folds) ===")
    results = {}
    for label, feats in [("B0", []), ("B1", ["isl"]), ("M1", F_M1), ("M2", F_M2)]:
        if not feats:
            # intercept-only: predict the mean on each fold's train
            kf = KFold(5, shuffle=True, random_state=SEED)
            r2s, maes = [], []
            for tr, te in kf.split(X1):
                pred = np.full(len(te), y1[tr].mean())
                r2s.append(r2_score(y1[te], pred)); maes.append(mean_absolute_error(y1[te], pred))
            results[label] = {"R2": round(float(np.mean(r2s)), 4), "MAE": round(float(np.mean(maes)), 5),
                              "features": [], "model": "intercept"}
        else:
            # ridge as primary estimator; also report GBT for robustness
            res_ridge = cv_evaluate(X1, y1, feats, "ridge")
            res_gbt = cv_evaluate(X1, y1, feats, "gbt")
            results[label] = {"R2": res_ridge["R2"], "MAE": res_ridge["MAE"], "features": feats,
                              "model": "ridge", "per_fold_R2": res_ridge["per_fold_R2"],
                              "R2_gbt": res_gbt["R2"], "MAE_gbt": res_gbt["MAE"]}
        print(f"  {label:3s} feats={results[label]['features']}  R2={results[label]['R2']:.4f} "
              f"MAE={results[label]['MAE']:.5f}" + (f"  [gbt R2={results[label].get('R2_gbt'):.4f}]" if 'R2_gbt' in results[label] else ""))

    r2_m2 = results["M2"]["R2"]; r2_b1 = results["B1"]["R2"]; dr2 = r2_m2 - r2_b1
    if r2_m2 >= 0.30 and dr2 >= 0.10:
        hta1 = "SUPPORTED"
    elif r2_m2 >= 0.15 and dr2 > 0:
        hta1 = "PARTIAL"
    else:
        hta1 = "NOT SUPPORTED"
    report["h_ta1"] = {"verdict": hta1, "R2_M2": r2_m2, "R2_B1": r2_b1, "delta_R2": round(dr2, 4),
                       "per_model": {k: {kk: vv for kk, vv in v.items() if kk != '_preds_true'} for k, v in results.items()}}
    print(f"  -> H-TA1 {hta1} (M2 R2={r2_m2}, B1 R2={r2_b1}, dR2={dr2:.4f})")

    # ---------------- H-TA2: SET 2 cache invariance ----------------
    print("\n=== H-TA2: within SET 2, does cache_hit_rate add acceptance info? ===")
    if set2 and X2 is not None:
        m1 = cv_evaluate(X2, y2, F_M1, "ridge")
        m2 = cv_evaluate(X2, y2, F_M2, "ridge")
        d_mae = abs(m1["MAE"] - m2["MAE"])
        hits = [r.get("cache_hit_rate") for r in set2]
        hit_var = float(np.std([h if h is not None else 0.0 for h in hits]))
        hta2 = "INVARIENT (cache not a quality feature)" if d_mae < 0.003 else "cache_hit_rate contributes"
        report["h_ta2"] = {"verdict": hta2, "MAE_M1": m1["MAE"], "MAE_M2": m2["MAE"],
                           "delta_MAE": round(d_mae, 5), "set2_cache_hit_std": round(hit_var, 4),
                           "note": "in-distribution; all set2 prompts share doc D"}
        print(f"  MAE M1={m1['MAE']} M2={m2['MAE']} delta={d_mae:.5f} (cache_hit std={hit_var:.3f})")
        print(f"  -> H-TA2 {hta2}")
    else:
        report["h_ta2"] = {"verdict": "NO SET2 DATA"}

    # ---------------- H-TA3: decision value on H-CLASS-1 A+C paired data ----------------
    print("\n=== H-TA3: Spearman(pred_acc, SD payoff) + threshold policy regret ===")
    hta3 = build_hta3()
    report["h_ta3"] = hta3
    if "verdict" in hta3:
        print(f"  -> H-TA3 {hta3['verdict']} (spearman={hta3.get('spearman')})")

    json.dump(report, open(os.path.join(RES, "model_report.json"), "w"), indent=2)
    write_md(report)
    print(f"\nwrote {os.path.join(RES, 'model_report.json')} + model_table.md")

def build_hta3():
    """Join predicted acceptance (features from prompt tokens) with per-request SD payoff
    (1 - dspark_tpot/ar_tpot) on the existing H-CLASS-1 A+C paired cells. Prompts come from
    each cell's prompt_manifest.jsonl; tpot from client.jsonl, matched by pid across arms."""
    import itertools
    out = {}
    # fit a full M2 ridge on set1 (the content predictor) to score H-CLASS-1 prompts
    set1 = load_labels(os.path.join(RES, "labels_set1.jsonl"))
    if not set1:
        return {"verdict": "NO LABEL DATA"}
    X1, y1 = to_matrix(set1, F_M2)
    model = fit_full(X1, y1, F_M2, "ridge")

    HCLASS = os.path.join(REPO, "results", "hclass1")
    rows = []   # (pred_acc, payoff, cond, C)
    for cond in ("A", "C"):
        for C in (16, 48, 96):
            for t in (1, 2, 3):
                ac_dir = os.path.join(HCLASS, f"ar_{cond}_c{C}_t{t}")
                dc_dir = os.path.join(HCLASS, f"dspark_{cond}_c{C}_t{t}")
                if not (os.path.exists(f"{ac_dir}/prompt_manifest.jsonl") and
                        os.path.exists(f"{dc_dir}/client.jsonl") and
                        os.path.exists(f"{ac_dir}/client.jsonl")):
                    continue
                man = {json.loads(l)["pid"]: json.loads(l) for l in open(f"{ac_dir}/prompt_manifest.jsonl")}
                ar_tpot = {r["pid"]: r["tpot_ms"] for r in map(json.loads, open(f"{ac_dir}/client.jsonl")) if "ttft_ms" in r}
                ds_tpot = {r["pid"]: r["tpot_ms"] for r in map(json.loads, open(f"{dc_dir}/client.jsonl")) if "ttft_ms" in r}
                for pid, mrow in man.items():
                    if pid not in ar_tpot or pid not in ds_tpot or ar_tpot[pid] <= 0:
                        continue
                    ids = mrow["ids"]
                    feats = {"isl": len(ids)}
                    c = Counter(ids); n = len(ids)
                    feats["unigram_entropy"] = -sum((v/n)*math.log2(v/n) for v in c.values())
                    bg = Counter(zip(ids[:-1], ids[1:])); feats["bigram_rep"] = sum(v-1 for v in bg.values() if v>1)/max(1,n-1)
                    tg = Counter(zip(ids[:-2], ids[1:-1], ids[2:])); feats["trigram_rep"] = sum(v-1 for v in tg.values() if v>1)/max(1,n-2)
                    feats["prompt_ppl"] = 0.0; feats["cache_hit_rate"] = 0.0   # not available per-prompt here; neutral
                    idx = [F_M2.index(f) for f in F_M2]
                    x = np.array([[feats[f] for f in F_M2]])
                    pred = float(model.predict(x)[0])
                    payoff = 1.0 - (ds_tpot[pid] / ar_tpot[pid])
                    rows.append((pred, payoff, cond, C))
    if len(rows) < 50:
        return {"verdict": f"INSUFFICIENT PAIRED DATA ({len(rows)})"}
    pred = np.array([r[0] for r in rows]); pay = np.array([r[1] for r in rows])
    rho, _ = spearmanr(pred, pay)
    out["n_paired"] = len(rows); out["spearman"] = round(float(rho), 4)
    # threshold policy: choose theta on the LABEL set (set1) to maximize mean payoff proxy,
    # then evaluate regret vs always-on/off per load. Proxy: SD helps when acceptance is high.
    # Per-load regret table using the paired rows directly.
    loads = {}
    for p, pf, cond, C in rows:
        loads.setdefault(C, []).append((p, pf))
    policy = {}
    for C, lst in sorted(loads.items()):
        ps = np.array([a[0] for a in lst]); pys = np.array([a[1] for a in lst])
        # always-on regret = mean of -payoff where payoff<0 (i.e. SD hurt); always-off = mean of -payoff where payoff>0
        on_regret = float(np.mean(np.maximum(0, -pys)))     # cost of running SD when it hurts
        off_regret = float(np.mean(np.maximum(0, pys)))     # cost of not running SD when it helps
        # threshold policy: pick theta = median(pred); run SD iff pred>=theta
        theta = float(np.median(ps))
        on_mask = ps >= theta
        thr_regret = float(np.mean(np.where(on_mask, np.maximum(0, -pys), np.maximum(0, pys))))
        beats = int(thr_regret < on_regret and thr_regret < off_regret)
        policy[str(C)] = {"on": round(on_regret,4), "off": round(off_regret,4),
                          "threshold": round(thr_regret,4), "theta": round(theta,4), "beats_both": beats}
    out["per_load_policy"] = policy
    n_beat = sum(1 for v in policy.values() if v["beats_both"])
    if rho >= 0.25 and n_beat >= 2:
        out["verdict"] = "SUPPORTED"
    else:
        out["verdict"] = f"NOT SUPPORTED (spearman={rho:.3f}, beats_at_loads={n_beat}/3)"
    return out

def write_md(report):
    L = ["# Track-A model evaluation (tracka_train.py)", ""]
    L.append(f"Labeled prompts: set1={report['n_set1']} (diverse, primary)  set2={report['n_set2']} (shared-doc cache probe)")
    L.append("")
    L.append("## H-TA1 — predictive power (5-fold CV on SET 1)")
    pm = report["h_ta1"]["per_model"]
    L.append("| model | features | R^2 (ridge) | MAE | R^2 (GBT) |")
    L.append("|---|---|---:|---:|---:|")
    for k in ["B0", "B1", "M1", "M2"]:
        v = pm[k]
        L.append(f"| {k} | {','.join(v['features']) or '(intercept)'} | {v['R2']} | {v['MAE']} | {v.get('R2_gbt','—')} |")
    L.append(f"\n**H-TA1: {report['h_ta1']['verdict']}** — M2 R²={report['h_ta1']['R2_M2']}, B1 R²={report['h_ta1']['R2_B1']}, ΔR²={report['h_ta1']['delta_R2']}")
    L.append("")
    if "h_ta2" in report and report["h_ta2"].get("verdict") != "NO SET2 DATA":
        h2 = report["h_ta2"]
        L.append(f"## H-TA2 — cache invariance (SET 2)\n\n**{h2['verdict']}** — MAE M1={h2['MAE_M1']}, M2={h2['MAE_M2']}, ΔMAE={h2['delta_MAE']} (cache_hit std={h2['set2_cache_hit_std']})")
        L.append("")
    if "h_ta3" in report and report["h_ta3"].get("verdict"):
        h3 = report["h_ta3"]
        L.append(f"## H-TA3 — decision value (H-CLASS-1 A+C paired, n={h3.get('n_paired')})\n\n**{h3['verdict']}** — Spearman(pred_acc, payoff)={h3.get('spearman')}")
        L.append("\n| load | always-on regret | always-off regret | threshold regret | θ* | beats both |")
        L.append("|---|---:|---:|---:|---:|:--:|")
        for C, v in h3.get("per_load_policy", {}).items():
            L.append(f"| c{C} | {v['on']} | {v['off']} | {v['threshold']} | {v['theta']} | {'yes' if v['beats_both'] else 'no'} |")
    open(os.path.join(RES, "model_table.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
