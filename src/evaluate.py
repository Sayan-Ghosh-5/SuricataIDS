"""
evaluate.py
Fair comparison (slide 14): the SAME held-out alerts go through
  BASELINE : Suricata only      -> every alert reaches the analyst
  PROPOSED : Suricata + ML      -> alerts split into ESCALATE / HUMAN_REVIEW / DEPRIORITIZE

Metrics (slide 15)   positive = "alert stays in the analyst queue" (ESCALATE or HUMAN_REVIEW)
  FPR = FP/(FP+TN)   Precision = TP/(TP+FP)   Recall = TP/(TP+FN)   F1 = 2PR/(P+R)
  Alert Reduction Rate = (A0 - A1) / A0 * 100
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

from train_model import get_models, split, ROOT
from decision_engine import decide, T_LOW, T_HIGH, ESCALATE, REVIEW, DEPRIORITIZE
from eve_parser import NUMERIC, CATEGORICAL

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, RED, GREY, AMBER, GREEN = "#1F4E79", "#C0392B", "#7F8C8D", "#E39B1B", "#2E8B57"


def triage_metrics(y_fp, p_fp, t_low=T_LOW, t_high=T_HIGH):
    """y_fp: 1 = false positive (benign), 0 = genuine."""
    action = np.array([decide(p, t_low, t_high) for p in p_fp])
    kept = action != DEPRIORITIZE                        # predicted positive (stays in analyst queue)
    genuine = (y_fp == 0)
    TP = int((kept & genuine).sum());  FN = int((~kept & genuine).sum())
    FP = int((kept & ~genuine).sum()); TN = int((~kept & ~genuine).sum())
    A0, A1 = len(y_fp), int(kept.sum())
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    return {
        "A0": A0, "A1": A1, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "FPR": FP / max(1, FP + TN), "Precision": prec, "Recall": rec,
        "F1": 2 * prec * rec / max(1e-9, prec + rec),
        "AlertReductionRate": (A0 - A1) / A0 * 100,
        "n_escalate": int((action == ESCALATE).sum()), "n_review": int((action == REVIEW).sum()),
        "n_deprioritize": int((action == DEPRIORITIZE).sum()),
        "escalate_precision": float(genuine[action == ESCALATE].mean()) if (action == ESCALATE).any() else float("nan"),
    }


def main():
    Xtr, Xte, ytr, yte = split()
    models, probs, res = get_models(), {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        probs[name] = m.predict_proba(Xte)[:, 1]
    p_rf = probs["Random Forest"]; y = yte.values

    # ---------- baseline vs proposed -------------------------------------------------
    n_gen, n_fp = int((y == 0).sum()), int((y == 1).sum())
    base = {"A0": len(y), "A1": len(y), "TP": n_gen, "FP": n_fp, "TN": 0, "FN": 0,
            "FPR": 1.0, "Precision": n_gen / len(y), "Recall": 1.0,
            "F1": 2 * (n_gen / len(y)) / (1 + n_gen / len(y)), "AlertReductionRate": 0.0}
    prop = triage_metrics(y, p_rf)
    res["baseline"], res["proposed"] = base, prop
    res["test_size"], res["genuine"], res["false_positive"] = len(y), n_gen, n_fp
    res["thresholds"] = {"T_LOW": T_LOW, "T_HIGH": T_HIGH}

    # ---------- threshold tuning on TRAINING data only (no test leakage) ---------------
    RECALL_TARGET = 0.95
    oof = cross_val_predict(get_models()["Random Forest"], Xtr, ytr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)[:, 1]
    t_high_tuned = 0.99
    for th in np.arange(0.50, 0.995, 0.005):
        if triage_metrics(ytr.values, oof, T_LOW, th)["Recall"] >= RECALL_TARGET:
            t_high_tuned = round(float(th), 3); break
    tuned = triage_metrics(y, p_rf, T_LOW, t_high_tuned)
    res["tuned"], res["T_HIGH_tuned"], res["recall_target"] = tuned, t_high_tuned, RECALL_TARGET

    # ---------- model comparison -----------------------------------------------------
    comp = []
    for name, p in probs.items():
        t = triage_metrics(y, p)
        comp.append({"Model": name, "ROC_AUC": roc_auc_score(y, p),
                     "F1_at_0.5": f1_score(y, (p >= 0.5).astype(int)),
                     "FPR": t["FPR"], "Recall": t["Recall"], "AlertReductionRate": t["AlertReductionRate"]})
    res["model_comparison"] = comp
    cv = cross_val_score(get_models()["Random Forest"], pd.concat([Xtr, Xte]), np.concatenate([ytr, yte]),
                         cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=-1)
    res["rf_cv_auc_mean"], res["rf_cv_auc_std"] = float(cv.mean()), float(cv.std())

    # ---------- plain 0.5 classifier confusion matrix --------------------------------
    cm = confusion_matrix(y, (p_rf >= 0.5).astype(int))
    res["confusion_0.5"] = cm.tolist()

    # ---------- threshold sweep ------------------------------------------------------
    sweep = []
    for th in np.arange(0.50, 0.99, 0.02):
        s = triage_metrics(y, p_rf, T_LOW, th)
        sweep.append({"T_HIGH": round(float(th), 2), "Recall": s["Recall"], "AlertReductionRate": s["AlertReductionRate"], "FPR": s["FPR"]})
    res["sweep"] = sweep

    # ---------- feature importance ---------------------------------------------------
    rf = models["Random Forest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    agg = {}
    for n, v in imp.items():
        base_name = n.split("__", 1)[1]
        for c in CATEGORICAL:
            if base_name.startswith(c + "_"):
                base_name = c
        agg[base_name] = agg.get(base_name, 0) + v
    imp = pd.Series(agg).sort_values(ascending=False)
    res["top_features"] = imp.head(10).round(4).to_dict()

    # =========================== FIGURES ==============================================
    # FIG: probability histogram with decision zones
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axvspan(0, T_LOW, color=RED, alpha=.10); ax.axvspan(T_LOW, T_HIGH, color=AMBER, alpha=.12); ax.axvspan(T_HIGH, 1, color=GREEN, alpha=.10)
    bins = np.linspace(0, 1, 41)
    ax.hist(p_rf[y == 0], bins, alpha=.85, color=RED, label="Genuine attack alerts")
    ax.hist(p_rf[y == 1], bins, alpha=.80, color=GREY, label="False-positive alerts")
    ax.set_xlabel("Predicted false-positive probability"); ax.set_ylabel("Number of alerts (test set)")
    for x, t in [(T_LOW / 2, "ESCALATE"), ((T_LOW + T_HIGH) / 2, "HUMAN REVIEW"), ((T_HIGH + 1) / 2, "DEPRIORITIZE")]:
        ax.text(x, ax.get_ylim()[1] * .93, t, ha="center", fontsize=8, weight="bold", color="#333")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_probability_zones.png"); plt.close(fig)

    # FIG: baseline vs proposed analyst queue
    labels3 = ["Baseline\n(Suricata only)", f"Proposed\nT_HIGH={T_HIGH}", f"Proposed (tuned)\nT_HIGH={t_high_tuned}"]
    gen3 = [n_gen, prop["TP"], tuned["TP"]]; fp3 = [n_fp, prop["FP"], tuned["FP"]]
    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.bar(labels3, gen3, color=RED, label="Genuine alerts in analyst queue")
    ax.bar(labels3, fp3, bottom=gen3, color=GREY, label="False positives in analyst queue")
    for i, (g, f) in enumerate(zip(gen3, fp3)):
        ax.text(i, g + f + 80, f"{g + f:,}", ha="center", weight="bold")
    ax.set_ylabel("Analyst-facing alerts (test set)"); ax.set_ylim(0, n_gen + n_fp + 700)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_baseline_vs_proposed.png"); plt.close(fig)

    # FIG: ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    for (name, p), c in zip(probs.items(), [BLUE, GREY, AMBER]):
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#bbb"); ax.set_xlabel("False-positive-class FPR"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_roc.png"); plt.close(fig)

    # FIG: threshold sweep
    sw = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sw["T_HIGH"], sw["AlertReductionRate"], color=BLUE, lw=2, label="Alert reduction rate (%)")
    ax.plot(sw["T_HIGH"], sw["Recall"] * 100, color=RED, lw=2, label="Recall of genuine attacks (%)")
    ax.axvline(T_HIGH, color=GREY, ls="--"); ax.text(T_HIGH + .005, 5, "default", fontsize=8, color=GREY)
    ax.axvline(t_high_tuned, color=AMBER, ls="--"); ax.text(t_high_tuned + .005, 15, "tuned", fontsize=8, color=AMBER)
    ax.set_xlabel("Deprioritize threshold (T_HIGH)"); ax.set_ylabel("Percent")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_threshold_sweep.png"); plt.close(fig)

    # FIG: feature importance
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(top.index, top.values, color=BLUE); ax.set_xlabel("Random Forest feature importance")
    fig.tight_layout(); fig.savefig(FIG / "fig_feature_importance.png"); plt.close(fig)

    # FIG: confusion matrix at 0.5
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Genuine", "False pos."]); ax.set_yticks([0, 1], ["Genuine", "False pos."])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", weight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig_confusion_matrix.png"); plt.close(fig)

    # FIG: architecture diagram
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-1, 34)
    boxes = [("Network\nTraffic", 2), ("Suricata\nNIDS", 17), ("EVE JSON\nalerts", 32), ("Feature\nExtraction", 47), ("Random Forest\nClassifier", 62)]
    for t, x in boxes:
        ax.add_patch(plt.Rectangle((x, 11), 12, 9, fc="#EAF1F8", ec=BLUE, lw=1.5)); ax.text(x + 6, 15.5, t, ha="center", va="center", fontsize=8, weight="bold")
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1], 15.5), xytext=(boxes[i][1] + 12, 15.5), arrowprops=dict(arrowstyle="->", color=BLUE))
    outs = [("Likely genuine\nESCALATE", 24, RED), ("Uncertain\nHUMAN REVIEW", 14, AMBER), ("Likely false pos.\nDEPRIORITIZE", 4, GREEN)]
    for t, y0, c in outs:
        ax.add_patch(plt.Rectangle((80, y0), 17, 7, fc="white", ec=c, lw=1.8)); ax.text(88.5, y0 + 3.5, t, ha="center", va="center", fontsize=7.5, weight="bold", color=c)
        ax.annotate("", xy=(80, y0 + 3.5), xytext=(74, 15.5), arrowprops=dict(arrowstyle="->", color=c))
    ax.text(88.5, 0.5, "-> SOC dashboard", ha="center", fontsize=8, style="italic")
    fig.tight_layout(); fig.savefig(FIG / "fig_architecture.png"); plt.close(fig)

    json.dump(res, open(ROOT / "data" / "results.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["test_size", "genuine", "false_positive", "baseline", "proposed", "rf_cv_auc_mean", "rf_cv_auc_std", "confusion_0.5", "top_features", "tuned", "T_HIGH_tuned"]}, indent=1, default=float))
    print(pd.DataFrame(comp).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
