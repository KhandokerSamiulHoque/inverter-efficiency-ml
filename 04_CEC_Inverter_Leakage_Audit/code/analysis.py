"""Paper 4: leakage-aware ML screening on the pvlib CEC inverter table."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import binary_metrics, ensure_dirs, model_scores, save_json, savefig, seed_everything, set_plot_style


PAPER = ROOT / "04_CEC_Inverter_Leakage_Audit"
RESULTS, FIGURES = ensure_dirs(PAPER)
RAW = ["Vac", "Pso", "Paco", "Pdco", "Vdco", "C0", "C1", "C2", "C3", "Pnt", "Vdcmax", "Idcmax", "Mppt_low", "Mppt_high"]
DERIVED = ["eta_10pct", "eta_20pct", "eta_30pct", "eta_50pct", "eta_75pct", "eta_100pct"]


def prepare() -> pd.DataFrame:
    snapshot = RESULTS / "cec_inverter_snapshot.csv"
    if snapshot.exists():
        return pd.read_csv(snapshot)
    database = pvlib.pvsystem.retrieve_sam("CECInverter").T.copy()
    numeric = database[RAW].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    for model_id, row in numeric.iterrows():
        values = row.to_dict()
        record = {"model_id": model_id, **values}
        for fraction in [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]:
            p_dc = fraction * values["Pdco"]
            p_ac = float(pvlib.inverter.sandia(values["Vdco"], p_dc, values))
            record[f"eta_{int(fraction * 100)}pct"] = np.clip(p_ac / p_dc, 0, 1)
        record["cec_weighted_efficiency"] = (
            0.04 * record["eta_10pct"] + 0.05 * record["eta_20pct"] + 0.12 * record["eta_30pct"]
            + 0.21 * record["eta_50pct"] + 0.53 * record["eta_75pct"] + 0.05 * record["eta_100pct"]
        )
        rows.append(record)
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    frame.to_csv(snapshot, index=False)
    return frame


def classifiers():
    return {
        "Logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        "SVM-RBF": SVC(C=2, probability=True, class_weight="balanced", random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(random_state=42),
    }


def main() -> None:
    seed_everything()
    set_plot_style()
    frame = prepare()
    target = frame["cec_weighted_efficiency"].to_numpy()
    threshold = float(np.median(target))
    y = (target >= threshold).astype(int)
    feature_sets = {
        "Direct target included": RAW + DERIVED + ["cec_weighted_efficiency"],
        "Derived components included": RAW + DERIVED,
        "Raw Sandia parameters only": RAW,
    }
    train_i, test_i = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    holdout_rows, cv_rows, fitted = [], [], {}
    for set_name, features in feature_sets.items():
        X = frame[features]
        for model_name, estimator in classifiers().items():
            pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
            pipeline.fit(X.iloc[train_i], y[train_i])
            score = model_scores(pipeline, X.iloc[test_i])
            pred = (score >= 0.5).astype(int)
            holdout_rows.append({"feature_set": set_name, "model": model_name, **binary_metrics(y[test_i], pred, score)})
            scores = cross_validate(pipeline, X, y, cv=cv, scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}, n_jobs=1)
            cv_rows.append({
                "feature_set": set_name,
                "model": model_name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(), "cv_accuracy_std": scores["test_accuracy"].std(ddof=1),
                "cv_f1_mean": scores["test_f1"].mean(), "cv_f1_std": scores["test_f1"].std(ddof=1),
                "cv_auc_mean": scores["test_roc_auc"].mean(), "cv_auc_std": scores["test_roc_auc"].std(ddof=1),
            })
            fitted[(set_name, model_name)] = pipeline
    holdout = pd.DataFrame(holdout_rows)
    cv_table = pd.DataFrame(cv_rows)
    holdout.to_csv(RESULTS / "classification_holdout_metrics.csv", index=False)
    cv_table.to_csv(RESULTS / "classification_cross_validation.csv", index=False)

    regressors = {
        "Elastic net": ElasticNet(alpha=0.0001, l1_ratio=0.25, max_iter=10000, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
    }
    X_raw = frame[RAW]
    reg_rows, reg_cv_rows = [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, estimator in regressors.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        pipe.fit(X_raw.iloc[train_i], target[train_i])
        pred = pipe.predict(X_raw.iloc[test_i])
        reg_rows.append({"model": name, "mae": mean_absolute_error(target[test_i], pred), "rmse": mean_squared_error(target[test_i], pred) ** 0.5, "r2": r2_score(target[test_i], pred)})
        values = cross_validate(pipe, X_raw, target, cv=kfold, scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}, n_jobs=1)
        reg_cv_rows.append({"model": name, "cv_mae_mean": -values["test_mae"].mean(), "cv_mae_std": values["test_mae"].std(ddof=1), "cv_rmse_mean": -values["test_rmse"].mean(), "cv_rmse_std": values["test_rmse"].std(ddof=1), "cv_r2_mean": values["test_r2"].mean(), "cv_r2_std": values["test_r2"].std(ddof=1)})
    pd.DataFrame(reg_rows).sort_values("mae").to_csv(RESULTS / "raw_parameter_regression_holdout.csv", index=False)
    pd.DataFrame(reg_cv_rows).sort_values("cv_mae_mean").to_csv(RESULTS / "raw_parameter_regression_cross_validation.csv", index=False)

    best_by_set = cv_table.loc[cv_table.groupby("feature_set")["cv_f1_mean"].idxmax()].sort_values("cv_f1_mean")
    plt.figure(figsize=(8.4, 4.8))
    labels = [f"{row.feature_set}\n({row.model})" for row in best_by_set.itertuples()]
    plt.barh(labels, best_by_set["cv_f1_mean"], xerr=best_by_set["cv_f1_std"], color=["#2a9d8f", "#f4a261", "#e76f51"])
    plt.xlim(0, 1.02)
    plt.xlabel("Five-fold F1 (mean ± SD)")
    plt.title("Target leakage inflates apparent inverter efficiency-tier performance")
    savefig(FIGURES / "figure_1_leakage_comparison.png")

    raw_best_row = cv_table[cv_table["feature_set"] == "Raw Sandia parameters only"].sort_values("cv_f1_mean", ascending=False).iloc[0]
    raw_best = fitted[("Raw Sandia parameters only", raw_best_row["model"])]
    raw_score = model_scores(raw_best, X_raw.iloc[test_i])
    raw_pred = (raw_score >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y[test_i], raw_pred, display_labels=["Below median", "At/above median"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Leakage-controlled holdout: {raw_best_row['model']}")
    savefig(FIGURES / "figure_2_raw_feature_confusion_matrix.png")

    importance = permutation_importance(raw_best, X_raw.iloc[test_i], y[test_i], scoring="f1", n_repeats=30, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": RAW, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(RESULTS / "raw_feature_permutation_importance.csv", index=False)
    top = imp.head(12).sort_values("importance_mean")
    plt.figure(figsize=(7.2, 5.0))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#457b9d")
    plt.xlabel("Decrease in held-out F1 after permutation")
    plt.title("Leakage-controlled raw-parameter importance")
    savefig(FIGURES / "figure_3_raw_feature_importance.png")

    plt.figure(figsize=(7.4, 4.5))
    plt.hist(target * 100, bins=45, color="#264653", alpha=0.82)
    plt.axvline(threshold * 100, color="#e76f51", linestyle="--", label=f"Median={threshold * 100:.2f}%")
    plt.xlabel("Computed CEC-weighted efficiency (%)")
    plt.ylabel("Inverter count")
    plt.title("Computed efficiency distribution in the pvlib CEC inverter snapshot")
    plt.legend()
    savefig(FIGURES / "figure_4_efficiency_distribution.png")

    summary = {
        "records": len(frame), "median_efficiency": threshold, "raw_parameters": RAW,
        "best_by_feature_set": best_by_set.to_dict(orient="records"),
        "raw_feature_best": raw_best_row.to_dict(),
        "limitations": ["The target is a deterministic function of public Sandia-model parameters, not an independent UL 1741 pass/fail outcome.", "CEC database inclusion is not equivalent to NRTL certification status, and the analysis must not be used to issue conformity decisions."],
    }
    save_json(RESULTS / "analysis_summary.json", summary)
    print(best_by_set.to_string(index=False))


if __name__ == "__main__":
    main()
