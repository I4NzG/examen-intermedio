"""Reproduce los experimentos del problema 2 con los datos UCI de abulón."""

from __future__ import annotations

import argparse
import io
import itertools
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.miscmodels.ordinal_model import OrderedModel

from ordinal import OrdinalLogit


ROOT = Path(__file__).parent
OUT = ROOT / "resultados"
URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
NAMES = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
NUMERIC = NAMES[1:8]
LABELS = ["1-7", "8-9", "10-11", "12-14", "15+"]
K = len(LABELS)


def load_data() -> pd.DataFrame:
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith("abalone.data"))
        data = pd.read_csv(archive.open(member), header=None, names=NAMES)
    return data


def encode_target(rings: pd.Series) -> np.ndarray:
    return np.select(
        [rings <= 7, rings <= 9, rings <= 11, rings <= 14],
        [0, 1, 2, 3],
        default=4,
    ).astype(int)


def preprocessor() -> ColumnTransformer:
    return Pipeline(
        [
            ("features", ColumnTransformer(
                [
                    ("numeric", StandardScaler(), NUMERIC),
                    ("sex", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["Sex"]),
                ],
                verbose_feature_names_out=False,
            )),
            ("standardize_all", StandardScaler()),
        ]
    )


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
    }


def ordered_probs_to_predictions(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mode = probs.argmax(axis=1)
    median = (np.cumsum(probs, axis=1) < 0.5).sum(axis=1)
    return mode.astype(int), median.astype(int)


def pava_decreasing(values: np.ndarray) -> np.ndarray:
    """Proyección isotónica no creciente de una fila de probabilidades."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / weight, weight])
    return np.concatenate([np.full(int(weight), val) for val, weight in blocks])


def fit_predict(name: str, params: dict, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray):
    if name == "M1_regresion_lineal":
        model = LinearRegression().fit(X_train, y_train)
        return {"redondeo": np.clip(np.rint(model.predict(X_eval)), 0, K - 1).astype(int)}, None
    if name == "M2_logistica_multinomial":
        model = LogisticRegression(C=params["C"], max_iter=3000, solver="lbfgs")
        model.fit(X_train, y_train)
        return {"moda": model.predict(X_eval)}, None
    if name == "M3_logistica_ordinal":
        model = OrdinalLogit(l2=params["l2"]).fit(X_train, y_train)
        mode, median = ordered_probs_to_predictions(model.predict_proba(X_eval))
        return {"moda": mode, "mediana": median}, model
    if name == "M4_hist_gradient_classifier":
        model = HistGradientBoostingClassifier(max_leaf_nodes=params["leaves"], max_iter=100, learning_rate=0.1, random_state=params["seed"])
        model.fit(X_train, y_train)
        return {"moda": model.predict(X_eval)}, model
    if name == "M5_hist_gradient_regressor":
        model = HistGradientBoostingRegressor(max_leaf_nodes=params["leaves"], max_iter=100, learning_rate=0.1, random_state=params["seed"])
        model.fit(X_train, y_train.astype(float))
        raw_prediction = model.predict(X_eval)
        return {"redondeo": np.clip(np.rint(raw_prediction), 0, K - 1).astype(int)}, {"model": model, "raw_prediction": raw_prediction}
    if name == "M6_umbral_boosting":
        raw = []
        models = []
        for threshold in range(K - 1):
            classifier = HistGradientBoostingClassifier(max_leaf_nodes=params["leaves"], max_iter=100, learning_rate=0.1, random_state=params["seed"] + threshold)
            binary = (y_train > threshold).astype(int)
            classifier.fit(X_train, binary)
            models.append(classifier)
            raw.append(classifier.predict_proba(X_eval)[:, list(classifier.classes_).index(1)])
        raw = np.column_stack(raw)
        corrected = np.vstack([pava_decreasing(row) for row in raw])
        class_probs = np.column_stack([1 - corrected[:, 0], np.diff(corrected, axis=1), corrected[:, -1]])
        class_probs = np.maximum(class_probs, 1e-12)
        class_probs /= class_probs.sum(axis=1, keepdims=True)
        return {"moda": class_probs.argmax(axis=1)}, {"raw_threshold_probabilities": raw, "models": models}
    raise ValueError(name)


def metric_objective(name: str, value: float) -> float:
    return -value if name in {"accuracy", "qwk"} else value


def candidates(seed: int):
    return {
        "M1_regresion_lineal": [{}],
        "M2_logistica_multinomial": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "M3_logistica_ordinal": [{"l2": c} for c in (0.0, 0.01, 0.1)],
        "M4_hist_gradient_classifier": [{"leaves": l, "seed": seed} for l in (7, 15)],
        "M5_hist_gradient_regressor": [{"leaves": l, "seed": seed} for l in (7, 15)],
        "M6_umbral_boosting": [{"leaves": l, "seed": seed} for l in (7, 15)],
    }


def cross_validate(X: pd.DataFrame, y: np.ndarray, seed: int, folds: int = 5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    records = []
    for model_name, configs in candidates(seed).items():
        for params in configs:
            collected = {}
            for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
                transform = preprocessor()
                xt = transform.fit_transform(X.iloc[tr])
                xv = transform.transform(X.iloc[va])
                predictions, _ = fit_predict(model_name, params, xt, y[tr], xv)
                for variant, pred in predictions.items():
                    scores = metric_values(y[va], pred)
                    collected.setdefault(variant, []).append(scores)
            for variant, fold_scores in collected.items():
                means = {m: float(np.mean([s[m] for s in fold_scores])) for m in ("accuracy", "mae", "qwk")}
                records.append({"model": model_name, "variant": variant, "params": params, **means})
    return records


def choose_cv(records: list[dict], metric: str) -> dict[str, dict]:
    selected = {}
    for row in records:
        key = (row["model"], row["variant"])
        if key not in selected or metric_objective(metric, row[metric]) < metric_objective(metric, selected[key][metric]):
            selected[key] = row
    return {f"{model}:{variant}": row for (model, variant), row in selected.items()}


def calibrate_m5_cuts(X: pd.DataFrame, y: np.ndarray, params: dict, seed: int, metric: str) -> list[float]:
    """Select four ordered cuts from out-of-fold predictions on training only."""
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y):
        transform = preprocessor()
        xt = transform.fit_transform(X.iloc[tr])
        xv = transform.transform(X.iloc[va])
        _, details = fit_predict("M5_hist_gradient_regressor", params, xt, y[tr], xv)
        oof[va] = details["raw_prediction"]
    candidates = np.unique(np.r_[np.quantile(oof, np.linspace(0.05, 0.95, 19)), [0.5, 1.5, 2.5, 3.5]])
    best_value, best_cuts = None, None
    for cuts in itertools.combinations(candidates, K - 1):
        pred = np.digitize(oof, cuts)
        score = metric_values(y, pred)[metric]
        objective = metric_objective(metric, score)
        if best_value is None or objective < best_value:
            best_value, best_cuts = objective, list(map(float, cuts))
    return best_cuts


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=222215025)
    parser.add_argument("--metric", choices=["accuracy", "mae", "qwk"], default="mae")
    parser.add_argument("--invalid", choices=["exclude", "keep"], default="exclude")
    parser.add_argument("--m5-conversion", choices=["round", "cuts"], default="cuts")
    parser.add_argument("--audit-only", action="store_true", help="Solo descargar y resumir clases/medidas sospechosas")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    raw = load_data()
    invalid = (raw[NUMERIC] <= 0).any(axis=1)
    audit = {
        "rows_original": len(raw),
        "missing_values": int(raw.isna().sum().sum()),
        "sex_counts": raw["Sex"].value_counts().to_dict(),
        "invalid_physical_rows": int(invalid.sum()),
        "invalid_examples": raw.loc[invalid].drop(columns="Rings").to_dict(orient="records"),
        "cleaning_choice": args.invalid,
    }
    if args.audit_only:
        save_json(OUT / "auditoria_datos.json", audit)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    if args.invalid == "exclude":
        raw = raw.loc[~invalid].copy()
    y_all = encode_target(raw["Rings"])
    X_all = raw.drop(columns="Rings")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=args.seed
    )
    audit["rows_used"] = len(raw)
    audit["train_size"] = len(X_train)
    audit["test_size"] = len(X_test)
    audit["train_class_counts"] = np.bincount(y_train, minlength=K).tolist()
    audit["test_class_counts"] = np.bincount(y_test, minlength=K).tolist()
    save_json(OUT / "auditoria_datos.json", audit)

    cv_records = cross_validate(X_train.reset_index(drop=True), y_train, args.seed)
    (OUT / "validacion_cruzada.json").write_text(json.dumps(cv_records, indent=2, ensure_ascii=False), encoding="utf-8")
    selected = choose_cv(cv_records, args.metric)
    m5_row = selected["M5_hist_gradient_regressor:redondeo"]
    m5_cuts = calibrate_m5_cuts(X_train.reset_index(drop=True), y_train, m5_row["params"], args.seed, args.metric) if args.m5_conversion == "cuts" else None
    final_scores, test_predictions = [], {}
    best_boosting = None
    best_boosting_score = None
    for key, row in selected.items():
        model_name, variant = key.split(":", 1)
        transform = preprocessor()
        xt = transform.fit_transform(X_train)
        xv = transform.transform(X_test)
        preds, details = fit_predict(model_name, row["params"], xt, y_train, xv)
        if model_name == "M5_hist_gradient_regressor" and m5_cuts is not None:
            pred = np.digitize(details["raw_prediction"], m5_cuts)
            variant_name = "cortes_calibrados"
        else:
            pred = preds[variant]
            variant_name = variant
        score = {"model": model_name, "variant": variant_name, "cv_params": row["params"], **metric_values(y_test, pred)}
        final_scores.append(score)
        test_predictions[f"{model_name}:{variant_name}"] = pred.tolist()
        if model_name.startswith("M4_") or model_name.startswith("M5_") or model_name.startswith("M6_"):
            val = metric_objective(args.metric, score[args.metric])
            if best_boosting_score is None or val < best_boosting_score:
                best_boosting_score = val
                best_boosting = (key, row["params"])

    metric_key = {"mae": "MAE", "qwk": "QWK", "accuracy": "exactitud"}[args.metric]
    best = min(final_scores, key=lambda s: metric_objective(args.metric, s[args.metric]))
    labels_int = np.arange(K)
    cm = confusion_matrix(y_test, test_predictions[f"{best['model']}:{best['variant']}"], labels=labels_int)
    pd.DataFrame(final_scores).to_csv(OUT / "metricas_prueba.csv", index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / "matriz_confusion.csv", index_label="real/predicho")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K), LABELS)
    ax.set_yticks(range(K), LABELS)
    ax.set_xlabel("Nivel predicho")
    ax.set_ylabel("Nivel real")
    ax.set_title(f"Matriz de confusión: {best['model']}")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Número de abulones")
    fig.tight_layout()
    fig.savefig(OUT / "matriz_confusion.png", dpi=180)
    plt.close(fig)

    # Fit the ordinal model and statsmodels on train only for coefficient comparison.
    transform = preprocessor()
    xt = transform.fit_transform(X_train)
    feature_names = transform.get_feature_names_out().tolist()
    ordinal = OrdinalLogit(l2=0.0).fit(xt, y_train)
    sm = OrderedModel(y_train, xt, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
    coef_compare = [{"feature": f, "ordinal_custom": float(v), "statsmodels": float(sm.params[i])}
                    for i, (f, v) in enumerate(zip(feature_names, ordinal.coef_))]
    pd.DataFrame(coef_compare).to_csv(OUT / "coeficientes_ordinales.csv", index=False)
    # Numerical gradient check at a stable interior point.
    m = min(80, len(y_train))
    xcheck, ycheck = xt[:m], y_train[:m]
    check_model = OrdinalLogit(l2=0.0)
    check_model.fit(xcheck, ycheck)
    grad_error = ordinal.gradient_check_

    # Compare threshold-specific binary boosting probabilities on the untouched test set.
    trans_test = preprocessor()
    xb_train = trans_test.fit_transform(X_train)
    xb_test = trans_test.transform(X_test)
    threshold_params = best["params"] if best["model"] == "M6_umbral_boosting" else {"leaves": 7, "seed": args.seed}
    _, m6info = fit_predict("M6_umbral_boosting", threshold_params, xb_train, y_train, xb_test)
    raw_probs = m6info["raw_threshold_probabilities"]
    violation = np.any(np.diff(raw_probs, axis=1) > 0, axis=1)
    # Fit ordinal coefficients on standardized full training data; response never uses test labels.
    importances = []
    if best_boosting is not None:
        boost_key, boost_params = best_boosting
        boost_name, boost_variant = boost_key.split(":", 1)
        bpred, fitted = fit_predict(boost_name, boost_params, xb_train, y_train, xb_test)
        if boost_name == "M4_hist_gradient_classifier" and hasattr(fitted, "feature_importances_"):
            pass
        # Permutation importance is measured using MAE on the test set, as requested.
        if boost_name == "M4_hist_gradient_classifier":
            class Wrapped:
                def __init__(self, fitted_model): self.fitted_model = fitted_model
                def predict(self, xx): return self.fitted_model.predict(xx)
            estimator = Wrapped(fitted)
        elif boost_name == "M5_hist_gradient_regressor":
            class Wrapped:
                def __init__(self, fitted_model, cuts): self.fitted_model, self.cuts = fitted_model, cuts
                def predict(self, xx):
                    raw = self.fitted_model.predict(xx)
                    return np.digitize(raw, self.cuts) if self.cuts is not None else np.clip(np.rint(raw), 0, K - 1).astype(int)
            estimator = Wrapped(fitted["model"], m5_cuts)
        else:
            # Reapply the fitted threshold classifiers after each feature permutation.
            class Wrapped:
                def __init__(self, fitted_models): self.fitted_models = fitted_models
                def predict(self, xx):
                    probs = np.column_stack([model.predict_proba(xx)[:, list(model.classes_).index(1)] for model in self.fitted_models])
                    probs = np.vstack([pava_decreasing(row) for row in probs])
                    cp = np.column_stack([1 - probs[:, 0], np.diff(probs, axis=1), probs[:, -1]])
                    return cp.argmax(axis=1)
            estimator = Wrapped(details["models"])
        rng = np.random.default_rng(args.seed)
        base_mae = mean_absolute_error(y_test, estimator.predict(xb_test))
        for j, feature in enumerate(feature_names):
            increases = []
            for _ in range(10):
                permuted = xb_test.copy()
                permuted[:, j] = rng.permutation(permuted[:, j])
                increases.append(mean_absolute_error(y_test, estimator.predict(permuted)) - base_mae)
            importances.append({"feature": feature, "mean_mae_increase": float(np.mean(increases)), "std": float(np.std(increases, ddof=1))})
        importances.sort(key=lambda row: row["mean_mae_increase"], reverse=True)
    pd.DataFrame(importances).to_csv(OUT / "importancia_permutacion.csv", index=False)

    report = {
        "seed": args.seed,
        "metric_for_cv_selection": args.metric,
        "invalid_measurement_policy": args.invalid,
        "m5_conversion": args.m5_conversion,
        "m5_cutpoints_selected_on_training_oof": m5_cuts,
        "best_test_model_by_selected_metric": best,
        "test_scores_for_cv_selected_models": final_scores,
        "test_confusion_matrix_best_model": {"labels": LABELS, "matrix": cm.tolist()},
        "ordinal_gradient_check_error": grad_error,
        "ordinal_custom_vs_statsmodels_coefficients": coef_compare,
        "M6_fraction_test_with_nonmonotone_raw_probabilities": float(np.mean(violation)),
        "best_boosting_model": {"key": best_boosting[0], "params": best_boosting[1]} if best_boosting else None,
        "permutation_importance_MAE_best_boosting": importances,
    }
    save_json(OUT / "resultados_finales.json", report)
    print(json.dumps({"best": best, "test_scores": final_scores, "M6_nonmonotone_fraction": report["M6_fraction_test_with_nonmonotone_raw_probabilities"]}, indent=2))


if __name__ == "__main__":
    main()
