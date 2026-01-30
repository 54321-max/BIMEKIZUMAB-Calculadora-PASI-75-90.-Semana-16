# -*- coding: utf-8 -*-
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA_PATH = "BIME EXCEL.xlsx"
OUTDIR = Path("models_bime")
OUTDIR.mkdir(exist_ok=True)

# Columnas reales del Excel
PASI_BASE = "PASI INICIO TTO"
PASI_W16 = "PASI Sem 12-16"

FEATURES = [
    "Sexo",
    "Edad (autocálculo)",
    "IMC (autocálculo)",
    "Artritis",
    "N biológicos previos",
    PASI_BASE,
]

RANDOM_STATE = 42

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_targets(df):
    df = df.copy()
    df = to_numeric(df, [PASI_BASE, PASI_W16])

    base = df[PASI_BASE]
    resp16 = (base - df[PASI_W16]) / base

    df["PASI75_w16"] = (resp16 >= 0.75).astype("float")
    df["PASI90_w16"] = (resp16 >= 0.90).astype("float")
    return df

def build_model(num_cols, cat_cols):
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ])

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=2000,
        random_state=RANDOM_STATE,
        early_stopping=True
    )

    pipe = Pipeline([("prep", preprocess), ("mlp", mlp)])
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=3)

def eval_oof(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "auc": float(roc_auc_score(y, probs)),
        "prauc": float(average_precision_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
    }

def main():
    df = pd.read_excel(DATA_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")].copy()
    df = make_targets(df)

    # Filas válidas: basal > 0 y PASI semana 12–16 disponible
    df = df[(df[PASI_BASE].notna()) & (df[PASI_BASE] > 0) & (df[PASI_W16].notna())].copy()

    feature_cols = [c for c in FEATURES if c in df.columns]
    if PASI_BASE not in feature_cols:
        raise ValueError("No encuentro 'PASI INICIO TTO' en el Excel.")
    if PASI_W16 not in df.columns:
        raise ValueError("No encuentro 'PASI Sem 12-16' en el Excel.")

    # Tipos: Sexo categórica; el resto numéricas (aunque Artritis sea Sí/No, la tratamos numérica y la convertimos)
    cat_cols = ["Sexo"] if "Sexo" in feature_cols else []
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Convertimos numéricas (si Artritis es Sí/No quedará NaN; eso se imputará)
    df = to_numeric(df, num_cols)

    X = df[feature_cols].copy()

    endpoints = ["PASI75_w16", "PASI90_w16"]
    metadata = {"features": feature_cols, "models": {}}

    for target in endpoints:
        y = df[target].astype(int)
        if y.nunique() < 2:
            print(f"[SKIP] {target}: solo una clase.")
            continue

        model = build_model(num_cols, cat_cols)
        scores = eval_oof(model, X, y)
        print(f"{target}: n={scores['n']} pos={scores['pos']} AUC={scores['auc']:.3f} PR-AUC={scores['prauc']:.3f} Brier={scores['brier']:.3f}")

        model.fit(X, y)
        out = OUTDIR / f"bime_{target}.joblib"
        joblib.dump(model, out)

        metadata["models"][target] = {"path": str(out), **scores}

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nListo. Modelos en:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
