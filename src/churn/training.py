from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


RANDOM_STATE = 42


@dataclass
class TrainArtifacts:
    comparison_df: pd.DataFrame
    best_model_name: str
    best_model_path: Path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    y = data["Churn"].map({"No": 0, "Yes": 1})
    X = data.drop(columns=["Churn", "customerID"], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def model_search_space() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {
                "model__C": [0.1, 1.0, 5.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10],
                "model__min_samples_split": [2, 10],
            },
        ),
        "svm": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0, 2.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale"],
            },
        ),
    }


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in df.iterrows()
    ]
    output_path.write_text("\n".join([header, separator, *rows]), encoding="utf-8")


def train_and_select_best(
    data_path: Path,
    reports_dir: Path,
    models_dir: Path,
) -> TrainArtifacts:
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X)
    search_space = model_search_space()

    results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for model_name, (estimator, param_grid) in search_space.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            refit=True,
        )

        grid.fit(X_train, y_train)
        metrics = evaluate_model(grid.best_estimator_, X_test, y_test)

        record: dict[str, Any] = {
            "model": model_name,
            "cv_best_f1": round(float(grid.best_score_), 4),
            "test_accuracy": round(metrics["accuracy"], 4),
            "test_precision": round(metrics["precision"], 4),
            "test_recall": round(metrics["recall"], 4),
            "test_f1": round(metrics["f1"], 4),
            "test_roc_auc": round(metrics["roc_auc"], 4),
            "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        }
        results.append(record)

        if best_record is None or record["test_f1"] > best_record["test_f1"]:
            best_record = {
                **record,
                "best_estimator": grid.best_estimator_,
            }

    if best_record is None:
        raise RuntimeError("Training did not produce any model results.")

    comparison_df = (
        pd.DataFrame(results)
        .sort_values(by="test_f1", ascending=False)
        .reset_index(drop=True)
    )

    comparison_csv_path = reports_dir / "model_comparison.csv"
    comparison_md_path = reports_dir / "model_comparison.md"
    summary_json_path = reports_dir / "best_model_summary.json"

    comparison_df.to_csv(comparison_csv_path, index=False)
    write_markdown_table(comparison_df, comparison_md_path)

    best_model_name = str(best_record["model"])
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_record["best_estimator"], best_model_path)

    summary_payload = {
        "best_model": best_model_name,
        "selection_metric": "test_f1",
        "metrics": {
            "cv_best_f1": best_record["cv_best_f1"],
            "test_accuracy": best_record["test_accuracy"],
            "test_precision": best_record["test_precision"],
            "test_recall": best_record["test_recall"],
            "test_f1": best_record["test_f1"],
            "test_roc_auc": best_record["test_roc_auc"],
        },
        "best_params": json.loads(str(best_record["best_params"])),
        "model_path": str(best_model_path),
        "dataset_path": str(data_path),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
    }

    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return TrainArtifacts(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_model_path=best_model_path,
    )
