from __future__ import annotations

from pathlib import Path

from churn.training import train_and_select_best


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    reports_dir = root / "reports"
    models_dir = root / "models"

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Please run: .venv/bin/python scripts/download_dataset.py")
        return 1

    artifacts = train_and_select_best(
        data_path=data_path,
        reports_dir=reports_dir,
        models_dir=models_dir,
    )

    print("Training completed successfully.")
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Saved model: {artifacts.best_model_path}")
    print("Comparison table: reports/model_comparison.csv")
    print("Best-model summary: reports/best_model_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
