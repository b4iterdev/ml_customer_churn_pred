| model | cv_best_f1 | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc | best_params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 0.6345 | 0.7644 | 0.5418 | 0.7273 | 0.621 | 0.8404 | {"model__max_depth": 10, "model__min_samples_split": 10, "model__n_estimators": 400} |
| logistic_regression | 0.6333 | 0.741 | 0.5078 | 0.7834 | 0.6162 | 0.8408 | {"model__C": 5.0, "model__solver": "lbfgs"} |
| svm | 0.6215 | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 | {"model__C": 1.0, "model__gamma": "scale", "model__kernel": "rbf"} |