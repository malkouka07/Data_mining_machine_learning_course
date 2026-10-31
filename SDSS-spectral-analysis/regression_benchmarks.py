"""Checkpointed wall-clock timings for the project's regression comparisons."""

from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits


def benchmark_regressions(X_train_full, X_validation_full, X_train_pca,
                          X_validation_pca, y_train, y_validation,
                          fixed_settings, retuned_settings, output_dir):
    """Fit each model once; report median time for three validation predictions.

    No model fitting, prediction, serialization or search runs concurrently with
    another model. Only the estimator's own eight workers are used.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = list(y_train.columns)
    for train, validation in [(X_train_full, X_validation_full),
                              (X_train_pca, X_validation_pca)]:
        assert train.index.equals(y_train.index)
        assert validation.index.equals(y_validation.index)
        assert train.columns.equals(validation.columns)
        assert np.isfinite(train.to_numpy()).all()
        assert np.isfinite(validation.to_numpy()).all()
    assert set(y_train.index).isdisjoint(y_validation.index)
    cases = [
        ("linear", "full", X_train_full, X_validation_full, None),
        ("linear", "PCA2", X_train_pca, X_validation_pca, None),
        ("forest", "PCA2_fixed", X_train_pca, X_validation_pca, fixed_settings),
        ("forest", "PCA2_retuned", X_train_pca, X_validation_pca, retuned_settings),
        ("forest", "full", X_train_full, X_validation_full, fixed_settings),
    ]
    rows = []
    for method, case, train, validation, settings in cases:
        for target in targets:
            result_path = output / f"{method}_{case}_{target}.json"
            if result_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                print(f"CACHED timing: {method} {case} {target}", flush=True)
            else:
                if method == "linear":
                    model = LinearRegression()
                    native_threads = 8
                else:
                    parameters = settings.loc[target]
                    model = RandomForestRegressor(
                        n_estimators=int(parameters["n_estimators"]),
                        max_features=float(parameters["max_features"]),
                        min_samples_leaf=int(parameters["min_samples_leaf"]),
                        max_depth=None, criterion="squared_error", bootstrap=True,
                        max_samples=None, random_state=42, n_jobs=8,
                    )
                    native_threads = 1
                print(f"START timing: {method} {case} {target}", flush=True)
                with threadpool_limits(limits=native_threads):
                    started = time.perf_counter()
                    model.fit(train, y_train[target])
                    fit_seconds = time.perf_counter() - started
                    prediction_seconds = []
                    for _ in range(3):
                        started = time.perf_counter()
                        prediction = model.predict(validation)
                        prediction_seconds.append(time.perf_counter() - started)
                assert np.isfinite(prediction).all()
                row = {
                    "method": method, "case": case, "target": target,
                    "input_features": train.shape[1],
                    "training_galaxies": len(train), "validation_galaxies": len(validation),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": float(np.median(prediction_seconds)),
                    "validation_MAE": float(mean_absolute_error(y_validation[target], prediction)),
                    "prediction_repetitions": 3,
                }
                if method == "forest":
                    row.update({name: model.get_params()[name]
                                for name in ["n_estimators", "max_features", "min_samples_leaf"]})
                temporary = result_path.with_suffix(".json.part")
                temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
                temporary.replace(result_path)
                with (output / f"{method}_{case}_{target}_predictions.npz").open("wb") as stream:
                    np.savez_compressed(stream, predictions=prediction,
                                        validation_ids=np.asarray(validation.index, dtype=str))
                print(f"DONE timing: {method} {case} {target}; "
                      f"fit={fit_seconds:.4f}s; predict={row['predict_seconds']:.4f}s; "
                      f"MAE={row['validation_MAE']:.6f}", flush=True)
                del model
            rows.append(row)
            results = pd.DataFrame(rows)
            temporary = output / "runtime_comparison.csv.part"
            results.to_csv(temporary, index=False)
            temporary.replace(output / "runtime_comparison.csv")
    return pd.DataFrame(rows)
