"""Checkpointed, sequential random-forest tuning on training/validation data."""

from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class SequentialRandomForestTuner:
    def __init__(self, X_train, y_train, X_validation, y_validation,
                 output_dir, random_state=42, n_jobs=8):
        if not X_train.index.equals(y_train.index):
            raise ValueError("Training input and target rows do not match")
        if not X_validation.index.equals(y_validation.index):
            raise ValueError("Validation input and target rows do not match")
        if not X_train.columns.equals(X_validation.columns):
            raise ValueError("Training and validation features do not match")
        if set(X_train.index) & set(X_validation.index):
            raise ValueError("Training and validation galaxies overlap")
        for frame in (X_train, y_train, X_validation, y_validation):
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Tuning requires finite inputs and targets")
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.targets = list(y_train.columns)
        self.random_state = random_state
        self.n_jobs = n_jobs
        digest = hashlib.sha256()
        for frame in (X_train, y_train, X_validation, y_validation):
            digest.update(str(frame.shape).encode())
            digest.update("\n".join(map(str, frame.index)).encode())
            digest.update("\n".join(map(str, frame.columns)).encode())
            digest.update(np.ascontiguousarray(frame.to_numpy()).tobytes())
        digest.update(f"sklearn={sklearn.__version__};seed={random_state}".encode())
        self.fingerprint = digest.hexdigest()
        self.output_dir = Path(output_dir) / self.fingerprint[:16]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "data_fingerprint": self.fingerprint,
            "sklearn_version": sklearn.__version__, "random_state": random_state,
            "training_galaxies": len(X_train), "validation_galaxies": len(X_validation),
            "input_features": X_train.shape[1], "targets": self.targets,
            "criterion": "squared_error", "bootstrap": True,
            "max_samples": None, "max_depth": None,
            "selection_metric": "validation_MAE", "test_set_used": False,
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")

    def _paths(self, target, n_estimators, max_features, min_samples_leaf):
        name = (f"{target}_trees-{int(n_estimators)}_"
                f"features-{float(max_features):g}_leaf-{int(min_samples_leaf)}")
        return self.output_dir / f"{name}.joblib", self.output_dir / f"{name}.json"

    def fit_trial(self, target, n_estimators, max_features, min_samples_leaf,
                  warm_model=None):
        model_path, result_path = self._paths(
            target, n_estimators, max_features, min_samples_leaf)
        label = (f"{target}: trees={n_estimators}, max_features={max_features:g}, "
                 f"min_samples_leaf={min_samples_leaf}")
        if model_path.exists() and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            model = joblib.load(model_path)
            print(f"CACHED {label}; MAE={result['validation_MAE']:.6f}", flush=True)
            return model, result

        previous_seconds = 0.0
        if warm_model is not None:
            params = warm_model.get_params()
            if (params["n_estimators"] >= n_estimators
                    or params["max_features"] != max_features
                    or params["min_samples_leaf"] != min_samples_leaf):
                raise ValueError("Warm start is only valid when adding trees with fixed settings")
            previous_seconds = getattr(warm_model, "_sdss_training_seconds", 0.0)
            model = warm_model
            model.set_params(n_estimators=int(n_estimators), warm_start=True, n_jobs=self.n_jobs)
        else:
            model = RandomForestRegressor(
                n_estimators=int(n_estimators), max_features=float(max_features),
                min_samples_leaf=int(min_samples_leaf), max_depth=None,
                bootstrap=True, max_samples=None, criterion="squared_error",
                random_state=self.random_state, n_jobs=self.n_jobs, warm_start=True)

        print(f"START {label}", flush=True)
        started = time.perf_counter()
        model.fit(self.X_train, self.y_train[target])
        incremental_seconds = time.perf_counter() - started
        model._sdss_training_seconds = previous_seconds + incremental_seconds
        predictions = model.predict(self.X_validation)
        if not np.isfinite(predictions).all():
            raise ValueError("The forest produced non-finite validation predictions")
        result = {
            "target": target, "n_estimators": int(n_estimators),
            "max_features": float(max_features), "min_samples_leaf": int(min_samples_leaf),
            "validation_MAE": float(mean_absolute_error(self.y_validation[target], predictions)),
            "training_seconds": model._sdss_training_seconds,
            "incremental_fit_seconds": incremental_seconds,
            "model_file": model_path.name,
        }
        # Commit the model before its result so an interrupted write cannot look complete.
        temporary_model = model_path.with_suffix(".joblib.part")
        joblib.dump(model, temporary_model, compress=3)
        temporary_model.replace(model_path)
        temporary_result = result_path.with_suffix(".json.part")
        temporary_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary_result.replace(result_path)
        print(f"DONE {label}; MAE={result['validation_MAE']:.6f}; "
              f"training={result['training_seconds']:.1f}s", flush=True)
        return model, result

    def _checkpoint(self, rows, name):
        frame = pd.DataFrame(rows)
        path = self.output_dir / name
        temporary = path.with_suffix(".csv.part")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        return frame

    def tune_tree_counts(self, candidates):
        rows = []
        for target in self.targets:
            warm_model = None
            for count in sorted(candidates):
                warm_model, result = self.fit_trial(target, count, 1.0, 1, warm_model)
                rows.append(result)
                self._checkpoint(rows, "stage1_tree_counts.csv")
        return pd.DataFrame(rows)

    def tune_feature_fractions(self, candidates, best_tree_counts):
        rows = []
        for target in self.targets:
            for fraction in candidates:
                model, result = self.fit_trial(target, best_tree_counts[target], fraction, 1)
                rows.append(result)
                self._checkpoint(rows, "stage2_feature_sampling.csv")
                del model
        return pd.DataFrame(rows)

    def tune_leaf_sizes(self, candidates, best_tree_counts, best_feature_fractions):
        rows = []
        for target in self.targets:
            for leaf_size in candidates:
                model, result = self.fit_trial(
                    target, best_tree_counts[target], best_feature_fractions[target], leaf_size)
                rows.append(result)
                self._checkpoint(rows, "stage3_leaf_sizes.csv")
                del model
        return pd.DataFrame(rows)

    @staticmethod
    def best_per_target(results):
        # Stable sorting keeps the first candidate in an exact MAE tie.
        return (results.sort_values("validation_MAE", kind="stable")
                .drop_duplicates("target").set_index("target"))

    def load_selected_models(self, best_results):
        models = {}
        for target, row in best_results.iterrows():
            path = self.output_dir / row["model_file"]
            models[target] = joblib.load(path)
        best_results.to_csv(self.output_dir / "best_settings.csv")
        predictions = pd.DataFrame(
            {target: model.predict(self.X_validation) for target, model in models.items()},
            index=self.X_validation.index)
        predictions.to_csv(self.output_dir / "best_validation_predictions.csv")
        return models
