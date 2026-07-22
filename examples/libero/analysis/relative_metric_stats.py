"""Episode-level statistics for calibrating raw relative depth to metric z.

This analysis intentionally uses only the first three gripper points:
root, left_base, and right_base.  It reads raw parquet values, before the
training-time quantile normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.model_selection import GroupKFold


DEFAULT_DATASET = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_31_no_noops_1.0.0_lerobot_10hz"
)
POINT_NAMES = ("root", "left_base", "right_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/libero/analysis/output/relative_metric_first3"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_samples(dataset_dir: Path) -> pd.DataFrame:
    paths = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet episodes under {dataset_dir / 'data'}")

    records: list[dict[str, float | int | str]] = []
    columns = ["episode_index", "frame_index", "task_index", "subtask_id", "depths_rel", "gripper_uvd"]
    for path in paths:
        table = pq.read_table(path, columns=columns).to_pydict()
        for row_index in range(len(table["episode_index"])):
            depth = np.asarray(table["depths_rel"][row_index], dtype=np.float64)
            gripper = np.asarray(table["gripper_uvd"][row_index], dtype=np.float64)
            if depth.ndim != 2 or gripper.shape != (6, 3):
                raise ValueError(f"Unexpected shapes in {path}: depth={depth.shape}, gripper={gripper.shape}")
            height, width = depth.shape
            for point_id, point_name in enumerate(POINT_NAMES):
                u, v, z = gripper[point_id]
                ui, vi = int(np.rint(u)), int(np.rint(v))
                in_bounds = 0 <= ui < width and 0 <= vi < height
                d_rel = float(depth[np.clip(vi, 0, height - 1), np.clip(ui, 0, width - 1)])
                records.append(
                    {
                        "episode": int(table["episode_index"][row_index]),
                        "frame": int(table["frame_index"][row_index]),
                        "task": int(table["task_index"][row_index]),
                        "subtask": int(table["subtask_id"][row_index]),
                        "point_id": point_id,
                        "point": point_name,
                        "u": float(u),
                        "v": float(v),
                        "d_rel": d_rel if in_bounds else np.nan,
                        "z_metric": float(z),
                        "in_bounds": in_bounds,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["d_rel", "z_metric"])
    return frame.loc[frame["in_bounds"]].reset_index(drop=True)


def design_matrix(frame: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    d = frame["d_rel"].to_numpy()
    x = (frame["u"].to_numpy() - 127.5) / 127.5
    y = (frame["v"].to_numpy() - 127.5) / 127.5
    episode_max = frame.groupby("episode")["frame"].transform("max").to_numpy()
    tau = frame["frame"].to_numpy() / np.maximum(episode_max, 1.0)
    p1 = (frame["point_id"].to_numpy() == 1).astype(np.float64)
    p2 = (frame["point_id"].to_numpy() == 2).astype(np.float64)
    spatial = [x, y, x * x, x * y, y * y]
    spatial_names = ["x", "y", "x2", "xy", "y2"]

    values = [np.ones_like(d), d]
    names = ["intercept", "d_rel"]
    if model in {"spatial_shift", "spatial_both", "full"}:
        values.extend(spatial)
        names.extend(spatial_names)
    if model in {"spatial_scale", "spatial_both", "full"}:
        values.extend([d * value for value in spatial])
        names.extend([f"d_{name}" for name in spatial_names])
    if model == "full":
        values.extend([tau, tau * tau, d * tau, p1, p2, d * p1, d * p2])
        names.extend(["time", "time2", "d_time", "left_base", "right_base", "d_left_base", "d_right_base"])
    return np.column_stack(values), names


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    x_train, _ = design_matrix(train, model)
    x_test, _ = design_matrix(test, model)
    y_train = train["z_metric"].to_numpy()
    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return x_test @ coef


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "bias_m": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error * error) / np.sum((y - y.mean()) ** 2)),
    }


def grouped_cv(frame: pd.DataFrame, models: list[str], folds: int) -> tuple[dict, pd.DataFrame]:
    episodes = frame["episode"].to_numpy()
    splitter = GroupKFold(n_splits=min(folds, frame["episode"].nunique()))
    predictions = {model: np.full(len(frame), np.nan) for model in models}
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, groups=episodes)):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        y = test["z_metric"].to_numpy()
        for model in models:
            prediction = fit_predict(train, test, model)
            predictions[model][test_index] = prediction
            fold_rows.append({"fold": fold, "model": model, **metrics(y, prediction)})

    result = {}
    prediction_frame = frame[["episode", "frame", "point_id", "z_metric"]].copy()
    for model in models:
        prediction_frame[model] = predictions[model]
        result[model] = metrics(frame["z_metric"].to_numpy(), predictions[model])
    return result, prediction_frame


def bootstrap_episode_improvements(
    prediction_frame: pd.DataFrame, models: list[str], repetitions: int, seed: int
) -> dict[str, dict[str, float]]:
    by_episode = []
    for episode, group in prediction_frame.groupby("episode"):
        row = {"episode": int(episode)}
        y = group["z_metric"].to_numpy()
        for model in models:
            row[model] = float(np.mean(np.abs(group[model].to_numpy() - y)))
        by_episode.append(row)
    episode_metrics = pd.DataFrame(by_episode)
    rng = np.random.default_rng(seed)
    count = len(episode_metrics)
    output = {}
    for model in models[1:]:
        differences = np.empty(repetitions)
        for index in range(repetitions):
            sample = rng.integers(0, count, size=count)
            differences[index] = np.mean(
                episode_metrics["global"].to_numpy()[sample]
                - episode_metrics[model].to_numpy()[sample]
            )
        output[model] = {
            "mae_improvement_m": float(differences.mean()),
            "ci95_low_m": float(np.quantile(differences, 0.025)),
            "ci95_high_m": float(np.quantile(differences, 0.975)),
        }
    return output


def bootstrap_global_coefficients(frame: pd.DataFrame, repetitions: int, seed: int) -> dict:
    groups = [group for _, group in frame.groupby("episode")]
    rng = np.random.default_rng(seed)
    coefficients = np.empty((repetitions, 2))
    for index in range(repetitions):
        sampled = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        x, _ = design_matrix(sampled, "global")
        coefficients[index], *_ = np.linalg.lstsq(x, sampled["z_metric"].to_numpy(), rcond=None)
    return {
        "intercept": float(coefficients[:, 0].mean()),
        "intercept_ci95": np.quantile(coefficients[:, 0], [0.025, 0.975]).tolist(),
        "scale": float(coefficients[:, 1].mean()),
        "scale_ci95": np.quantile(coefficients[:, 1], [0.025, 0.975]).tolist(),
        "bootstrap_scale_intercept_correlation": float(np.corrcoef(coefficients.T)[0, 1]),
    }


def oracle_episode_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode, group in frame.groupby("episode"):
        x, _ = design_matrix(group, "global")
        y = group["z_metric"].to_numpy()
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        prediction = x @ coef
        rows.append(
            {
                "episode": int(episode),
                "samples": len(group),
                "scale": float(coef[1]),
                "shift": float(coef[0]),
                "mae_m": float(np.mean(np.abs(prediction - y))),
                "rel_std": float(group["d_rel"].std()),
                "design_condition": float(np.linalg.cond(x)),
            }
        )
    return pd.DataFrame(rows)


def frame_identifiability(frame: pd.DataFrame) -> dict[str, float]:
    spreads = []
    conditions = []
    leave_one_out_errors = []
    for _, group in frame.groupby(["episode", "frame"]):
        if len(group) != 3:
            continue
        d = group["d_rel"].to_numpy()
        z = group["z_metric"].to_numpy()
        spreads.append(float(np.ptp(d)))
        conditions.append(float(np.linalg.cond(np.column_stack([np.ones(3), d]))))
        for held_out in range(3):
            keep = np.arange(3) != held_out
            if abs(d[keep][1] - d[keep][0]) < 1e-8:
                continue
            coef = np.polyfit(d[keep], z[keep], 1)
            leave_one_out_errors.append(abs(float(np.polyval(coef, d[held_out]) - z[held_out])))
    return {
        "frames": len(spreads),
        "median_relative_range": float(np.median(spreads)),
        "p10_relative_range": float(np.quantile(spreads, 0.1)),
        "median_design_condition": float(np.median(conditions)),
        "p90_design_condition": float(np.quantile(conditions, 0.9)),
        "leave_one_point_out_mae_m": float(np.mean(leave_one_out_errors)),
        "leave_one_point_out_median_ae_m": float(np.median(leave_one_out_errors)),
    }


def residual_correlations(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    x, _ = design_matrix(frame, "global")
    coef, *_ = np.linalg.lstsq(x, frame["z_metric"].to_numpy(), rcond=None)
    residual = frame["z_metric"].to_numpy() - x @ coef
    result = {}
    for name in ["d_rel", "u", "v", "frame", "point_id"]:
        rho, p_value = stats.spearmanr(residual, frame[name].to_numpy())
        result[name] = {"spearman_rho": float(rho), "p_value_naive": float(p_value)}
    return result


def make_figure(frame: pd.DataFrame, episode_stats: pd.DataFrame, cv: dict, path: Path) -> None:
    x, _ = design_matrix(frame, "global")
    coef, *_ = np.linalg.lstsq(x, frame["z_metric"].to_numpy(), rcond=None)
    residual_cm = 100.0 * (frame["z_metric"].to_numpy() - x @ coef)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for point_name, group in frame.groupby("point"):
        sample = group.iloc[:: max(1, len(group) // 1500)]
        axes[0, 0].scatter(sample["d_rel"], sample["z_metric"], s=5, alpha=0.35, label=point_name)
    axes[0, 0].set(xlabel="raw relative depth", ylabel="metric camera z (m)", title="First three gripper points")
    axes[0, 0].legend()

    scatter = axes[0, 1].scatter(frame["u"], frame["v"], c=residual_cm, s=4, alpha=0.35, cmap="coolwarm")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="u", ylabel="v", title="Global-affine residual (cm)")
    figure.colorbar(scatter, ax=axes[0, 1])

    axes[1, 0].scatter(episode_stats["scale"], episode_stats["shift"], c=episode_stats["episode"], cmap="viridis")
    axes[1, 0].set(xlabel="episode-oracle scale", ylabel="episode-oracle shift", title="Scale/shift coupling")

    model_names = list(cv)
    axes[1, 1].bar(model_names, [100.0 * cv[name]["mae_m"] for name in model_names])
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set(ylabel="episode-held-out MAE (cm)", title="Grouped cross-validation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_samples(args.dataset_dir)
    models = ["global", "spatial_shift", "spatial_scale", "spatial_both", "full"]
    cv, predictions = grouped_cv(frame, models, args.folds)
    episode_stats = oracle_episode_stats(frame)

    pearson = stats.pearsonr(frame["d_rel"], frame["z_metric"])
    spearman = stats.spearmanr(frame["d_rel"], frame["z_metric"])
    report = {
        "scope": {
            "points": list(POINT_NAMES),
            "episodes": int(frame["episode"].nunique()),
            "frames": int(frame[["episode", "frame"]].drop_duplicates().shape[0]),
            "samples": len(frame),
            "dataset_dir": str(args.dataset_dir),
        },
        "raw_association": {
            "pearson_r": float(pearson.statistic),
            "pearson_p_value_naive": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value_naive": float(spearman.pvalue),
        },
        "global_coefficients_cluster_bootstrap": bootstrap_global_coefficients(frame, args.bootstrap, args.seed),
        "episode_held_out_cv": cv,
        "cv_mae_improvement_over_global_cluster_bootstrap": bootstrap_episode_improvements(
            predictions, models, args.bootstrap, args.seed
        ),
        "global_residual_correlations": residual_correlations(frame),
        "frame_level_identifiability": frame_identifiability(frame),
        "episode_oracle_summary": episode_stats.describe().to_dict(),
    }

    frame.to_csv(args.output_dir / "samples_first3.csv", index=False)
    episode_stats.to_csv(args.output_dir / "episode_oracle_affine.csv", index=False)
    predictions.to_csv(args.output_dir / "grouped_cv_predictions.csv", index=False)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    make_figure(frame, episode_stats, cv, args.output_dir / "summary.png")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
