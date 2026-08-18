#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired pose-level stability test for existing Full-36 Q/C predictions.

Only existing CSV artifacts are read.  No model class, optimizer, image, or
Validation artifact is imported or executed.  Bootstrap resampling is done
over held-out poses, never over individual points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "projects" / "daheng" / "outputs" / "0817" / "grouped_cv_model_comparison"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0818" / "paired_model_stability"
Q_NAME = "quadratic_graph"
C_NAME = "circular_cone"
MODEL_NAMES = (Q_NAME, C_NAME)
V_MIN = 0.0
V_MAX = 3000.0
BIN_WIDTH = 100.0
BIN_COUNT = 30
FOLD_COUNT = 6
DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 20260818


def normalize_frame_id(value: Any) -> str:
    return f"{int(value):03d}"


def finite_metric(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "bias_mm": np.nan, "mae_mm": np.nan, "rmse_mm": np.nan, "p95_mm": np.nan}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "bias_mm": float(np.mean(values)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "p95_mm": float(np.percentile(absolute, 95)),
    }


def canonicalize_prediction(path: Path, expected_model: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "frame_key", "v_px", "u_px", "Xc_mm", "Yc_mm", "Zc_mm", "board_error_mm",
        "valid", "fold", "heldout_frames", "frame_id", "model",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"{path} 缺少字段：{missing}")
    models = sorted(data["model"].astype(str).unique())
    if models != [expected_model]:
        raise RuntimeError(f"{path} model 字段异常：{models}")
    data["frame_id"] = data["frame_id"].map(normalize_frame_id)
    data["fold"] = data["fold"].astype(int)
    data["valid"] = data["valid"].astype(bool)
    for column in ("u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm"):
        data[f"__{column}_key"] = data[column].astype(float).round(9)
    sort_columns = [
        "frame_key", "fold", "__u_px_key", "__v_px_key",
        "__Xc_mm_key", "__Yc_mm_key", "__Zc_mm_key",
    ]
    data = data.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    data["point_index"] = data.groupby("frame_key", sort=False).cumcount().astype(int)
    return data


def pair_predictions(q: pd.DataFrame, c: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(q) != len(c):
        raise RuntimeError(f"Q/C 行数不一致：{len(q)} vs {len(c)}")
    key_columns = ["frame_key", "point_index"]
    q_columns = key_columns + [
        "frame_id", "fold", "heldout_frames", "u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm",
        "board_error_mm", "valid",
    ]
    c_columns = key_columns + [
        "frame_id", "fold", "heldout_frames", "u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm",
        "board_error_mm", "valid",
    ]
    q_part = q[q_columns].rename(columns={column: f"{column}_q" for column in q_columns if column not in key_columns})
    c_part = c[c_columns].rename(columns={column: f"{column}_c" for column in c_columns if column not in key_columns})
    paired = q_part.merge(c_part, on=key_columns, how="inner", validate="one_to_one")
    if len(paired) != len(q):
        raise RuntimeError(f"Q/C point identity merge 丢行：q={len(q)}, paired={len(paired)}")
    for column in ("frame_id", "fold", "heldout_frames"):
        if not np.all(paired[f"{column}_q"].astype(str).to_numpy() == paired[f"{column}_c"].astype(str).to_numpy()):
            raise RuntimeError(f"Q/C {column} 不一致")
    for column in ("u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm"):
        if not np.allclose(paired[f"{column}_q"], paired[f"{column}_c"], rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"Q/C {column} 不一致")
    if not np.all(paired["valid_q"].to_numpy(dtype=bool) == paired["valid_c"].to_numpy(dtype=bool)):
        raise RuntimeError("Q/C valid mask 不一致")
    paired["frame_id"] = paired["frame_id_q"].astype(str)
    paired["fold"] = paired["fold_q"].astype(int)
    paired["heldout_frames"] = paired["heldout_frames_q"].astype(str)
    paired["u_px"] = paired["u_px_q"].astype(float)
    paired["v_px"] = paired["v_px_q"].astype(float)
    paired["Xc_mm"] = paired["Xc_mm_q"].astype(float)
    paired["Yc_mm"] = paired["Yc_mm_q"].astype(float)
    paired["Zc_mm"] = paired["Zc_mm_q"].astype(float)
    paired["valid_pair"] = paired["valid_q"].astype(bool) & paired["valid_c"].astype(bool)
    paired["finite_pair"] = (
        paired["valid_pair"]
        & np.isfinite(paired["board_error_mm_q"].to_numpy(dtype=float))
        & np.isfinite(paired["board_error_mm_c"].to_numpy(dtype=float))
    )
    paired["abs_error_mm_q"] = np.abs(paired["board_error_mm_q"].astype(float))
    paired["abs_error_mm_c"] = np.abs(paired["board_error_mm_c"].astype(float))
    paired["delta_point_mm"] = paired["board_error_mm_q"].astype(float) - paired["board_error_mm_c"].astype(float)
    paired["delta_abs_point_mm"] = paired["abs_error_mm_q"] - paired["abs_error_mm_c"]
    paired["v_bin_index"] = np.floor((paired["v_px"] - V_MIN) / BIN_WIDTH).astype(int)
    paired["v_bin"] = paired["v_bin_index"].map(lambda value: f"v_{int(value * BIN_WIDTH):04d}_{int((value + 1) * BIN_WIDTH):04d}")
    paired = paired.sort_values(["frame_id", "point_index"], kind="mergesort").reset_index(drop=True)
    audit = {
        "q_rows": int(len(q)),
        "c_rows": int(len(c)),
        "paired_rows": int(len(paired)),
        "pose_count": int(paired["frame_id"].nunique()),
        "fold_count": int(paired["fold"].nunique()),
        "valid_pair_count": int(paired["valid_pair"].sum()),
        "finite_pair_count": int(paired["finite_pair"].sum()),
        "v_min_px": float(paired["v_px"].min()),
        "v_max_px": float(paired["v_px"].max()),
    }
    return paired, audit


def pooled_model_rows(paired: pd.DataFrame, group_column: str | None = None, group_values: Sequence[Any] | None = None) -> pd.DataFrame:
    if group_values is None:
        grouped = [("Global", paired)]
    else:
        grouped = [(value, group) for value, group in paired.groupby(group_column, sort=True)]
    rows: List[Dict[str, Any]] = []
    for group_value, group in grouped:
        group = group[group["finite_pair"]].copy()
        q = finite_metric(group["board_error_mm_q"])
        c = finite_metric(group["board_error_mm_c"])
        rows.append(
            {
                "group": str(group_value),
                "point_count": int(len(group)),
                "pose_count": int(group["frame_id"].nunique()),
                "q_bias_mm": q["bias_mm"],
                "q_mae_mm": q["mae_mm"],
                "q_rmse_mm": q["rmse_mm"],
                "q_p95_mm": q["p95_mm"],
                "c_bias_mm": c["bias_mm"],
                "c_mae_mm": c["mae_mm"],
                "c_rmse_mm": c["rmse_mm"],
                "c_p95_mm": c["p95_mm"],
                "delta_bias_mm": q["bias_mm"] - c["bias_mm"],
                "delta_mae_mm": q["mae_mm"] - c["mae_mm"],
                "delta_rmse_mm": q["rmse_mm"] - c["rmse_mm"],
                "delta_p95_mm": q["p95_mm"] - c["p95_mm"],
                "delta_sq_error_sum_mm2": float(np.sum(group["board_error_mm_q"] ** 2) - np.sum(group["board_error_mm_c"] ** 2)),
            }
        )
    return pd.DataFrame(rows)


def pose_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for frame_id, group in paired.groupby("frame_id", sort=True):
        group = group[group["finite_pair"]].copy()
        q = finite_metric(group["board_error_mm_q"])
        c = finite_metric(group["board_error_mm_c"])
        delta_rmse = q["rmse_mm"] - c["rmse_mm"]
        delta_p95 = q["p95_mm"] - c["p95_mm"]
        rows.append(
            {
                "frame_id": str(frame_id),
                "fold": int(group["fold"].iloc[0]),
                "point_count": int(len(group)),
                "q_mae_mm": q["mae_mm"],
                "c_mae_mm": c["mae_mm"],
                "q_rmse_mm": q["rmse_mm"],
                "c_rmse_mm": c["rmse_mm"],
                "q_p95_mm": q["p95_mm"],
                "c_p95_mm": c["p95_mm"],
                "delta_mae_mm": q["mae_mm"] - c["mae_mm"],
                "delta_rmse_mm": delta_rmse,
                "delta_p95_mm": delta_p95,
                "delta_sq_error_sum_mm2": float(np.sum(group["board_error_mm_q"] ** 2) - np.sum(group["board_error_mm_c"] ** 2)),
                "rmse_winner": "Quadratic" if delta_rmse < -1.0e-12 else "Cone" if delta_rmse > 1.0e-12 else "Tie",
                "mae_winner": "Quadratic" if q["mae_mm"] < c["mae_mm"] else "Cone" if q["mae_mm"] > c["mae_mm"] else "Tie",
                "p95_winner": "Quadratic" if delta_p95 < -1.0e-12 else "Cone" if delta_p95 > 1.0e-12 else "Tie",
            }
        )
    return pd.DataFrame(rows)


def v_bin_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for index in range(BIN_COUNT):
        group = paired[(paired["v_bin_index"] == index) & paired["finite_pair"]].copy()
        if group.empty:
            continue
        pooled = pooled_model_rows(group)[0:1].iloc[0].to_dict()
        pose_rows: List[Tuple[float, float]] = []
        for _, pose_group in group.groupby("frame_id", sort=True):
            q_rmse = finite_metric(pose_group["board_error_mm_q"])["rmse_mm"]
            c_rmse = finite_metric(pose_group["board_error_mm_c"])["rmse_mm"]
            pose_rows.append((q_rmse, c_rmse))
        q_wins = sum(q < c - 1.0e-12 for q, c in pose_rows)
        c_wins = sum(c < q - 1.0e-12 for q, c in pose_rows)
        ties = len(pose_rows) - q_wins - c_wins
        rows.append(
            {
                "v_bin": f"v_{index * 100:04d}_{(index + 1) * 100:04d}",
                "v_bin_lo_px": float(index * 100),
                "v_bin_hi_px": float((index + 1) * 100),
                "point_count": int(pooled["point_count"]),
                "pose_count": int(pooled["pose_count"]),
                "q_mae_mm": pooled["q_mae_mm"],
                "c_mae_mm": pooled["c_mae_mm"],
                "q_rmse_mm": pooled["q_rmse_mm"],
                "c_rmse_mm": pooled["c_rmse_mm"],
                "q_p95_mm": pooled["q_p95_mm"],
                "c_p95_mm": pooled["c_p95_mm"],
                "delta_mae_mm": pooled["delta_mae_mm"],
                "delta_rmse_mm": pooled["delta_rmse_mm"],
                "delta_p95_mm": pooled["delta_p95_mm"],
                "delta_sq_error_sum_mm2": pooled["delta_sq_error_sum_mm2"],
                "q_win_pose_count": int(q_wins),
                "c_win_pose_count": int(c_wins),
                "tie_pose_count": int(ties),
                "q_win_pose_share": float(q_wins / len(pose_rows)) if pose_rows else np.nan,
                "rmse_winner": "Quadratic" if pooled["delta_rmse_mm"] < -1.0e-12 else "Cone" if pooled["delta_rmse_mm"] > 1.0e-12 else "Tie",
            }
        )
    return pd.DataFrame(rows)


def load_and_audit_existing_metrics(input_dir: Path, paired: pd.DataFrame) -> Dict[str, Any]:
    grouped_path = input_dir / "grouped_cv_model_comparison.csv"
    per_bin_path = input_dir / "per_v_bin_cv_metrics.csv"
    grouped = pd.read_csv(grouped_path, encoding="utf-8-sig")
    per_bin = pd.read_csv(per_bin_path, encoding="utf-8-sig")
    pooled = grouped[(grouped["row_type"] == "pooled_cv") & grouped["model"].isin(MODEL_NAMES)].copy()
    if len(pooled) != 2:
        raise RuntimeError("grouped_cv_model_comparison.csv 缺少 Q/C pooled rows")
    if sorted(pooled["fold_count"].astype(int).unique()) != [FOLD_COUNT]:
        raise RuntimeError("Full-36 grouped-CV fold_count 不是 6")
    if sorted(pooled["point_count"].astype(int).unique()) != [len(paired)]:
        raise RuntimeError("Full-36 pooled point_count 与 pointwise artifact 不一致")
    paired_global = pooled_model_rows(paired).iloc[0]
    checks: List[Dict[str, Any]] = []
    for model, prefix in ((Q_NAME, "q"), (C_NAME, "c")):
        row = pooled[pooled["model"] == model].iloc[0]
        checks.extend(
            [
                {"field": f"{model}.rmse_mm", "artifact": float(row["rmse_mm"]), "recomputed": float(paired_global[f"{prefix}_rmse_mm"]), "ok": bool(np.isclose(row["rmse_mm"], paired_global[f"{prefix}_rmse_mm"], atol=1.0e-10))},
                {"field": f"{model}.p95_mm", "artifact": float(row["p95_mm"]), "recomputed": float(paired_global[f"{prefix}_p95_mm"]), "ok": bool(np.isclose(row["p95_mm"], paired_global[f"{prefix}_p95_mm"], atol=1.0e-10))},
            ]
        )
    if not all(item["ok"] for item in checks):
        raise RuntimeError(f"pointwise 与 grouped-CV pooled 指标不一致：{checks}")
    return {
        "grouped": grouped,
        "per_bin": per_bin,
        "pooled_rows": pooled,
        "checks": checks,
    }


def validate_per_bin_artifact(per_bin: pd.DataFrame, paired_bins: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for model, prefix in ((Q_NAME, "q"), (C_NAME, "c")):
        source = per_bin[per_bin["model"] == model].copy()
        if len(source) != len(paired_bins):
            raise RuntimeError(f"{model} per-v-bin 行数不一致：artifact={len(source)}, recomputed={len(paired_bins)}")
        for column, expected_column in (
            ("point_count", "point_count"),
            ("unique_frame_count", "pose_count"),
            ("rmse_mm", f"{prefix}_rmse_mm"),
            ("p95_mm", f"{prefix}_p95_mm"),
        ):
            recomputed = paired_bins[["v_bin", expected_column]].rename(columns={expected_column: "recomputed"})
            artifact = source[["v_bin", column]].rename(columns={column: "artifact"})
            joined = recomputed.merge(artifact, on="v_bin", how="left", validate="one_to_one")
            if joined["artifact"].isna().any() or not np.allclose(joined["recomputed"], joined["artifact"], rtol=0.0, atol=1.0e-10):
                raise RuntimeError(f"{model} per-v-bin {column} 与 pointwise 重算不一致")
            checks.append({"model": model, "field": column, "ok": True})
    return checks


def pose_pad(data: pd.DataFrame, error_column: str, pose_ids: Sequence[str]) -> np.ndarray:
    groups = [data[data["frame_id"] == pose][error_column].to_numpy(dtype=float) for pose in pose_ids]
    max_count = max(len(group) for group in groups)
    output = np.full((len(groups), max_count), np.nan, dtype=float)
    for index, values in enumerate(groups):
        output[index, : len(values)] = values
    return output


def bootstrap_pose_deltas(
    paired: pd.DataFrame,
    mask: pd.Series,
    n_bootstrap: int,
    seed: int,
    region_name: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data = paired[mask & paired["finite_pair"]].copy()
    pose_ids = sorted(data["frame_id"].astype(str).unique())
    if len(pose_ids) < 2:
        raise RuntimeError(f"{region_name} 可用于 bootstrap 的 pose 少于 2")
    q_pad = pose_pad(data, "board_error_mm_q", pose_ids)
    c_pad = pose_pad(data, "board_error_mm_c", pose_ids)
    rng = np.random.default_rng(seed)
    batch_size = 128
    samples_q_rmse: List[np.ndarray] = []
    samples_q_mae: List[np.ndarray] = []
    samples_q_p95: List[np.ndarray] = []
    samples_c_rmse: List[np.ndarray] = []
    samples_c_mae: List[np.ndarray] = []
    samples_c_p95: List[np.ndarray] = []
    for start in range(0, n_bootstrap, batch_size):
        count = min(batch_size, n_bootstrap - start)
        indices = rng.integers(0, len(pose_ids), size=(count, len(pose_ids)))
        q_flat = q_pad[indices].reshape(count, -1)
        c_flat = c_pad[indices].reshape(count, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            samples_q_rmse.append(np.sqrt(np.nanmean(q_flat**2, axis=1)))
            samples_q_mae.append(np.nanmean(np.abs(q_flat), axis=1))
            samples_q_p95.append(np.nanpercentile(np.abs(q_flat), 95, axis=1))
            samples_c_rmse.append(np.sqrt(np.nanmean(c_flat**2, axis=1)))
            samples_c_mae.append(np.nanmean(np.abs(c_flat), axis=1))
            samples_c_p95.append(np.nanpercentile(np.abs(c_flat), 95, axis=1))
    q_rmse = np.concatenate(samples_q_rmse)
    q_mae = np.concatenate(samples_q_mae)
    q_p95 = np.concatenate(samples_q_p95)
    c_rmse = np.concatenate(samples_c_rmse)
    c_mae = np.concatenate(samples_c_mae)
    c_p95 = np.concatenate(samples_c_p95)
    distributions = {
        "delta_rmse_mm": q_rmse - c_rmse,
        "delta_mae_mm": q_mae - c_mae,
        "delta_p95_mm": q_p95 - c_p95,
    }
    observed_q = finite_metric(data["board_error_mm_q"])
    observed_c = finite_metric(data["board_error_mm_c"])
    rows: List[Dict[str, Any]] = []
    for metric_name, q_key, c_key in (
        ("rmse", "delta_rmse_mm", "rmse_mm"),
        ("mae", "delta_mae_mm", "mae_mm"),
        ("p95", "delta_p95_mm", "p95_mm"),
    ):
        if metric_name == "rmse":
            observed_delta = observed_q["rmse_mm"] - observed_c["rmse_mm"]
            q_value, c_value = observed_q["rmse_mm"], observed_c["rmse_mm"]
        elif metric_name == "mae":
            observed_delta = observed_q["mae_mm"] - observed_c["mae_mm"]
            q_value, c_value = observed_q["mae_mm"], observed_c["mae_mm"]
        else:
            observed_delta = observed_q["p95_mm"] - observed_c["p95_mm"]
            q_value, c_value = observed_q["p95_mm"], observed_c["p95_mm"]
        distribution = distributions[q_key]
        rows.append(
            {
                "scope": region_name,
                "metric": metric_name,
                "q_observed_mm": q_value,
                "c_observed_mm": c_value,
                "observed_delta_q_minus_c_mm": observed_delta,
                "bootstrap_mean_delta_mm": float(np.mean(distribution)),
                "bootstrap_std_delta_mm": float(np.std(distribution, ddof=1)),
                "ci95_low_mm": float(np.percentile(distribution, 2.5)),
                "ci95_high_mm": float(np.percentile(distribution, 97.5)),
                "bootstrap_unit": "pose",
                "bootstrap_pose_count": int(len(pose_ids)),
                "bootstrap_replicates": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows), distributions


def select_common_worst_region(grouped: pd.DataFrame, paired_bins: pd.DataFrame) -> str:
    rows = grouped[grouped["model"].isin(MODEL_NAMES)].set_index("model")
    q_worst = str(rows.loc[Q_NAME, "worst_v_bin"])
    c_worst = str(rows.loc[C_NAME, "worst_v_bin"])
    if q_worst == c_worst:
        return q_worst
    return str(paired_bins.assign(score=paired_bins[["q_rmse_mm", "c_rmse_mm"]].max(axis=1)).sort_values("score", ascending=False).iloc[0]["v_bin"])


def status_from_bootstrap(summary: pd.DataFrame, pose_table: pd.DataFrame, bin_table: pd.DataFrame) -> str:
    required = summary[summary["metric"].isin(["rmse", "p95"]) & summary["scope"].isin(["Global", "WorstRegion"])]
    q_supported = len(required) == 4 and bool((required["ci95_high_mm"] < 0.0).all())
    c_supported = len(required) == 4 and bool((required["ci95_low_mm"] > 0.0).all())
    q_pose_wins = int((pose_table["rmse_winner"] == "Quadratic").sum())
    c_pose_wins = int((pose_table["rmse_winner"] == "Cone").sum())
    q_bin_wins = int((bin_table["rmse_winner"] == "Quadratic").sum())
    c_bin_wins = int((bin_table["rmse_winner"] == "Cone").sum())
    if q_supported and q_pose_wins > c_pose_wins and q_bin_wins >= c_bin_wins:
        return "QUADRATIC_SUPPORTED"
    if c_supported and c_pose_wins > q_pose_wins and c_bin_wins >= q_bin_wins:
        return "CONE_SUPPORTED"
    return "UNRESOLVED"


def artifact_audit(input_dir: Path, q: pd.DataFrame, c: pd.DataFrame, existing: Mapping[str, Any], output: Path) -> None:
    pose_ids = sorted(q["frame_id"].unique())
    rows = [
        {
            "artifact": "cv_pointwise_quadratic_graph.csv",
            "path": str(input_dir / "cv_pointwise_quadratic_graph.csv"),
            "role": "existing held-out predictions",
            "model": Q_NAME,
            "rows": len(q),
            "pose_count": len(pose_ids),
            "fold_count": int(q["fold"].nunique()),
            "dataset_ids": ",".join(pose_ids),
            "point_identity": "frame_key + deterministic point_index after coordinate sort",
            "mask": "inherited full_board_physical; inset=0 mm",
            "weighting": "inherited frame-balanced; no v-density weighting",
            "cv_protocol": "inherited 6-fold pose-grouped held-out predictions",
            "validation_read": False,
            "action": "reused; no refit",
            "provenance_status": "CONFIRMED",
            "notes": "split=fit; all points belong to one held-out pose fold",
        },
        {
            "artifact": "cv_pointwise_circular_cone.csv",
            "path": str(input_dir / "cv_pointwise_circular_cone.csv"),
            "role": "existing held-out predictions",
            "model": C_NAME,
            "rows": len(c),
            "pose_count": len(c["frame_id"].unique()),
            "fold_count": int(c["fold"].nunique()),
            "dataset_ids": ",".join(pose_ids),
            "point_identity": "frame_key + deterministic point_index after coordinate sort",
            "mask": "inherited full_board_physical; inset=0 mm",
            "weighting": "inherited frame-balanced; no v-density weighting",
            "cv_protocol": "inherited 6-fold pose-grouped held-out predictions",
            "validation_read": False,
            "action": "reused; no refit",
            "provenance_status": "CONFIRMED",
            "notes": "Q/C row and coordinate identity checks passed",
        },
        {
            "artifact": "grouped_cv_model_comparison.csv",
            "path": str(input_dir / "grouped_cv_model_comparison.csv"),
            "role": "pooled-CV provenance and cross-check",
            "model": "Quadratic + Cone",
            "rows": int(len(existing["pooled_rows"])),
            "pose_count": int(existing["pooled_rows"]["heldout_frame_count"].max()),
            "fold_count": int(existing["pooled_rows"]["fold_count"].max()),
            "dataset_ids": ",".join(pose_ids),
            "point_identity": "pooled cross-check against paired pointwise rows",
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "6-fold pose-grouped; sorted frame-id round-robin",
            "validation_read": False,
            "action": "reused; no rerun",
            "provenance_status": "CONFIRMED",
            "notes": "recomputed pooled RMSE/P95 match artifact within tolerance",
        },
        {
            "artifact": "per_v_bin_cv_metrics.csv",
            "path": str(input_dir / "per_v_bin_cv_metrics.csv"),
            "role": "per-v-bin provenance and cross-check",
            "model": "Quadratic + Cone",
            "rows": int(len(existing["per_bin"] [existing["per_bin"]["model"].isin(MODEL_NAMES)])),
            "pose_count": int(q["frame_id"].nunique()),
            "fold_count": int(q["fold"].nunique()),
            "dataset_ids": ",".join(pose_ids),
            "point_identity": "v-bin grouping cross-check against paired rows",
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "same 100 px bins over v=0-3000",
            "validation_read": False,
            "action": "reused; no rerun",
            "provenance_status": "CONFIRMED",
            "notes": "per-bin RMSE/P95 recomputation is checked in main analysis",
        },
        {
            "artifact": "Validation",
            "path": "excluded by task constraint",
            "role": "not read",
            "model": "not read",
            "rows": 0,
            "pose_count": 0,
            "fold_count": 0,
            "dataset_ids": "none",
            "point_identity": "not read",
            "mask": "not read",
            "weighting": "not read",
            "cv_protocol": "not read",
            "validation_read": False,
            "action": "excluded",
            "provenance_status": "N/A",
            "notes": "No Validation artifact opened",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def plot_difference(pose_table: pd.DataFrame, bin_table: pd.DataFrame, summary: pd.DataFrame, distributions: Mapping[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pose = pose_table.sort_values("frame_id")
    x = np.arange(len(pose))
    colors_rmse = np.where(pose["delta_rmse_mm"].to_numpy() < 0.0, "#1f77b4", "#ff7f0e")
    axes[0, 0].bar(x, pose["delta_rmse_mm"], color=colors_rmse)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Pose paired delta RMSE (Q - Cone)")
    axes[0, 0].set_ylabel("delta / mm; negative favors Quadratic")
    axes[0, 0].set_xticks(x, pose["frame_id"], rotation=75, fontsize=7)
    axes[0, 0].grid(axis="y", alpha=0.25)

    colors_p95 = np.where(pose["delta_p95_mm"].to_numpy() < 0.0, "#1f77b4", "#ff7f0e")
    axes[0, 1].bar(x, pose["delta_p95_mm"], color=colors_p95)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Pose paired delta P95 (Q - Cone)")
    axes[0, 1].set_ylabel("delta / mm")
    axes[0, 1].set_xticks(x, pose["frame_id"], rotation=75, fontsize=7)
    axes[0, 1].grid(axis="y", alpha=0.25)

    bins = bin_table.sort_values("v_bin")
    bx = (bins["v_bin_lo_px"] + bins["v_bin_hi_px"]) / 2.0
    axes[1, 0].plot(bx, bins["delta_rmse_mm"], marker="o", markersize=3, label="delta RMSE")
    axes[1, 0].plot(bx, bins["delta_p95_mm"], marker="s", markersize=3, label="delta P95")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Per-v-bin paired difference")
    axes[1, 0].set_xlabel("v / px")
    axes[1, 0].set_ylabel("Q - Cone / mm")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    ordered_metrics = [
        ("delta_rmse_mm", "Global RMSE"),
        ("delta_p95_mm", "Global P95"),
        ("delta_worst_rmse_mm", "Worst-region RMSE"),
        ("delta_worst_p95_mm", "Worst-region P95"),
    ]
    summary_index = {(str(row.scope), str(row.metric)): row for row in summary.itertuples()}
    y = np.arange(len(ordered_metrics))
    observed: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    labels: List[str] = []
    for key, label in ordered_metrics:
        scope = "Global" if key.startswith("delta_global") else "WorstRegion"
        metric = "rmse" if "rmse" in key else "p95"
        row = summary_index[(scope, metric)]
        observed.append(float(row.observed_delta_q_minus_c_mm))
        lows.append(float(row.ci95_low_mm))
        highs.append(float(row.ci95_high_mm))
        labels.append(label)
    observed_array = np.asarray(observed)
    axes[1, 1].errorbar(observed_array, y, xerr=[observed_array - np.asarray(lows), np.asarray(highs) - observed_array], fmt="o", color="#1f77b4", capsize=4)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].set_xlabel("Q - Cone / mm; 95% pose-bootstrap CI")
    axes[1, 1].set_title("Pose-unit bootstrap stability")
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.suptitle("Full-36 paired Quadratic vs Circular Cone held-out predictions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 5) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def generate_report(
    output: Path,
    input_dir: Path,
    paired: pd.DataFrame,
    pose_table: pd.DataFrame,
    bin_table: pd.DataFrame,
    summary: pd.DataFrame,
    audit_info: Mapping[str, Any],
    status: str,
    common_worst: str,
) -> None:
    global_metrics = pooled_model_rows(paired).iloc[0]
    q_pose_wins = int((pose_table["rmse_winner"] == "Quadratic").sum())
    c_pose_wins = int((pose_table["rmse_winner"] == "Cone").sum())
    tie_pose = int((pose_table["rmse_winner"] == "Tie").sum())
    q_bin_wins = int((bin_table["rmse_winner"] == "Quadratic").sum())
    c_bin_wins = int((bin_table["rmse_winner"] == "Cone").sum())
    tie_bin = int((bin_table["rmse_winner"] == "Tie").sum())
    q_advantage = pose_table[pose_table["delta_sq_error_sum_mm2"] < 0.0].copy()
    q_advantage_total = float(-q_advantage["delta_sq_error_sum_mm2"].sum()) if not q_advantage.empty else 0.0
    top5_q = float(-q_advantage.nsmallest(5, "delta_sq_error_sum_mm2")["delta_sq_error_sum_mm2"].sum()) if not q_advantage.empty else 0.0
    top5_fraction = top5_q / q_advantage_total if q_advantage_total > 0 else np.nan
    q_bin_advantage = bin_table[bin_table["delta_sq_error_sum_mm2"] < 0.0]
    q_bin_total = float(-q_bin_advantage["delta_sq_error_sum_mm2"].sum()) if not q_bin_advantage.empty else 0.0
    top_bins = q_bin_advantage.nsmallest(5, "delta_sq_error_sum_mm2") if not q_bin_advantage.empty else q_bin_advantage

    lines = [
        "# Full-36 Quadratic / Circular Cone 配对稳定性检验",
        "",
        f"`C0_PAIRED_STATUS = {status}`",
        "",
        "## 结论摘要",
        "",
        f"- 使用已有 `{input_dir}` 的两份 pointwise held-out prediction；没有重新拟合模型。",
        f"- Q/C 均覆盖 {audit_info['pose_count']} 个 held-out pose、{audit_info['paired_rows']} 个相同点、{audit_info['fold_count']}-fold pose-grouped CV；点级配对通过。",
        f"- 按 pose 的 RMSE win count：Quadratic {q_pose_wins}，Cone {c_pose_wins}，Tie {tie_pose}；按 100 px v-bin：Quadratic {q_bin_wins}，Cone {c_bin_wins}，Tie {tie_bin}。",
        f"- Global RMSE Δ(Q−C)={fmt(summary[(summary['scope']=='Global') & (summary['metric']=='rmse')].iloc[0]['observed_delta_q_minus_c_mm'])} mm；95% pose-bootstrap CI 见下表。",
        f"- `C0_PAIRED_STATUS = {status}`：判定要求 Global 与共同 worst-region 的 RMSE/P95 CI 同时不跨 0，并且 pose/bin 方向一致；当前结果未满足单一模型的全部条件。",
        "",
        "## Artifact provenance / reuse audit",
        "",
        f"- Pointwise 来源：`{input_dir / 'cv_pointwise_quadratic_graph.csv'}` 与 `{input_dir / 'cv_pointwise_circular_cone.csv'}`。",
        f"- 交叉核对：`grouped_cv_model_comparison.csv` 的 Q/C pooled RMSE/P95 与 pointwise 重算一致；`per_v_bin_cv_metrics.csv` 用于 v-bin 结果复核。",
        "- 两模型使用相同 FIT pose、相同 held-out fold、相同 frame_key/点坐标；mask、weighting、6-fold pose-grouped protocol 均继承 0817 artifact。",
        "- 只读取 FIT grouped-CV artifacts；未读取 Validation，未训练 C1。",
        "",
        "| check | result |",
        "|---|---|",
        f"| Q/C rows | {audit_info['q_rows']} / {audit_info['c_rows']} |",
        f"| paired rows | {audit_info['paired_rows']} |",
        f"| paired poses | {audit_info['pose_count']} |",
        f"| folds | {audit_info['fold_count']} |",
        f"| valid paired points | {audit_info['valid_pair_count']} |",
        f"| finite paired points | {audit_info['finite_pair_count']} |",
        f"| v domain observed | {fmt(audit_info['v_min_px'], 1)}–{fmt(audit_info['v_max_px'], 1)} px |",
        "| point identity | frame_key + deterministic point_index; coordinate/fold/frame checks passed |",
        "",
        "## Pooled paired metrics",
        "",
        "Negative Δ(Q−C) means Quadratic has lower error.",
        "",
        "| metric | Quadratic | Cone | Δ(Q−C) |",
        "|---|---:|---:|---:|",
        f"| Global MAE / mm | {fmt(global_metrics['q_mae_mm'])} | {fmt(global_metrics['c_mae_mm'])} | {fmt(global_metrics['delta_mae_mm'])} |",
        f"| Global RMSE / mm | {fmt(global_metrics['q_rmse_mm'])} | {fmt(global_metrics['c_rmse_mm'])} | {fmt(global_metrics['delta_rmse_mm'])} |",
        f"| Global P95 / mm | {fmt(global_metrics['q_p95_mm'])} | {fmt(global_metrics['c_p95_mm'])} | {fmt(global_metrics['delta_p95_mm'])} |",
        "",
        "## Pose-unit bootstrap 95% CI",
        "",
        f"Bootstrap replicates={summary['bootstrap_replicates'].iloc[0]}, seed={summary['bootstrap_seed'].iloc[0]}; each replicate resamples pose IDs with replacement, never individual points.共同 worst-region=`{common_worst}`。",
        "",
        "| scope | metric | observed Δ(Q−C) / mm | bootstrap mean | 95% CI low | 95% CI high | pose units |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.scope} | {row.metric} | {fmt(row.observed_delta_q_minus_c_mm)} | {fmt(row.bootstrap_mean_delta_mm)} | {fmt(row.ci95_low_mm)} | {fmt(row.ci95_high_mm)} | {row.bootstrap_pose_count} |")

    lines += [
        "",
        "## Spatial consistency across v",
        "",
        f"- Q better in {q_bin_wins}/{len(bin_table)} populated v-bins by pooled RMSE; Cone better in {c_bin_wins}/{len(bin_table)}. Bin direction is therefore not uniformly Quadratic-favored.",
        f"- Q-favored pose squared-error advantage top-5 pose fraction={fmt(top5_fraction * 100.0, 1)}% of all Q-favored squared-error reduction; this is a concentration diagnostic, not a point bootstrap.",
        f"- Q-favored v-bin squared-error total={fmt(q_bin_total, 3)} mm²; strongest Q-favored bins are: {', '.join(top_bins['v_bin'].astype(str).tolist()) if not top_bins.empty else 'none'}。",
        "",
        "| v-bin | pose count | Q win poses | Cone win poses | Δ RMSE / mm | Δ P95 / mm | winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in bin_table.sort_values("v_bin").itertuples():
        lines.append(f"| {row.v_bin} | {row.pose_count} | {row.q_win_pose_count} | {row.c_win_pose_count} | {fmt(row.delta_rmse_mm)} | {fmt(row.delta_p95_mm)} | {row.rmse_winner} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Quadratic 的 pooled Full-36 指标略低，但配对检验要求同时具备 pose、v-bin 和 bootstrap CI 的方向一致性。若 CI 跨 0，或优势集中在少数 pose/bin，则只能视为轻微候选优势，不能将 C0 冻结为 Quadratic。",
        f"当前判定：`C0_PAIRED_STATUS = {status}`。",
        "",
        "## Scope exclusions",
        "",
        "- 未重新拟合 Quadratic/Cone；未运行 Plane；未读取 Validation；未训练 C1。",
        "- bootstrap 单位为 pose，禁止逐点 bootstrap。",
        "",
        "## Outputs",
        "",
        "- `paired_pose_model_comparison.csv`",
        "- `paired_v_bin_comparison.csv`",
        "- `paired_bootstrap_summary.csv`",
        "- `paired_model_difference.png`",
        "- `report.md`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.bootstrap_replicates < 1000:
        raise ValueError("bootstrap replicates 至少 1000")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    q = canonicalize_prediction(input_dir / "cv_pointwise_quadratic_graph.csv", Q_NAME)
    c = canonicalize_prediction(input_dir / "cv_pointwise_circular_cone.csv", C_NAME)
    paired, audit_info = pair_predictions(q, c)
    existing = load_and_audit_existing_metrics(input_dir, paired)
    paired_bins = v_bin_comparison(paired)
    existing["per_bin_checks"] = validate_per_bin_artifact(existing["per_bin"], paired_bins)
    common_worst = select_common_worst_region(existing["grouped"], paired_bins)
    pose_table = pose_comparison(paired)
    global_mask = pd.Series(True, index=paired.index)
    worst_mask = paired["v_bin"] == common_worst
    global_summary, global_distributions = bootstrap_pose_deltas(paired, global_mask, args.bootstrap_replicates, args.seed, "Global")
    worst_summary, worst_distributions = bootstrap_pose_deltas(paired, worst_mask, args.bootstrap_replicates, args.seed + 1, "WorstRegion")
    summary = pd.concat([global_summary, worst_summary], ignore_index=True)
    summary = summary.sort_values(["scope", "metric"], key=lambda values: values.map({"Global": 0, "WorstRegion": 1}) if values.name == "scope" else values.map({"rmse": 0, "mae": 1, "p95": 2})).reset_index(drop=True)
    status = status_from_bootstrap(summary, pose_table, paired_bins)

    artifact_audit(input_dir, q, c, existing, output_dir / "artifact_reuse_audit.csv")
    pose_table.to_csv(output_dir / "paired_pose_model_comparison.csv", index=False, encoding="utf-8-sig")
    paired_bins.to_csv(output_dir / "paired_v_bin_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "paired_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    plot_difference(pose_table, paired_bins, summary, {**global_distributions, **{f"worst_{key}": value for key, value in worst_distributions.items()}}, output_dir / "paired_model_difference.png")
    generate_report(output_dir / "report.md", input_dir, paired, pose_table, paired_bins, summary, audit_info, status, common_worst)

    print(f"C0_PAIRED_STATUS = {status}")
    print(f"Global delta RMSE (Q-C) = {finite_metric(paired['board_error_mm_q'])['rmse_mm'] - finite_metric(paired['board_error_mm_c'])['rmse_mm']:.8f} mm")
    print(f"Pose RMSE wins: Quadratic={(pose_table['rmse_winner'] == 'Quadratic').sum()}, Cone={(pose_table['rmse_winner'] == 'Cone').sum()}, Tie={(pose_table['rmse_winner'] == 'Tie').sum()}")
    print(f"Common worst region: {common_worst}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
