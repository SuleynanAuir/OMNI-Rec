import json
import os
from typing import Dict, List, Optional, Tuple

import fire
import pandas as pd


def _resolve_trainer_state(path: str) -> str:
    if os.path.isfile(path):
        if os.path.basename(path) != "trainer_state.json":
            raise FileNotFoundError(f"Expected trainer_state.json file, got: {path}")
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    candidates: List[Tuple[float, str]] = []
    for root, _, files in os.walk(path):
        if "trainer_state.json" in files:
            file_path = os.path.join(root, "trainer_state.json")
            candidates.append((os.path.getmtime(file_path), file_path))

    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found under directory: {path}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_metrics(state_path: str) -> pd.DataFrame:
    with open(state_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    log_history = data.get("log_history", [])
    rows: List[Dict[str, float]] = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        row: Dict[str, float] = {"step": float(step)}

        if "loss" in entry:
            row["loss"] = float(entry["loss"])
        elif "train/loss" in entry:
            row["loss"] = float(entry["train/loss"])

        if "grad_norm" in entry:
            row["grad_norm"] = float(entry["grad_norm"])
        elif "train/grad_norm" in entry:
            row["grad_norm"] = float(entry["train/grad_norm"])

        if "learning_rate" in entry:
            row["learning_rate"] = float(entry["learning_rate"])
        elif "train/learning_rate" in entry:
            row["learning_rate"] = float(entry["train/learning_rate"])

        if len(row) > 1:
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No metric entries found in {state_path}. Expected keys like loss/grad_norm/learning_rate."
        )

    frame = pd.DataFrame(rows)
    frame = frame.groupby("step", as_index=False).last().sort_values("step")
    return frame


def _prepare_output_dir(state_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        target_dir = output_dir
    else:
        parent_dir = os.path.dirname(state_path)
        target_dir = os.path.join(parent_dir, "training_analysis")

    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def _smooth_series(series: pd.Series, smooth_window: int) -> pd.Series:
    if smooth_window <= 1:
        return series
    return series.rolling(window=smooth_window, min_periods=1).mean()


def plot(
    path: str,
    output_dir: str = "",
    smooth_window: int = 1,
) -> None:
    import matplotlib.pyplot as plt

    state_path = _resolve_trainer_state(path)
    frame = _extract_metrics(state_path)
    target_dir = _prepare_output_dir(state_path, output_dir)

    csv_path = os.path.join(target_dir, "training_metrics.csv")
    frame.to_csv(csv_path, index=False)

    metrics = ["loss", "grad_norm", "learning_rate"]
    available = [name for name in metrics if name in frame.columns]

    if not available:
        raise ValueError("No supported metrics found to visualize.")

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.2 * len(available)), dpi=200)
    if len(available) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, available):
        y = _smooth_series(frame[metric_name], smooth_window=smooth_window)
        axis.plot(frame["step"], y, linewidth=1.8)
        axis.set_title(f"{metric_name} vs step")
        axis.set_xlabel("step")
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.25)

    fig.tight_layout()

    fig_path = os.path.join(target_dir, "training_metrics.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[TrainViz] trainer_state={state_path}")
    print(f"[TrainViz] rows={len(frame)}")
    print(f"[TrainViz] csv={csv_path}")
    print(f"[TrainViz] fig={fig_path}")


if __name__ == "__main__":
    fire.Fire(plot)
