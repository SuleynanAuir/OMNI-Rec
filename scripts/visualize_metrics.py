import json
import math
import os
from collections import Counter

import fire
import numpy as np
import pandas as pd


def _clean_text(value):
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("\n").strip()


def _load_item_catalog(item_path):
    actual_path = item_path
    if not actual_path.endswith(".txt") and os.path.exists(f"{actual_path}.txt"):
        actual_path = f"{actual_path}.txt"
    if not os.path.exists(actual_path):
        return set()

    with open(actual_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    item_ids = set()
    for line in lines:
        parts = line.split("\t")
        if not parts:
            continue
        item_ids.add(_clean_text(parts[0]))
    return item_ids


def _load_train_popularity(train_file):
    if not train_file or not os.path.exists(train_file):
        return Counter(), 0

    train_df = pd.read_csv(train_file)
    if "item_sid" in train_df.columns:
        values = train_df["item_sid"].astype(str).map(_clean_text)
    elif "item_id" in train_df.columns:
        values = train_df["item_id"].astype(str).map(_clean_text)
    else:
        return Counter(), 0

    counter = Counter(values.tolist())
    return counter, sum(counter.values())


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean_ci95(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    mean_value = float(np.mean(array))
    if array.size == 1:
        return mean_value, 0.0
    std_value = float(np.std(array, ddof=1))
    ci95 = 1.96 * std_value / math.sqrt(array.size)
    return mean_value, float(ci95)


def _gini_index(counts):
    array = np.asarray(counts, dtype=float)
    if array.size == 0:
        return 0.0
    if np.all(array == 0):
        return 0.0
    sorted_array = np.sort(array)
    n = sorted_array.size
    cumulative = np.cumsum(sorted_array)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def _normalized_entropy(counter):
    if not counter:
        return 0.0
    counts = np.asarray(list(counter.values()), dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    entropy = -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return float(entropy / max_entropy)


def _tail_frequency_threshold(pop_counter):
    if not pop_counter:
        return 0
    counts = np.asarray(list(pop_counter.values()), dtype=float)
    return int(np.quantile(counts, 0.2))


def _create_output_layout(output_dir):
    metrics_dir = os.path.join(output_dir, "metrics")
    figures_dir = os.path.join(output_dir, "figures")
    ranking_dir = os.path.join(figures_dir, "ranking")
    quality_dir = os.path.join(figures_dir, "quality")
    distribution_dir = os.path.join(figures_dir, "distribution")
    diagnostics_dir = os.path.join(figures_dir, "diagnostics")

    for directory in [
        output_dir,
        metrics_dir,
        figures_dir,
        ranking_dir,
        quality_dir,
        distribution_dir,
        diagnostics_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    return {
        "output": output_dir,
        "metrics": metrics_dir,
        "figures": figures_dir,
        "ranking": ranking_dir,
        "quality": quality_dir,
        "distribution": distribution_dir,
        "diagnostics": diagnostics_dir,
    }


def _prepare_plot_style():
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")


def _compute_metrics(records, item_set, pop_counter, pop_total, topk_list):
    max_beam = max((len(record.get("predict", [])) for record in records), default=0)
    valid_topk = [k for k in topk_list if k <= max_beam] if max_beam > 0 else []
    tail_threshold = _tail_frequency_threshold(pop_counter)

    metrics = {
        "K": valid_topk,
        "HR": [],
        "HR_CI95": [],
        "NDCG": [],
        "NDCG_CI95": [],
        "MRR": [],
        "MRR_CI95": [],
        "Precision": [],
        "Precision_CI95": [],
        "Recall": [],
        "Recall_CI95": [],
        "F1": [],
        "F1_CI95": [],
        "MAP": [],
        "MAP_CI95": [],
        "InvalidRate": [],
        "Coverage": [],
        "UniqueRatio": [],
        "Novelty": [],
        "Novelty_CI95": [],
        "PopularityBias": [],
        "PopularityBias_CI95": [],
        "TailItemRatio": [],
        "Entropy": [],
        "Gini": [],
    }

    first_hit_ranks = []
    top1_counter = Counter()
    all_pred_counter = Counter()
    invalid_count_all = 0
    all_pred_count = 0

    for record in records:
        predicts = [_clean_text(item) for item in record.get("predict", [])]
        target = record.get("output", "")
        if isinstance(target, list):
            target = target[0] if target else ""
        target = _clean_text(target)

        if predicts:
            top1_counter[predicts[0]] += 1
            all_pred_counter.update(predicts)

        hit_rank = None
        for index, item in enumerate(predicts):
            if item == target:
                hit_rank = index + 1
                break
        first_hit_ranks.append(hit_rank if hit_rank is not None else 0)

        for item in predicts:
            all_pred_count += 1
            if item_set and item not in item_set:
                invalid_count_all += 1

    prediction_counter_by_k = {}

    for k in valid_topk:
        hit_scores = []
        ndcg_scores = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        map_scores = []
        invalid_values = []
        coverage_pool = set()
        unique_ratio_values = []
        novelty_values = []
        pop_bias_values = []
        tail_ratio_values = []
        pred_counter_at_k = Counter()

        for record in records:
            predicts = [_clean_text(item) for item in record.get("predict", [])][:k]
            target = record.get("output", "")
            if isinstance(target, list):
                target = target[0] if target else ""
            target = _clean_text(target)

            hit_rank = None
            for index, item in enumerate(predicts):
                if item == target:
                    hit_rank = index + 1
                    break

            hit_value = 1.0 if hit_rank is not None else 0.0
            hit_scores.append(hit_value)
            ndcg_scores.append(1.0 / math.log2(hit_rank + 1.0) if hit_rank is not None else 0.0)
            mrr_scores.append(1.0 / hit_rank if hit_rank is not None else 0.0)

            precision = hit_value / len(predicts) if predicts else 0.0
            recall = hit_value
            f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            ap_value = 1.0 / hit_rank if hit_rank is not None else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            map_scores.append(ap_value)

            if len(predicts) == 0:
                invalid_values.append(0.0)
                unique_ratio_values.append(0.0)
                novelty_values.append(0.0)
                pop_bias_values.append(0.0)
                tail_ratio_values.append(0.0)
                continue

            if item_set:
                invalid_values.append(sum(item not in item_set for item in predicts) / len(predicts))
            else:
                invalid_values.append(0.0)

            coverage_pool.update(predicts)
            pred_counter_at_k.update(predicts)
            unique_ratio_values.append(len(set(predicts)) / len(predicts))

            if pop_counter and pop_total > 0:
                novelty = []
                pop_percent = []
                tail_hits = 0
                for item in predicts:
                    freq = pop_counter.get(item, 0)
                    novelty.append(-math.log2((freq + 1.0) / (pop_total + len(pop_counter) + 1.0)))
                    pop_percent.append(freq / pop_total)
                    if freq <= tail_threshold:
                        tail_hits += 1
                novelty_values.append(float(np.mean(novelty)))
                pop_bias_values.append(float(np.mean(pop_percent)))
                tail_ratio_values.append(tail_hits / len(predicts))
            else:
                novelty_values.append(0.0)
                pop_bias_values.append(0.0)
                tail_ratio_values.append(0.0)

        prediction_counter_by_k[k] = pred_counter_at_k

        metric_mean, metric_ci = _mean_ci95(hit_scores)
        metrics["HR"].append(metric_mean)
        metrics["HR_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(ndcg_scores)
        metrics["NDCG"].append(metric_mean)
        metrics["NDCG_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(mrr_scores)
        metrics["MRR"].append(metric_mean)
        metrics["MRR_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(precision_scores)
        metrics["Precision"].append(metric_mean)
        metrics["Precision_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(recall_scores)
        metrics["Recall"].append(metric_mean)
        metrics["Recall_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(f1_scores)
        metrics["F1"].append(metric_mean)
        metrics["F1_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(map_scores)
        metrics["MAP"].append(metric_mean)
        metrics["MAP_CI95"].append(metric_ci)

        metrics["InvalidRate"].append(float(np.mean(invalid_values)) if invalid_values else 0.0)
        metrics["Coverage"].append(_safe_div(len(coverage_pool), len(item_set)) if item_set else 0.0)
        metrics["UniqueRatio"].append(float(np.mean(unique_ratio_values)) if unique_ratio_values else 0.0)

        metric_mean, metric_ci = _mean_ci95(novelty_values)
        metrics["Novelty"].append(metric_mean)
        metrics["Novelty_CI95"].append(metric_ci)

        metric_mean, metric_ci = _mean_ci95(pop_bias_values)
        metrics["PopularityBias"].append(metric_mean)
        metrics["PopularityBias_CI95"].append(metric_ci)

        metrics["TailItemRatio"].append(float(np.mean(tail_ratio_values)) if tail_ratio_values else 0.0)
        metrics["Entropy"].append(_normalized_entropy(pred_counter_at_k))
        metrics["Gini"].append(_gini_index(list(pred_counter_at_k.values())))

    summary = {
        "num_samples": len(records),
        "max_beam": max_beam,
        "valid_item_count": len(item_set),
        "global_invalid_rate": _safe_div(invalid_count_all, all_pred_count),
        "hit_rate_any": _safe_div(sum(rank > 0 for rank in first_hit_ranks), len(first_hit_ranks)),
        "mean_first_hit_rank": float(np.mean([rank for rank in first_hit_ranks if rank > 0])) if any(rank > 0 for rank in first_hit_ranks) else 0.0,
        "median_first_hit_rank": float(np.median([rank for rank in first_hit_ranks if rank > 0])) if any(rank > 0 for rank in first_hit_ranks) else 0.0,
        "tail_frequency_threshold": tail_threshold,
    }

    return metrics, summary, first_hit_ranks, top1_counter, all_pred_counter, prediction_counter_by_k


def _save_metrics_tables(metrics, summary, output_layout, top1_counter, all_pred_counter):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_layout["metrics"], "metrics_table.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_layout["output"], "metrics_table.csv"), index=False)

    corr_columns = [
        "HR",
        "NDCG",
        "MRR",
        "Precision",
        "Recall",
        "F1",
        "MAP",
        "InvalidRate",
        "Coverage",
        "UniqueRatio",
        "Novelty",
        "PopularityBias",
        "TailItemRatio",
        "Entropy",
        "Gini",
    ]
    available_columns = [column for column in corr_columns if column in metrics_df.columns]
    corr_df = metrics_df[available_columns].corr(method="pearson") if len(available_columns) >= 2 else pd.DataFrame()
    corr_df.to_csv(os.path.join(output_layout["metrics"], "metrics_correlation.csv"), index=True)

    top1_df = pd.DataFrame(top1_counter.most_common(), columns=["item", "count"])
    top1_df.to_csv(os.path.join(output_layout["metrics"], "top1_frequency.csv"), index=False)
    top1_df.to_csv(os.path.join(output_layout["output"], "top1_frequency.csv"), index=False)

    all_pred_df = pd.DataFrame(all_pred_counter.most_common(), columns=["item", "count"])
    all_pred_df.to_csv(os.path.join(output_layout["metrics"], "all_prediction_frequency.csv"), index=False)

    payload = {
        "summary": summary,
        "per_k": metrics_df.to_dict(orient="records"),
        "output_layout": output_layout,
    }
    with open(os.path.join(output_layout["metrics"], "metrics_summary.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    with open(os.path.join(output_layout["output"], "metrics_summary.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return metrics_df, corr_df


def _plot_curves(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=220)
    for metric_name, ci_name, marker in [
        ("HR", "HR_CI95", "o"),
        ("NDCG", "NDCG_CI95", "s"),
        ("MRR", "MRR_CI95", "^")
    ]:
        y_values = np.asarray(metrics[metric_name], dtype=float)
        ci_values = np.asarray(metrics[ci_name], dtype=float)
        ax.plot(k_values, y_values, marker=marker, linewidth=2.0, label=f"{metric_name}@K")
        ax.fill_between(k_values, np.clip(y_values - ci_values, 0.0, 1.0), np.clip(y_values + ci_values, 0.0, 1.0), alpha=0.15)

    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Ranking Quality vs. K (95% CI)")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_ranking_quality_curves_ci95.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_quality_safety(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(k_values, metrics["InvalidRate"], marker="o", linewidth=2.0, label="Invalid Rate@K")
    ax.plot(k_values, metrics["Coverage"], marker="s", linewidth=2.0, label="Coverage@K")
    ax.plot(k_values, metrics["UniqueRatio"], marker="^", linewidth=2.0, label="Unique Ratio@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Ratio")
    ax.set_title("Generation Quality and Safety vs. K")
    ax.set_xticks(k_values)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_quality_safety_curves.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_long_tail(all_pred_counter, figures_dir):
    import matplotlib.pyplot as plt

    if not all_pred_counter:
        return

    sorted_counts = sorted(all_pred_counter.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(ranks, sorted_counts, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted Item Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Predicted Item Long-Tail Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_long_tail_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_first_hit(first_hit_ranks, figures_dir):
    import matplotlib.pyplot as plt

    positive_ranks = [rank for rank in first_hit_ranks if rank > 0]
    if not positive_ranks:
        return

    max_rank = max(positive_ranks)
    bins = np.arange(1, max_rank + 2) - 0.5
    counts, _ = np.histogram(positive_ranks, bins=bins)
    cumulative = np.cumsum(counts) / max(len(first_hit_ranks), 1)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=180)
    ax1.bar(np.arange(1, max_rank + 1), counts, alpha=0.75, label="Hit Count")
    ax1.set_xlabel("First Hit Rank")
    ax1.set_ylabel("Count")
    ax1.set_title("First-Hit Rank Distribution")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, max_rank + 1), cumulative, color="crimson", linewidth=2.0, marker="o", label="Cumulative Hit Rate")
    ax2.set_ylabel("Cumulative Ratio")
    ax2.set_ylim(0, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_first_hit_rank_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_bias_novelty(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=180)
    ax1.plot(k_values, metrics["Novelty"], color="tab:blue", marker="o", linewidth=2.0, label="Novelty@K")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Novelty")
    ax1.set_title("Novelty and Popularity Bias vs. K")
    ax1.set_xticks(k_values)

    ax2 = ax1.twinx()
    ax2.plot(k_values, metrics["PopularityBias"], color="tab:orange", marker="s", linewidth=2.0, label="Popularity Bias@K")
    ax2.set_ylabel("Average Popularity")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_novelty_popularity_bias.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_precision_recall_f1(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=220)
    ax.plot(k_values, metrics["Precision"], marker="o", linewidth=2.0, label="Precision@K")
    ax.plot(k_values, metrics["Recall"], marker="s", linewidth=2.0, label="Recall@K")
    ax.plot(k_values, metrics["F1"], marker="^", linewidth=2.0, label="F1@K")
    ax.plot(k_values, metrics["MAP"], marker="d", linewidth=2.0, label="MAP@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 / MAP vs. K")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "06_precision_recall_f1_map.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_coverage_tradeoff(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    hr_values = np.asarray(metrics["HR"], dtype=float)
    coverage_values = np.asarray(metrics["Coverage"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=220)
    scatter = ax.scatter(coverage_values, hr_values, c=k_values, cmap="viridis", s=72)
    for index, k_value in enumerate(k_values):
        ax.annotate(f"K={k_value}", (coverage_values[index], hr_values[index]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    color_bar = fig.colorbar(scatter, ax=ax)
    color_bar.set_label("K")
    ax.set_xlabel("Coverage@K")
    ax.set_ylabel("HR@K")
    ax.set_title("Accuracy-Coverage Trade-off")
    ax.set_xlim(0, min(1.05, max(0.05, float(np.max(coverage_values)) + 0.05)))
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "07_accuracy_coverage_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_top1_bar(top1_counter, figures_dir, top_n=20):
    import matplotlib.pyplot as plt

    if not top1_counter:
        return

    rows = top1_counter.most_common(top_n)
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=220)
    ax.bar(np.arange(len(labels)), values, color="#4c72b0")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_xlabel("Item ID")
    ax.set_ylabel("Top-1 Frequency")
    ax.set_title(f"Top-{top_n} Most Frequent Top-1 Predictions")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "08_top1_frequency_bar_top20.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_lorenz_curve(all_pred_counter, figures_dir):
    import matplotlib.pyplot as plt

    if not all_pred_counter:
        return

    counts = np.asarray(sorted(all_pred_counter.values()), dtype=float)
    cumulative_counts = np.cumsum(counts)
    lorenz_y = np.insert(cumulative_counts / cumulative_counts[-1], 0, 0.0)
    lorenz_x = np.linspace(0.0, 1.0, lorenz_y.size)
    gini_value = _gini_index(counts)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=220)
    ax.plot(lorenz_x, lorenz_y, linewidth=2.0, label=f"Lorenz Curve (Gini={gini_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.4, label="Perfect Equality")
    ax.set_xlabel("Cumulative Share of Items")
    ax.set_ylabel("Cumulative Share of Predictions")
    ax.set_title("Exposure Concentration (Lorenz Curve)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "09_lorenz_curve.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_metric_correlation(corr_df, figures_dir):
    import matplotlib.pyplot as plt

    if corr_df.empty:
        return

    matrix = corr_df.values
    labels = corr_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=220)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Metric Correlation Heatmap")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.2f}", ha="center", va="center", fontsize=6)

    color_bar = fig.colorbar(image, ax=ax)
    color_bar.set_label("Pearson Correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "10_metric_correlation_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_gini(metrics, figures_dir):
    import matplotlib.pyplot as plt

    k_values = metrics["K"]
    if not k_values:
        return

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4), dpi=220)
    ax1.plot(k_values, metrics["Entropy"], color="tab:purple", marker="o", linewidth=2.0, label="Normalized Entropy@K")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Entropy")
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(k_values, metrics["Gini"], color="tab:red", marker="s", linewidth=2.0, label="Gini@K")
    ax2.set_ylabel("Gini")
    ax2.set_ylim(0, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title("Diversity Entropy and Concentration")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "11_entropy_gini_curves.png"), bbox_inches="tight")
    plt.close(fig)


def _generate_file_manifest(output_dir):
    manifest = []
    for root_path, _, file_names in os.walk(output_dir):
        for file_name in sorted(file_names):
            relative_path = os.path.relpath(os.path.join(root_path, file_name), output_dir)
            manifest.append(relative_path)
    return manifest


def analyze(
    path,
    item_path,
    train_file="",
    output_dir="",
    topk_list="1,3,5,10,20,50",
):
    import matplotlib.pyplot as plt  # noqa: F401

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(path), "analysis")
    output_layout = _create_output_layout(output_dir)

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if isinstance(topk_list, str):
        topk = [int(value.strip()) for value in topk_list.split(",") if value.strip()]
    else:
        topk = list(topk_list)
    topk = sorted(list(set(topk)))

    item_set = _load_item_catalog(item_path)
    pop_counter, pop_total = _load_train_popularity(train_file)

    _prepare_plot_style()
    metrics, summary, first_hit_ranks, top1_counter, all_pred_counter, _ = _compute_metrics(
        records=records,
        item_set=item_set,
        pop_counter=pop_counter,
        pop_total=pop_total,
        topk_list=topk,
    )

    metrics_df, corr_df = _save_metrics_tables(
        metrics=metrics,
        summary=summary,
        output_layout=output_layout,
        top1_counter=top1_counter,
        all_pred_counter=all_pred_counter,
    )

    _plot_curves(metrics, output_layout["ranking"])
    _plot_precision_recall_f1(metrics, output_layout["ranking"])
    _plot_accuracy_coverage_tradeoff(metrics, output_layout["ranking"])

    _plot_quality_safety(metrics, output_layout["quality"])
    _plot_bias_novelty(metrics, output_layout["quality"])

    _plot_long_tail(all_pred_counter, output_layout["distribution"])
    _plot_first_hit(first_hit_ranks, output_layout["distribution"])
    _plot_top1_bar(top1_counter, output_layout["distribution"], top_n=20)
    _plot_lorenz_curve(all_pred_counter, output_layout["distribution"])

    _plot_metric_correlation(corr_df, output_layout["diagnostics"])
    _plot_entropy_gini(metrics, output_layout["diagnostics"])

    manifest = _generate_file_manifest(output_dir)
    manifest_path = os.path.join(output_layout["output"], "manifest.json")
    manifest_with_self = sorted(set(manifest + ["manifest.json"]))
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({"files": manifest_with_self}, file, ensure_ascii=False, indent=2)

    print(f"[Analysis] records={len(records)}")
    print(f"[Analysis] output_dir={output_dir}")
    print(f"[Analysis] metrics_rows={len(metrics_df)}")
    print("[Analysis] generated files:")
    for file_path in manifest_with_self:
        print(f"  - {file_path}")


if __name__ == "__main__":
    fire.Fire(analyze)
