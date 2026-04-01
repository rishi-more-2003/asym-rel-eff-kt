"""
Generate publication-quality figures from experiment results.
Outputs PDF figures to documentation/figures/.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(__file__), "data", "runs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "documentation", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CONDITIONS = ["solo", "peer", "role_labeled", "relational"]
COND_LABELS = {
    "solo": "Solo",
    "peer": "Symmetric Peer",
    "role_labeled": "Role-Labeled",
    "relational": "Relational (Ours)",
}
COND_COLORS = {
    "solo": "#7f8c8d",
    "peer": "#3498db",
    "role_labeled": "#e67e22",
    "relational": "#e74c3c",
}
SEEDS = [1, 2, 3]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_metrics(condition: str, seed: int) -> list[dict]:
    path = os.path.join(RUNS_DIR, f"{condition}_seed{seed}", "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_results() -> dict:
    path = os.path.join(RUNS_DIR, "evaluation_results.json")
    with open(path) as f:
        return json.load(f)


def rolling_avg(values: list[float], window: int = 10) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i + 1]))
    return out


# ── Figure 1: Learning Curves ───────────────────────────────────────
def fig_learning_curves():
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for cond in CONDITIONS:
        all_curves = []
        max_ep = 0
        for seed in SEEDS:
            metrics = load_metrics(cond, seed)
            if not metrics:
                continue
            rewards = [m["reward"] for m in sorted(metrics, key=lambda x: x["episode"])]
            smoothed = rolling_avg(rewards, window=10)
            all_curves.append(smoothed)
            max_ep = max(max_ep, len(smoothed))

        if not all_curves:
            continue

        for i in range(len(all_curves)):
            while len(all_curves[i]) < max_ep:
                all_curves[i].append(all_curves[i][-1])

        arr = np.array(all_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        episodes = np.arange(len(mean))

        ax.plot(episodes, mean, label=COND_LABELS[cond], color=COND_COLORS[cond], linewidth=1.5)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=COND_COLORS[cond])

    ax.set_xlabel("Episode")
    ax.set_ylabel("Task Completion Rate (rolling avg, w=10)")
    ax.set_title("Learning Curves Across Conditions")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "learning_curves.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "learning_curves.png"))
    plt.close(fig)
    print("  [OK] learning_curves")


# ── Figure 2: H1 Transfer Accuracy ──────────────────────────────────
def fig_h1_transfer():
    eval_data = load_eval_results()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    means, stds, labels, colors = [], [], [], []
    for cond in CONDITIONS:
        h1_list = eval_data[cond].get("h1", [])
        if not h1_list:
            continue
        vals = [h["avg_completion_rate"] for h in h1_list]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(COND_LABELS[cond])
        colors.append(COND_COLORS[cond])

    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
                  linewidth=0.5, capsize=4, width=0.6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Avg. Completion Rate")
    ax.set_title("H1: Transfer Accuracy on Held-Out Tasks")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(os.path.join(FIG_DIR, "h1_transfer.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "h1_transfer.png"))
    plt.close(fig)
    print("  [OK] h1_transfer")


# ── Figure 3: Transfer by Difficulty ─────────────────────────────────
def fig_transfer_by_difficulty():
    eval_data = load_eval_results()
    fig, ax = plt.subplots(figsize=(7, 3.8))

    difficulties = [str(d) for d in range(1, 9)]
    n_diff = len(difficulties)
    n_cond = len(CONDITIONS)
    bar_width = 0.18
    offsets = np.arange(n_diff)

    for i, cond in enumerate(CONDITIONS):
        h1_list = eval_data[cond].get("h1", [])
        if not h1_list:
            continue
        all_by_diff = defaultdict(list)
        for h1 in h1_list:
            for d, v in h1["by_difficulty"].items():
                all_by_diff[d].append(v)

        means = [np.mean(all_by_diff.get(d, [0])) for d in difficulties]
        stds = [np.std(all_by_diff.get(d, [0])) for d in difficulties]
        pos = offsets + (i - n_cond / 2 + 0.5) * bar_width
        ax.bar(pos, means, bar_width, yerr=stds, label=COND_LABELS[cond],
               color=COND_COLORS[cond], edgecolor="black", linewidth=0.3,
               capsize=2, alpha=0.85)

    ax.set_xticks(offsets)
    ax.set_xticklabels([f"D{d}" for d in range(1, 9)])
    ax.set_xlabel("Task Difficulty (number of steps)")
    ax.set_ylabel("Completion Rate")
    ax.set_title("Transfer Accuracy by Difficulty Level")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "transfer_by_difficulty.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "transfer_by_difficulty.png"))
    plt.close(fig)
    print("  [OK] transfer_by_difficulty")


# ── Figure 4: Training Success Rate & LoRA Updates ───────────────────
def fig_training_success():
    eval_data = load_eval_results()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Completion at checkpoints
    ax = axes[0]
    checkpoints = [40, 80, 120, 160]
    for cond in CONDITIONS:
        h2_list = eval_data[cond].get("h2", [])
        if not h2_list:
            continue
        all_cp = defaultdict(list)
        for h2 in h2_list:
            for cp_str, val in h2["completion_at_checkpoints"].items():
                all_cp[int(cp_str)].append(val)
        means = [np.mean(all_cp[cp]) for cp in checkpoints]
        stds = [np.std(all_cp[cp]) for cp in checkpoints]
        ax.errorbar(checkpoints, means, yerr=stds, label=COND_LABELS[cond],
                    color=COND_COLORS[cond], marker="o", markersize=4,
                    linewidth=1.5, capsize=3)
    ax.set_xlabel("Episode Checkpoint")
    ax.set_ylabel("Cumulative Completion Rate")
    ax.set_title("(a) H2: Training Completion at Checkpoints")
    ax.legend(fontsize=7)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Total LoRA updates
    ax = axes[1]
    for i, cond in enumerate(CONDITIONS):
        h2_list = eval_data[cond].get("h2", [])
        if not h2_list:
            continue
        updates = [h2["total_lora_updates"] for h2 in h2_list]
        ax.bar(i, np.mean(updates), yerr=np.std(updates),
               color=COND_COLORS[cond], edgecolor="black", linewidth=0.5,
               capsize=4, width=0.6, alpha=0.85)
        ax.text(i, np.mean(updates) + 2, f"{np.mean(updates):.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("Total LoRA Updates")
    ax.set_title("(b) Habit Formation (LoRA Updates)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_success.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "training_success.png"))
    plt.close(fig)
    print("  [OK] training_success")


# ── Figure 5: Teaching Efficiency ────────────────────────────────────
def fig_teaching_efficiency():
    eval_data = load_eval_results()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Avg turns
    ax = axes[0]
    for i, cond in enumerate(CONDITIONS):
        eff_list = eval_data[cond].get("efficiency", [])
        if not eff_list:
            continue
        turns = [e["avg_turns"] for e in eff_list]
        ax.bar(i, np.mean(turns), yerr=np.std(turns),
               color=COND_COLORS[cond], edgecolor="black", linewidth=0.5,
               capsize=4, width=0.6, alpha=0.85)
        ax.text(i, np.mean(turns) + 0.1, f"{np.mean(turns):.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("Avg. Turns to Completion")
    ax.set_title("(a) Teaching Efficiency")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Success rate
    ax = axes[1]
    for i, cond in enumerate(CONDITIONS):
        eff_list = eval_data[cond].get("efficiency", [])
        if not eff_list:
            continue
        rates = [e["successful_episodes"] / e["total_episodes"] for e in eff_list]
        ax.bar(i, np.mean(rates), yerr=np.std(rates),
               color=COND_COLORS[cond], edgecolor="black", linewidth=0.5,
               capsize=4, width=0.6, alpha=0.85)
        ax.text(i, np.mean(rates) + 0.01, f"{np.mean(rates)*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_title("(b) Training Success Rate")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "teaching_efficiency.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "teaching_efficiency.png"))
    plt.close(fig)
    print("  [OK] teaching_efficiency")


# ── Figure 6: Category Heatmap ──────────────────────────────────────
def fig_category_heatmap():
    eval_data = load_eval_results()

    categories = sorted(set())
    cond_cat_vals = {}
    for cond in CONDITIONS:
        h1_list = eval_data[cond].get("h1", [])
        if not h1_list:
            continue
        cat_agg = defaultdict(list)
        for h1 in h1_list:
            for cat, val in h1["by_category"].items():
                cat_agg[cat].append(val)
                categories = sorted(set(list(categories) + [cat]))
        cond_cat_vals[cond] = {c: np.mean(cat_agg.get(c, [0])) for c in categories}

    categories = sorted(categories)
    cat_labels = [c.replace("_", " ").title() for c in categories]

    matrix = np.zeros((len(CONDITIONS), len(categories)))
    for i, cond in enumerate(CONDITIONS):
        for j, cat in enumerate(categories):
            matrix[i, j] = cond_cat_vals.get(cond, {}).get(cat, 0)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_title("Transfer Accuracy: Condition × Category")

    for i in range(len(CONDITIONS)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Completion Rate", fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, "category_heatmap.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "category_heatmap.png"))
    plt.close(fig)
    print("  [OK] category_heatmap")


# ── Figure 7: Salience & LTM Evolution ──────────────────────────────
def fig_salience_ltm():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Salience over episodes
    ax = axes[0]
    for cond in CONDITIONS:
        all_salience = []
        max_ep = 0
        for seed in SEEDS:
            metrics = load_metrics(cond, seed)
            if not metrics:
                continue
            sal = [m["salience"] for m in sorted(metrics, key=lambda x: x["episode"])]
            smoothed = rolling_avg(sal, window=10)
            all_salience.append(smoothed)
            max_ep = max(max_ep, len(smoothed))
        if not all_salience:
            continue
        for i in range(len(all_salience)):
            while len(all_salience[i]) < max_ep:
                all_salience[i].append(all_salience[i][-1])
        arr = np.array(all_salience)
        mean = arr.mean(axis=0)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=COND_LABELS[cond], color=COND_COLORS[cond], linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Salience Score (rolling avg)")
    ax.set_title("(a) Salience Signal Over Training")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: LTM size growth
    ax = axes[1]
    for cond in CONDITIONS:
        all_ltm = []
        max_ep = 0
        for seed in SEEDS:
            metrics = load_metrics(cond, seed)
            if not metrics:
                continue
            ltm = [m["ltm_size"] for m in sorted(metrics, key=lambda x: x["episode"])]
            all_ltm.append(ltm)
            max_ep = max(max_ep, len(ltm))
        if not all_ltm:
            continue
        for i in range(len(all_ltm)):
            while len(all_ltm[i]) < max_ep:
                all_ltm[i].append(all_ltm[i][-1])
        arr = np.array(all_ltm)
        mean = arr.mean(axis=0)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=COND_LABELS[cond], color=COND_COLORS[cond], linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("LTM Entries")
    ax.set_title("(b) Long-Term Memory Growth")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "salience_ltm.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "salience_ltm.png"))
    plt.close(fig)
    print("  [OK] salience_ltm")


# ── Figure 8: Difficulty Progression ─────────────────────────────────
def fig_difficulty_progression():
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for cond in CONDITIONS:
        all_diffs = []
        max_ep = 0
        for seed in SEEDS:
            metrics = load_metrics(cond, seed)
            if not metrics:
                continue
            diffs = [m["difficulty"] for m in sorted(metrics, key=lambda x: x["episode"])]
            smoothed = rolling_avg(diffs, window=10)
            all_diffs.append(smoothed)
            max_ep = max(max_ep, len(smoothed))
        if not all_diffs:
            continue
        for i in range(len(all_diffs)):
            while len(all_diffs[i]) < max_ep:
                all_diffs[i].append(all_diffs[i][-1])
        arr = np.array(all_diffs)
        mean = arr.mean(axis=0)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=COND_LABELS[cond], color=COND_COLORS[cond], linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Task Difficulty (rolling avg)")
    ax.set_title("Curriculum Difficulty Progression")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "difficulty_progression.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "difficulty_progression.png"))
    plt.close(fig)
    print("  [OK] difficulty_progression")


# ── Figure 9: Competence Levels Over Time ────────────────────────────
def fig_competence_levels():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    level_order = ["none", "full_support", "guided_discovery", "minimal_support"]
    level_colors = {
        "none": "#bdc3c7",
        "full_support": "#e74c3c",
        "guided_discovery": "#f39c12",
        "minimal_support": "#27ae60",
    }
    level_labels = {
        "none": "None/Solo",
        "full_support": "Full Support",
        "guided_discovery": "Guided Discovery",
        "minimal_support": "Minimal Support",
    }

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        for seed in SEEDS:
            metrics = load_metrics(cond, seed)
            if not metrics:
                continue
            sorted_m = sorted(metrics, key=lambda x: x["episode"])
            episodes = [m["episode"] for m in sorted_m]
            levels = [m.get("competence_level", "none") for m in sorted_m]

            level_map = {l: i for i, l in enumerate(level_order)}
            numeric = [level_map.get(l, 0) for l in levels]
            smoothed = rolling_avg(numeric, window=8)
            ax.plot(episodes, smoothed, alpha=0.5, linewidth=1)

        ax.set_title(COND_LABELS[cond], fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Scaffolding Level")
        ax.set_yticks(range(len(level_order)))
        ax.set_yticklabels([level_labels[l] for l in level_order], fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptive Scaffolding Progression", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "competence_levels.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "competence_levels.png"))
    plt.close(fig)
    print("  [OK] competence_levels")


# ── Summary Statistics Table (printed to console) ────────────────────
def print_summary_table():
    eval_data = load_eval_results()
    print()
    print("  +------------------+----------+----------+----------+---------+")
    print("  | Metric           |  Solo    |  Peer    | Role-Lab | Relat.  |")
    print("  +------------------+----------+----------+----------+---------+")

    for metric_name, key_path in [
        ("H1 Transfer", lambda c: [h["avg_completion_rate"] for h in eval_data[c].get("h1", [])]),
        ("Train Success", lambda c: [e["successful_episodes"]/e["total_episodes"] for e in eval_data[c].get("efficiency", [])]),
        ("Avg Turns", lambda c: [e["avg_turns"] for e in eval_data[c].get("efficiency", [])]),
        ("LoRA Updates", lambda c: [float(h["total_lora_updates"]) for h in eval_data[c].get("h2", [])]),
    ]:
        vals = []
        for cond in CONDITIONS:
            v = key_path(cond)
            if v:
                vals.append(f"{np.mean(v):.3f}")
            else:
                vals.append("  ---  ")
        print(f"  | {metric_name:<16} | {vals[0]:>8} | {vals[1]:>8} | {vals[2]:>8} | {vals[3]:>7} |")

    print("  +------------------+----------+----------+----------+---------+")


if __name__ == "__main__":
    print("Generating figures...")
    fig_learning_curves()
    fig_h1_transfer()
    fig_transfer_by_difficulty()
    fig_training_success()
    fig_teaching_efficiency()
    fig_category_heatmap()
    fig_salience_ltm()
    fig_difficulty_progression()
    fig_competence_levels()
    print_summary_table()
    print(f"\nAll figures saved to {FIG_DIR}")
