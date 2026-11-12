"""Render the MME comparison table from the results stored in this project."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent


def read_scores(path):
    with path.open(newline="") as file:
        return {key: float(value) for key, value in next(csv.DictReader(file)).items()}


def read_scores_from_logs(pattern):
    """Calculate the MME category scores using its pairwise scoring rule."""
    matches = list(DATA_ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one MME log for {pattern}, found {len(matches)}")
    with matches[0].open() as file:
        logs = json.load(file)["logs"]

    pairs = defaultdict(lambda: defaultdict(list))
    for entry in logs:
        result = entry.get("mme_percetion_score") or entry.get("mme_cognition_score")
        pairs[result["category"]][result["question_id"]].append(result["score"])

    scores = {}
    for category, questions in pairs.items():
        if any(len(pair) != 2 for pair in questions.values()):
            raise ValueError(f"Incomplete MME question pair in {category}")
        scores[category] = sum(
            (sum(pair) / 2 + (sum(pair) == 2)) * 100 for pair in questions.values()
        ) / len(questions)

    scores["cognition"] = sum(scores[key] for key in (
        "code_reasoning", "numerical_calculation", "text_translation", "commonsense_reasoning"
    ))
    scores["perception"] = sum(scores[key] for key in (
        "artwork", "celebrity", "count", "color", "position", "OCR", "landmark",
        "scene", "existence", "posters"
    ))
    return scores


models = [
    (
        "LLaVA 1.5 7B\nvanilla (100% tokens)",
        read_scores_from_logs("vanilla-llava/vanilla_2gpu_*/**/mme.json"),
    ),
    (
        "LLaVA 1.5 7B\nDivPrune (R=9.8% kept)",
        read_scores(DATA_ROOT / "divprune/divprune.csv"),
    ),
    (
        "LLaVA 1.5 7B\nFastV (K=3, R=9.8% kept)",
        read_scores(DATA_ROOT / "fastv/fastv-R098/fastv-R098.csv"),
    ),
    (
        "LLaVA 1.5 7B\nFastV (K=3, R=0.01% kept)",
        read_scores_from_logs("fastv/fastv_k3_keep0_2gpu_*/**/mme.json"),
    ),
]

metrics = [
    ("Perception", "perception"),
    ("Cognition", "cognition"),
    ("Code reasoning", "code_reasoning"),
    ("Numerical calculation", "numerical_calculation"),
    ("Text translation", "text_translation"),
    ("Commonsense reasoning", "commonsense_reasoning"),
    ("Artwork", "artwork"),
    ("Celebrity", "celebrity"),
    ("Count", "count"),
    ("Color", "color"),
    ("Position", "position"),
    ("OCR", "OCR"),
    ("Landmark", "landmark"),
    ("Scene", "scene"),
    ("Existence", "existence"),
    ("Posters", "posters"),
]


def render():
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    navy = "#1e3354"
    ink = "#172238"
    muted = "#56647c"
    stripe = "#eff3f9"
    rule = "#d5dfeb"

    ax.text(0.055, 0.95, "MME benchmark", fontsize=24, weight="bold", color=ink, va="center")
    ax.text(
        0.055, 0.905,
        "Perception, cognition, and 14 subtask scores · highest score in bold",
        fontsize=12, color=muted, va="center",
    )

    left, right, top = 0.025, 0.975, 0.78
    header_height, row_height = 0.05, 0.035
    metric_width = 0.19
    score_width = (right - left - metric_width) / len(models)
    edges = [left, left + metric_width] + [left + metric_width + i * score_width for i in range(1, len(models) + 1)]

    for column in range(len(edges) - 1):
        ax.add_patch(Rectangle((edges[column], top - header_height), edges[column + 1] - edges[column], header_height,
                               facecolor=navy, edgecolor=rule, linewidth=0.8))
    ax.text(left + 0.02, top - header_height / 2, "Metric", color="white", fontsize=12, weight="bold", va="center")
    for column, (label, _) in enumerate(models, start=1):
        ax.text((edges[column] + edges[column + 1]) / 2, top - header_height / 2,
                label, ha="center", va="center", color="white", fontsize=10.5, weight="bold", linespacing=1.1)

    for row, (label, key) in enumerate(metrics):
        y_top = top - header_height - row * row_height
        y_mid = y_top - row_height / 2
        values = [scores[key] for _, scores in models]
        for column in range(len(edges) - 1):
            ax.add_patch(Rectangle((edges[column], y_top - row_height), edges[column + 1] - edges[column],
                                   row_height, facecolor="white" if row % 2 == 0 else stripe,
                                   edgecolor=rule, linewidth=0.8))
        ax.text(left + 0.02, y_mid, label, va="center", fontsize=11.3,
                weight="bold" if row < 2 else "normal")
        for column, value in enumerate(values, start=1):
            ax.text(edges[column + 1] - 0.025, y_mid, f"{value:.4f}", ha="right", va="center",
                    fontsize=11.3, family="DejaVu Sans Mono",
                    weight="bold" if value == max(values) else "normal")

    ax.text(0.055, 0.075,
            "Cognition = code reasoning + numerical calculation + text translation + commonsense reasoning.",
            fontsize=10.8, color=muted, va="center")
    ax.text(0.055, 0.052,
            "Perception = artwork + celebrity + count + color + position + OCR + landmark + scene + existence + posters.",
            fontsize=10.8, color=muted, va="center")
    fig.savefig(ROOT / "benchmark.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
