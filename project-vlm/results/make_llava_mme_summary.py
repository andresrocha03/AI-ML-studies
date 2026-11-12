"""Render the LLaVA 1.5 7B MME table from saved FastV, DivPrune, and vanilla runs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
RUNS = [
    ("Vanilla\n100% kept", "vanilla-llava/vanilla_2gpu_*/**/mme.json", "log"),
    ("DivPrune\n9.8% kept", "divprune/divprune.csv", "csv"),
    ("FastV K=3\n9.8% kept", "fastv/fastv-R098/fastv-R098.csv", "csv"),
    ("FastV K=3\n11.1% kept", "fastv/fastv_keep64_*/**/mme.json", "log"),
    ("FastV K=3\n1.0% kept", "fastv/fastv_keep6_*/fastv_keep6.csv", "csv"),
    ("FastV K=3\n0% kept", "fastv/fastv_k3_keep0_*/**/mme.json", "log"),
]
COGNITION = ("code_reasoning", "numerical_calculation", "text_translation", "commonsense_reasoning")
PERCEPTION = ("artwork", "celebrity", "count", "color", "position", "OCR", "landmark",
              "scene", "existence", "posters")
METRICS = [
    ("Total", "total"),
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


def read_scores(pattern, kind):
    paths = list(ROOT.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one result for {pattern}; found {len(paths)}")
    with paths[0].open(newline="") as file:
        if kind == "csv":
            scores = {key: float(value) for key, value in next(csv.DictReader(file)).items()}
        else:
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
            scores["cognition"] = sum(scores[key] for key in COGNITION)
            scores["perception"] = sum(scores[key] for key in PERCEPTION)
    scores["total"] = scores["perception"] + scores["cognition"]
    return scores


def main():
    scores_by_run = [read_scores(pattern, kind) for _, pattern, kind in RUNS]
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    navy, ink, muted = "#1e3354", "#172238", "#56647c"
    stripe, rule = "#eff3f9", "#d5dfeb"
    ax.text(0.04, 0.95, "LLaVA 1.5 7B · MME scores", fontsize=23,
            weight="bold", color=ink, va="center")
    ax.text(0.04, 0.91, "Perception, cognition, and all 14 subtasks · highest score in bold",
            fontsize=11.5, color=muted, va="center")

    left, right, top = 0.025, 0.975, 0.83
    header_height, row_height = 0.065, 0.039
    metric_width = 0.205
    score_width = (right - left - metric_width) / len(RUNS)
    edges = [left, left + metric_width] + [left + metric_width + i * score_width
                                           for i in range(1, len(RUNS) + 1)]

    for col in range(len(edges) - 1):
        ax.add_patch(Rectangle((edges[col], top - header_height),
                               edges[col + 1] - edges[col], header_height,
                               facecolor=navy, edgecolor=rule, linewidth=0.8))
    ax.text(left + 0.012, top - header_height / 2, "Metric", color="white",
            fontsize=11, weight="bold", va="center")
    for col, (label, _, _) in enumerate(RUNS, start=1):
        ax.text((edges[col] + edges[col + 1]) / 2, top - header_height / 2,
                label, ha="center", va="center", color="white",
                fontsize=10, weight="bold", linespacing=1.1)

    for row, (label, key) in enumerate(METRICS):
        y_top = top - header_height - row * row_height
        y_mid = y_top - row_height / 2
        values = [scores[key] for scores in scores_by_run]
        for col in range(len(edges) - 1):
            ax.add_patch(Rectangle((edges[col], y_top - row_height),
                                   edges[col + 1] - edges[col], row_height,
                                   facecolor="white" if row % 2 == 0 else stripe,
                                   edgecolor=rule, linewidth=0.8))
        ax.text(left + 0.012, y_mid, label, va="center", fontsize=10.6,
                weight="bold" if row < 3 else "normal")
        for col, value in enumerate(values, start=1):
            ax.text(edges[col + 1] - 0.012, y_mid, f"{value:,.2f}",
                    ha="right", va="center", fontsize=10.2,
                    family="DejaVu Sans Mono",
                    weight="bold" if value == max(values) else "normal")

    ax.text(0.04, 0.055, "Total = perception + cognition. Kept percentages use 576 image tokens.",
            fontsize=10, color=muted, va="center")
    ax.text(0.04, 0.035, "The 64-token run is labeled K=3 as requested; its run.log records K=2.",
            fontsize=10, color=muted, va="center")
    fig.savefig(ROOT / "llava_mme_table.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
