"""Build a compact table from the profiling run summaries."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = [
    ("Qwen 3.5 2B", "Qwen3.5-2B_*"),
    ("Qwen 3.5 4B", "Qwen3.5-4B_*"),
    ("LLaVA 1.5 7B", "llava-v1.5-7b_*"),
    ("LLaVA 1.5 13B", "llava-v1.5-13b_*"),
]
METRICS = [
    ("Mean latency (ms)", "mean_latency_ms"),
    ("Median latency (ms)", "median_latency_ms"),
    ("P95 latency (ms)", "p95_latency_ms"),
    ("Mean prefill (ms)", "mean_prefill_latency_ms"),
    ("Memory after load (MiB)", "memory_after_load_mb"),
    ("Peak allocated (MiB)", "max_peak_allocated_mb"),
    ("Peak reserved (MiB)", "max_peak_reserved_mb"),
    ("Extra inference memory (MiB)", "max_additional_inference_memory_mb"),
]


def read_summary(run_pattern):
    matches = list(ROOT.glob(f"{run_pattern}/summary.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one summary for {run_pattern}; found {len(matches)}")
    with matches[0].open(newline="") as file:
        return next(csv.DictReader(file))


def main():
    summaries = [read_summary(pattern) for _, pattern in RUNS]
    sample_counts = {int(row["num_samples"]) for row in summaries}
    tasks = {row["task"].upper() for row in summaries}
    if len(sample_counts) != 1 or len(tasks) != 1:
        raise ValueError("All runs must have the same task and sample count")

    cells = [
        [label, *(f"{float(row[key]):,.1f}" for row in summaries)]
        for label, key in METRICS
    ]
    fig, ax = plt.subplots(figsize=(12.5, 4.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.suptitle("Profiling summary", x=0.04, y=0.96, ha="left",
                 fontsize=18, fontweight="bold", color="#172238")
    table = ax.table(
        cellText=cells,
        colLabels=["Metric", *(name for name, _ in RUNS)],
        cellLoc="right",
        colLoc="center",
        colWidths=[0.32, 0.17, 0.17, 0.17, 0.17],
        bbox=[0.04, 0.13, 0.92, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dfeb")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#1e3354" if row == 0 else
                           ("white" if row % 2 else "#eff3f9"))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        if col == 0:
            cell.get_text().set_ha("left")

    fig.text(0.04, 0.07,
             f"{tasks.pop()} · {sample_counts.pop()} measured examples per model · "
             "Tesla P100 · 13B uses 4-bit quantization",
             fontsize=9, color="#56647c")
    fig.savefig(ROOT / "profiling.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
