"""
Generate a bar chart of skill compression vs the terse control arm.

Reads evals/snapshots/results.json and writes evals/snapshots/results.png.

Run: uv run --with tiktoken --with matplotlib python evals/plot.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken

ENCODING = tiktoken.get_encoding("o200k_base")
SNAPSHOT = Path(__file__).parent / "snapshots" / "results.json"
OUTPUT = Path(__file__).parent / "snapshots" / "results.png"


def count(text: str) -> int:
    return len(ENCODING.encode(text))


def main() -> None:
    data = json.loads(SNAPSHOT.read_text())
    arms = data["arms"]
    meta = data.get("metadata", {})

    terse_tokens = [count(o) for o in arms["__terse__"]]

    rows = []
    for skill, outputs in arms.items():
        if skill in ("__baseline__", "__terse__"):
            continue
        skill_tokens = [count(o) for o in outputs]
        savings = [
            1 - (s / t) if t else 0.0 for s, t in zip(skill_tokens, terse_tokens)
        ]
        rows.append(
            (
                skill,
                statistics.median(savings),
                statistics.mean(savings),
                min(savings),
                max(savings),
            )
        )

    rows.sort(key=lambda r: -r[2])
    names = [r[0] for r in rows]
    medians = [r[1] * 100 for r in rows]
    means = [r[2] * 100 for r in rows]
    mins = [r[3] * 100 for r in rows]
    maxs = [r[4] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], medians, width, label="median", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], means, width, label="mean", color="#f58518")

    # min/max range as error bars on the mean
    err_low = [m - lo for m, lo in zip(means, mins)]
    err_high = [hi - m for m, hi in zip(means, maxs)]
    ax.errorbar(
        [i + width / 2 for i in x],
        means,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1,
        label="min / max",
    )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Output savings vs terse control (%)")
    ax.set_title(
        f"caveman skill compression — {meta.get('model', '?')}, n={meta.get('n_prompts', '?')}"
    )
    ax.invert_yaxis()  # negative = compression, show downward
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
