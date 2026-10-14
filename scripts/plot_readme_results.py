"""Session-error comparison for the README figure.

Numbers are the design-grid comparison already printed with the study.
The image has no caption band.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Session error: pointwise, Bonferroni, certificate. Same order as the study.
ROWS = [
    ("Baseline", 0.0750, 0.0200, 0.0150),
    ("20 units", 0.0865, 0.0090, 0.0100),
    ("60 units", 0.0900, 0.0000, 0.0000),
    ("Homogeneous", 0.0000, 0.0000, 0.0000),
    ("Transfer x2", 0.0650, 0.0050, 0.0050),
    ("Weight 1", 0.0745, 0.0070, 0.0075),
    ("Weight 8", 0.0750, 0.0150, 0.0150),
    ("Cell 4/25", 0.1400, 0.0050, 0.0050),
    ("Cell 12/25", 0.0720, 0.0060, 0.0065),
    ("Two projections", 0.0760, 0.0035, 0.0045),
    ("Two cells", 0.0350, 0.0000, 0.0000),
    ("Balanced", 0.0745, 0.0020, 0.0050),
    ("Release 40", 0.1100, 0.0050, 0.0050),
    ("Release 160", 0.0700, 0.0050, 0.0050),
]


def main() -> None:
    labels = [row[0] for row in ROWS]
    pointwise = np.array([row[1] for row in ROWS])
    bonferroni = np.array([row[2] for row in ROWS])
    certificate = np.array([row[3] for row in ROWS])
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.bar(x - width, pointwise, width, label="Pointwise", color="#b7c4ce")
    ax.bar(x, bonferroni, width, label="Bonferroni", color="#7f8c9a")
    ax.bar(x + width, certificate, width, label="Certificate", color="#1f4e79")
    ax.axhline(0.10, color="#8c3a3a", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Session error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 0.16)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    dest = ROOT / "docs" / "figures" / "fig_results.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
