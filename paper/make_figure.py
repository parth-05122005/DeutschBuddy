"""Draw Fig. 1 (system architecture) for the IEEE paper, sized for one column (3.5 in).

Run:  python paper/make_figure.py   →  paper/architecture.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture.png"
W, H = 4.1, 1.25  # box size in data units

# name, sub-line, centre x, centre y, highlighted
BOXES = {
    "docs": ("Documents", "Markdown, PDF, TXT", 2.4, 11.0, False),
    "chunk": ("Chunker", "split by heading", 2.4, 9.1, False),
    "embed": ("Gemini embedder", "RETRIEVAL_DOCUMENT", 2.4, 7.2, False),
    "question": ("Question", "English or German", 7.6, 11.0, False),
    "qembed": ("Embed query", "RETRIEVAL_QUERY", 7.6, 9.1, False),
    "chroma": ("ChromaDB", "cosine HNSW index", 5.0, 5.2, True),
    "prompt": ("Prompt builder", "rules, level, passages", 2.4, 3.1, False),
    "llm": ("Gemini Flash-Lite", "streams the answer", 7.6, 3.1, False),
    "answer": ("Answer + sources", "shown to the learner", 7.6, 1.1, False),
}
ARROWS = [("docs", "chunk"), ("chunk", "embed"), ("embed", "chroma"), ("question", "qembed"),
          ("qembed", "chroma"), ("chroma", "prompt"), ("prompt", "llm"), ("llm", "answer")]
LABELS = {("embed", "chroma"): "chunk vectors", ("qembed", "chroma"): "query vector",
          ("chroma", "prompt"): "top-k passages"}


def edge_point(box, toward):
    """Point on the border of `box` in the direction of `toward` (both are box keys)."""
    _, _, x, y, _ = BOXES[box]
    _, _, tx, ty, _ = BOXES[toward]
    dx, dy = tx - x, ty - y
    scale = min((W / 2) / abs(dx) if dx else float("inf"), (H / 2) / abs(dy) if dy else float("inf"))
    return x + dx * scale, y + dy * scale


def main():
    fig, ax = plt.subplots(figsize=(3.5, 4.3), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 12.6)
    ax.axis("off")
    plt.rcParams["font.family"] = "Times New Roman"

    ax.text(2.4, 12.15, "INDEXING (once per document)", ha="center", fontsize=6.2, weight="bold", color="#444444")
    ax.text(7.6, 12.15, "ANSWERING (every question)", ha="center", fontsize=6.2, weight="bold", color="#444444")

    for name, sub, x, y, highlight in BOXES.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.25",
                                    linewidth=1.1 if highlight else 0.6, edgecolor="#000000",
                                    facecolor="#E3E3E3" if highlight else "#FFFFFF"))
        ax.text(x, y + 0.2, name, ha="center", va="center", fontsize=6.8, weight="bold")
        ax.text(x, y - 0.3, sub, ha="center", va="center", fontsize=5.6, color="#333333")

    for a, b in ARROWS:
        start, end = edge_point(a, b), edge_point(b, a)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, linewidth=0.6,
                                     color="#000000", shrinkA=1, shrinkB=1))
        if (a, b) in LABELS:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            right = mx > 5 or (a, b) == ("chroma", "prompt")
            ax.text(mx + (0.35 if right else -0.35), my, LABELS[(a, b)], fontsize=5.4, style="italic",
                    ha="left" if right else "right", va="center", color="#333333")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
