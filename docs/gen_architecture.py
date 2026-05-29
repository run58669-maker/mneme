"""Generate the Mneme architecture diagram (docs/architecture.png).

Run:  python docs/gen_architecture.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")

# palette
C_USER = "#e8eef7"
C_AGENT = "#dCeBfa"
C_QWEN = "#ffe9c7"      # highlight Qwen integration
C_ENGINE = "#e3f3e6"
C_DB = "#f2f2f2"
EDGE = "#5b6b7b"


def box(ax, x, y, w, h, text, fc, fs=11, bold=False, ec="#9aa7b5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#21303f")


def arrow(ax, p1, p2, text="", off=0.0, color=EDGE, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 linewidth=1.5, color=color,
                                 connectionstyle="arc3,rad=0"))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + off, text, ha="center", va="center", fontsize=8.5,
                color="#3a4a5a", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


fig, ax = plt.subplots(figsize=(11, 6.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(6, 8.5, "Mneme  ·  MemoryAgent architecture", ha="center", fontsize=16,
        fontweight="bold", color="#1b2a38")
ax.text(6, 8.0, "brain-inspired associative memory  +  Qwen Cloud reasoning", ha="center",
        fontsize=10.5, color="#5b6b7b", style="italic")

# boxes
box(ax, 0.6, 5.6, 2.2, 1.1, "User", C_USER, bold=True)
box(ax, 4.4, 5.4, 3.2, 1.5, "MemoryAgent\n(orchestration loop)", C_AGENT, fs=11.5, bold=True)
box(ax, 9.0, 5.5, 2.6, 1.3, "Qwen Cloud\n(Model Studio)", C_QWEN, fs=11, bold=True, ec="#e0a23a")
box(ax, 9.0, 3.9, 2.6, 1.0, "• chat  → reply\n• embeddings → recall", C_QWEN, fs=9)
box(ax, 4.0, 2.3, 4.0, 1.5, "Mneme engine\ngraph memory + dynamics", C_ENGINE, fs=11.5, bold=True)
box(ax, 3.4, 0.5, 5.2, 1.1,
    "SQLite graph:  nodes —weighted edges→ nodes\nHebbian decay · spreading activation · sleep consolidation",
    C_DB, fs=8.6)

# arrows
arrow(ax, (2.8, 6.15), (4.4, 6.15), "message")
arrow(ax, (4.4, 5.9), (2.8, 5.9), "reply", off=-0.35)
arrow(ax, (7.6, 6.3), (9.0, 6.15), "generate")
arrow(ax, (7.6, 5.7), (9.0, 5.0), "embed", off=-0.3)
arrow(ax, (6.0, 5.4), (6.0, 3.8), "recall / remember / sleep")
arrow(ax, (6.0, 2.3), (6.0, 1.6), "")

# step legend
steps = ("loop:  1) recall relevant memories (spreading activation)   "
         "2) build context-budgeted prompt   3) Qwen reply   4) remember   ·   sleep(): distil → gist → fade")
ax.text(6, 0.12, steps, ha="center", fontsize=8.2, color="#5b6b7b")

plt.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
