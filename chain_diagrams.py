"""Draws three figures of the tennis game chain: transition graph, canonical matrix and properties.
Needs tennis_markov.py in the same folder. Figures are saved to figures/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tennis_markov import game_matrix, absorbing_analysis, game_formula, IDX, NAMES, PTS, FIGURES

CP, CQ = "#2563eb", "#ea580c"            # p (blue) and q (orange)
C_TR, C_DE, C_GS, C_GR = "#dbeafe", "#fde68a", "#bbf7d0", "#fecaca"


def arrow(ax, a, b, color, rad=0.0, lab=None, off=(0, 0)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                shrinkA=13, shrinkB=13, connectionstyle=f"arc3,rad={rad}"))
    if lab:
        mx, my = (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]
        ax.text(mx, my, lab, color=color, fontsize=9, fontweight="bold", ha="center", va="center")


def node(ax, xy, txt, color, fs=10):
    ax.text(*xy, txt, ha="center", va="center", fontsize=fs, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="#334155", lw=1.1))


def fig_diagram():
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = {}
    for a in range(4):
        for b in range(4):
            if (a, b) != (3, 3):
                pos[(a, b)] = (b, -a)
    pos["D"] = (3, -3); pos["AS"] = (3, -4.2); pos["AR"] = (4.2, -3)
    yS, xR = -5.3, 5.2

    # absorbing boxes
    ax.add_patch(FancyBboxPatch((-0.5, -5.9), 4.1, 0.6, boxstyle="round,pad=0.02", fc=C_GS, ec="#166534", lw=2))
    ax.text(1.55, -5.6, "GAME SERVER   (absorbing state)", ha="center", va="center", fontweight="bold", color="#166534")
    ax.add_patch(FancyBboxPatch((5.1, -3.6), 0.7, 4.1, boxstyle="round,pad=0.02", fc=C_GR, ec="#991b1b", lw=2))
    ax.text(5.45, -1.55, "GAME RECEIVER   (absorbing state)", rotation=90, ha="center", va="center", fontweight="bold", color="#991b1b")

    for (a, b), xy in [(k, v) for k, v in pos.items() if isinstance(k, tuple)]:
        node(ax, xy, f"{PTS[a]}-{PTS[b]}", C_TR)
    node(ax, pos["D"], "Deuce\n40-40", C_DE)
    node(ax, pos["AS"], "Adv.\nserver", C_DE, 9)
    node(ax, pos["AR"], "Adv.\nreceiver", C_DE, 9)

    for a in range(4):
        for b in range(4):
            if (a, b) == (3, 3):
                continue
            o = pos[(a, b)]
            # server wins the point (p): goes down
            if a == 3:
                arrow(ax, o, (b, yS + 0.02), CP, lab="p", off=(0.14, 0))
            else:
                d = pos["D"] if (a + 1, b) == (3, 3) else pos[(a + 1, b)]
                arrow(ax, o, d, CP, lab="p", off=(0.14, 0))
            # receiver wins the point (q): goes right
            if b == 3:
                arrow(ax, o, (xR - 0.1, -a), CQ, lab="q", off=(0, 0.14))
            else:
                d = pos["D"] if (a, b + 1) == (3, 3) else pos[(a, b + 1)]
                arrow(ax, o, d, CQ, lab="q", off=(0, 0.14))

    # deuce and advantages
    arrow(ax, pos["D"], pos["AS"], CP, rad=0.25, lab="p", off=(-0.38, 0))
    arrow(ax, pos["AS"], pos["D"], CQ, rad=0.25, lab="q", off=(0.38, 0))
    arrow(ax, pos["D"], pos["AR"], CQ, rad=0.25, lab="q", off=(0, -0.38))
    arrow(ax, pos["AR"], pos["D"], CP, rad=0.25, lab="p", off=(0, 0.38))
    arrow(ax, pos["AS"], (3, yS + 0.02), CP, lab="p", off=(0.14, 0))
    arrow(ax, pos["AR"], (xR - 0.1, -3), CQ, lab="q", off=(0, 0.14))

    ax.text(-0.75, 0.75, "Each state = score (server points – receiver points)", fontsize=11, fontweight="bold")
    ax.text(-0.75, 0.45, "→ blue: the server wins the point (prob. p)       → orange: the receiver wins the point (prob. q = 1 − p)",
            fontsize=10, color="#334155")
    ax.text(-0.75, -6.25, "18 transient states (blue/yellow)  +  2 absorbing states (green/red) = 20 states",
            fontsize=10, color="#334155")
    ax.set_xlim(-0.85, 6.0); ax.set_ylim(-6.5, 1.1); ax.axis("off")
    plt.tight_layout(); plt.savefig(FIGURES / "chain_diagram.png", dpi=160); plt.close()


def fig_matrix(p=0.64):
    P = game_matrix(p)
    short = [n.replace("Adv. server", "Adv.S").replace("Adv. receiver", "Adv.R")
              .replace("Game server", "Game S").replace("Game receiver", "Game R") for n in NAMES]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.imshow(np.where(P > 0, P, np.nan), cmap="Blues", vmin=0, vmax=1.4)
    for i in range(20):
        for j in range(20):
            if P[i, j] > 0:
                t = "1" if P[i, j] == 1 else ("p" if abs(P[i, j] - p) < 1e-12 else "q")
                ax.text(j, i, t, ha="center", va="center", fontsize=8,
                        color={"p": CP, "q": CQ, "1": "#111"}[t], fontweight="bold")
    ax.axhline(17.5, c="k", lw=2); ax.axvline(17.5, c="k", lw=2)
    ax.text(5, 13, "Q\n(transient → transient)", ha="center", va="center", fontsize=12, alpha=.5, fontweight="bold")
    ax.text(8.5, 18.5, "0", ha="center", va="center", fontsize=13, alpha=.5, fontweight="bold")
    ax.text(16.2, 6, "R", ha="center", va="center", fontsize=13, alpha=.5, fontweight="bold")
    ax.text(18.0, 20.2, "I  (2×2 identity)", ha="center", va="center", fontsize=10, alpha=.6, fontweight="bold")
    ax.set_xticks(range(20)); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(20)); ax.set_yticklabels(short, fontsize=8)
    ax.xaxis.tick_top(); ax.set_xlabel("to"); ax.set_ylabel("from")
    ax.set_title("Transition matrix P in canonical form   P = [ Q  R ; 0  I ]    (p = 0.64; q = 0.36)", pad=45)
    plt.tight_layout(); plt.savefig(FIGURES / "canonical_matrix.png", dpi=160); plt.close()


def grid(ax, vals, title, fmt, cmap):
    M = np.full((4, 4), np.nan)
    for (a, b), v in vals.items():
        M[a, b] = v
    ax.imshow(M, cmap=cmap)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for a in range(4):
        for b in range(4):
            v = M[a, b]
            lab = f"{PTS[a]}-{PTS[b]}" if (a, b) != (3, 3) else "Deuce"
            dark = (v - lo) / (hi - lo) > 0.55
            ax.text(b, a, f"{lab}\n{fmt.format(v)}", ha="center", va="center", fontsize=9.5,
                    color="white" if dark else "black", fontweight="bold" if dark else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(PTS); ax.set_yticks(range(4)); ax.set_yticklabels(PTS)
    ax.set_xlabel("receiver points"); ax.set_ylabel("server points"); ax.set_title(title, fontsize=11)


def fig_properties(p=0.64):
    P = game_matrix(p)
    N, B, t, var = absorbing_analysis(P)
    keys = {}
    for a in range(4):
        for b in range(4):
            n = "Deuce" if (a, b) == (3, 3) else f"{PTS[a]}-{PTS[b]}"
            keys[(a, b)] = IDX[n]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    grid(ax[0, 0], {k: B[i, 0] for k, i in keys.items()}, f"P(server wins the game | score)   [p = {p}]", "{:.2f}", "Greens")
    grid(ax[0, 1], {k: t[i] for k, i in keys.items()}, "Expected number of points until the end (fundamental matrix)", "{:.1f}", "Oranges")

    ps = np.linspace(0, 1, 201)
    ax[1, 0].plot(ps, ps, "--", c="gray", label="y = p")
    ax[1, 0].plot(ps, game_formula(ps), c="#16a34a", lw=2.2, label="P(win the game)")
    for pp in (0.55, 0.60, 0.64, 0.70):
        ax[1, 0].plot(pp, game_formula(pp), "o", c="k"); ax[1, 0].annotate(f"{game_formula(pp):.2f}", (pp, game_formula(pp)), xytext=(5, -14), textcoords="offset points", fontsize=9)
    ax[1, 0].set_xlabel("p = P(win a point)"); ax[1, 0].set_title("Amplification: point → game"); ax[1, 0].legend()

    v = np.zeros(20); v[IDX["0-0"]] = 1; end = [0]
    for n in range(1, 21):
        v = v @ P; end.append(v[18] + v[19])
    ax[1, 1].bar(range(1, 21), np.diff(end), color="#2563eb")
    ax[1, 1].set_xticks(range(4, 21, 2)); ax[1, 1].set_xlabel("n (number of points)")
    ax[1, 1].set_title("P(game ends exactly at point n), starting at 0-0\n(it can only end at n = 4, 5, 6, 8, 10, 12, …)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGURES / "chain_properties.png", dpi=160); plt.close()


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    fig_diagram(); fig_matrix(); fig_properties()
    print("Saved to figures/: chain_diagram.png, canonical_matrix.png, chain_properties.png")
