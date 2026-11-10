"""Gera 3 figuras da cadeia do ténis (game): diagrama, matriz canónica e propriedades.
Precisa de tenis_markov.py na mesma pasta."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from tenis_markov import matriz_game, analise_absorvente, game_formula, IDX, NAMES, PTS

CP, CQ = "#2563eb", "#ea580c"            # p (azul) e q (laranja)
C_TR, C_DE, C_GS, C_GR = "#dbeafe", "#fde68a", "#bbf7d0", "#fecaca"


def seta(ax, a, b, cor, rad=0.0, lab=None, off=(0, 0)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=cor, lw=1.4,
                shrinkA=13, shrinkB=13, connectionstyle=f"arc3,rad={rad}"))
    if lab:
        mx, my = (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]
        ax.text(mx, my, lab, color=cor, fontsize=9, fontweight="bold", ha="center", va="center")


def no(ax, xy, txt, cor, fs=10):
    ax.text(*xy, txt, ha="center", va="center", fontsize=fs, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc=cor, ec="#334155", lw=1.1))


def fig_diagrama():
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = {}
    for a in range(4):
        for b in range(4):
            if (a, b) != (3, 3):
                pos[(a, b)] = (b, -a)
    pos["D"] = (3, -3); pos["VS"] = (3, -4.2); pos["VR"] = (4.2, -3)
    yS, xR = -5.3, 5.2

    # caixas absorventes
    ax.add_patch(FancyBboxPatch((-0.5, -5.9), 4.1, 0.6, boxstyle="round,pad=0.02", fc=C_GS, ec="#166534", lw=2))
    ax.text(1.55, -5.6, "GAME SERVIDOR   (estado absorvente)", ha="center", va="center", fontweight="bold", color="#166534")
    ax.add_patch(FancyBboxPatch((5.1, -3.6), 0.7, 4.1, boxstyle="round,pad=0.02", fc=C_GR, ec="#991b1b", lw=2))
    ax.text(5.45, -1.55, "GAME RECEÇÃO   (estado absorvente)", rotation=90, ha="center", va="center", fontweight="bold", color="#991b1b")

    for (a, b), xy in [(k, v) for k, v in pos.items() if isinstance(k, tuple)]:
        no(ax, xy, f"{PTS[a]}-{PTS[b]}", C_TR)
    no(ax, pos["D"], "Iguais\n40-40", C_DE)
    no(ax, pos["VS"], "Vant.\nServidor", C_DE, 9)
    no(ax, pos["VR"], "Vant.\nReceção", C_DE, 9)

    for a in range(4):
        for b in range(4):
            if (a, b) == (3, 3):
                continue
            o = pos[(a, b)]
            # servidor ganha o ponto (p): desce
            if a == 3:
                seta(ax, o, (b, yS + 0.02), CP, lab="p", off=(0.14, 0))
            else:
                d = pos["D"] if (a + 1, b) == (3, 3) else pos[(a + 1, b)]
                seta(ax, o, d, CP, lab="p", off=(0.14, 0))
            # receção ganha o ponto (q): vai para a direita
            if b == 3:
                seta(ax, o, (xR - 0.1, -a), CQ, lab="q", off=(0, 0.14))
            else:
                d = pos["D"] if (a, b + 1) == (3, 3) else pos[(a, b + 1)]
                seta(ax, o, d, CQ, lab="q", off=(0, 0.14))

    # deuce e vantagens
    seta(ax, pos["D"], pos["VS"], CP, rad=0.25, lab="p", off=(-0.38, 0))
    seta(ax, pos["VS"], pos["D"], CQ, rad=0.25, lab="q", off=(0.38, 0))
    seta(ax, pos["D"], pos["VR"], CQ, rad=0.25, lab="q", off=(0, -0.38))
    seta(ax, pos["VR"], pos["D"], CP, rad=0.25, lab="p", off=(0, 0.38))
    seta(ax, pos["VS"], (3, yS + 0.02), CP, lab="p", off=(0.14, 0))
    seta(ax, pos["VR"], (xR - 0.1, -3), CQ, lab="q", off=(0, 0.14))

    ax.text(-0.75, 0.75, "Cada estado = score (pontos do servidor – pontos do recetor)", fontsize=11, fontweight="bold")
    ax.text(-0.75, 0.45, "→ azul: o servidor ganha o ponto (prob. p)       → laranja: o recetor ganha o ponto (prob. q = 1 − p)",
            fontsize=10, color="#334155")
    ax.text(-0.75, -6.25, "18 estados transientes (azul/amarelo)  +  2 estados absorventes (verde/vermelho) = 20 estados",
            fontsize=10, color="#334155")
    ax.set_xlim(-0.85, 6.0); ax.set_ylim(-6.5, 1.1); ax.axis("off")
    plt.tight_layout(); plt.savefig("diagrama_cadeia.png", dpi=160); plt.close()


def fig_matriz(p=0.64):
    P = matriz_game(p)
    short = [n.replace("Iguais(40-40)", "40-40").replace("Vant.Servidor", "Vant.S").replace("Vant.Rececao", "Vant.R")
              .replace("Game Servidor", "Game S").replace("Game Rececao", "Game R") for n in NAMES]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.imshow(np.where(P > 0, P, np.nan), cmap="Blues", vmin=0, vmax=1.4)
    for i in range(20):
        for j in range(20):
            if P[i, j] > 0:
                t = "1" if P[i, j] == 1 else ("p" if abs(P[i, j] - p) < 1e-12 else "q")
                ax.text(j, i, t, ha="center", va="center", fontsize=8,
                        color={"p": CP, "q": CQ, "1": "#111"}[t], fontweight="bold")
    ax.axhline(17.5, c="k", lw=2); ax.axvline(17.5, c="k", lw=2)
    ax.text(5, 13, "Q\n(transientes → transientes)", ha="center", va="center", fontsize=12, alpha=.5, fontweight="bold")
    ax.text(8.5, 18.5, "0", ha="center", va="center", fontsize=13, alpha=.5, fontweight="bold")
    ax.text(16.2, 6, "R", ha="center", va="center", fontsize=13, alpha=.5, fontweight="bold")
    ax.text(18.0, 20.2, "I  (2×2 identidade)", ha="center", va="center", fontsize=10, alpha=.6, fontweight="bold")
    ax.set_xticks(range(20)); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(20)); ax.set_yticklabels(short, fontsize=8)
    ax.xaxis.tick_top(); ax.set_xlabel("para"); ax.set_ylabel("de")
    ax.set_title("Matriz de transição P na forma canónica   P = [ Q  R ; 0  I ]    (p = 0,64; q = 0,36)", pad=45)
    plt.tight_layout(); plt.savefig("matriz_canonica.png", dpi=160); plt.close()


def grelha(ax, vals, titulo, fmt, cmap):
    M = np.full((4, 4), np.nan)
    for (a, b), v in vals.items():
        M[a, b] = v
    ax.imshow(M, cmap=cmap)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for a in range(4):
        for b in range(4):
            v = M[a, b]
            lab = f"{PTS[a]}-{PTS[b]}" if (a, b) != (3, 3) else "Iguais"
            escuro = (v - lo) / (hi - lo) > 0.55
            ax.text(b, a, f"{lab}\n{fmt.format(v)}", ha="center", va="center", fontsize=9.5,
                    color="white" if escuro else "black", fontweight="bold" if escuro else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(PTS); ax.set_yticks(range(4)); ax.set_yticklabels(PTS)
    ax.set_xlabel("pontos do recetor"); ax.set_ylabel("pontos do servidor"); ax.set_title(titulo, fontsize=11)


def fig_propriedades(p=0.64):
    P = matriz_game(p)
    N, B, t, var = analise_absorvente(P)
    chaves = {}
    for a in range(4):
        for b in range(4):
            n = "Iguais(40-40)" if (a, b) == (3, 3) else f"{PTS[a]}-{PTS[b]}"
            chaves[(a, b)] = IDX[n]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    grelha(ax[0, 0], {k: B[i, 0] for k, i in chaves.items()}, f"P(servidor ganha o game | score)   [p = {p}]", "{:.2f}", "Greens")
    grelha(ax[0, 1], {k: t[i] for k, i in chaves.items()}, "Nº esperado de pontos até ao fim (matriz fundamental)", "{:.1f}", "Oranges")

    ps = np.linspace(0, 1, 201)
    ax[1, 0].plot(ps, ps, "--", c="gray", label="y = p")
    ax[1, 0].plot(ps, game_formula(ps), c="#16a34a", lw=2.2, label="P(ganhar o game)")
    for pp in (0.55, 0.60, 0.64, 0.70):
        ax[1, 0].plot(pp, game_formula(pp), "o", c="k"); ax[1, 0].annotate(f"{game_formula(pp):.2f}", (pp, game_formula(pp)), xytext=(5, -14), textcoords="offset points", fontsize=9)
    ax[1, 0].set_xlabel("p = P(ganhar um ponto)"); ax[1, 0].set_title("Amplificação: ponto → game"); ax[1, 0].legend()

    v = np.zeros(20); v[IDX["0-0"]] = 1; fim = [0]
    for n in range(1, 21):
        v = v @ P; fim.append(v[18] + v[19])
    ax[1, 1].bar(range(1, 21), np.diff(fim), color="#2563eb")
    ax[1, 1].set_xticks(range(4, 21, 2)); ax[1, 1].set_xlabel("n (nº de pontos)")
    ax[1, 1].set_title("P(game acaba exatamente ao ponto n),  início 0-0\n(só pode acabar em n = 4, 5, 6, 8, 10, 12, …)", fontsize=10)
    plt.tight_layout(); plt.savefig("propriedades.png", dpi=160); plt.close()


if __name__ == "__main__":
    fig_diagrama(); fig_matriz(); fig_propriedades()
    print("Geradas: diagrama_cadeia.png, matriz_canonica.png, propriedades.png")
