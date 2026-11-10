"""
Ténis como Cadeia de Markov absorvente - Trabalho de Grupo 2

PARTE A  Cadeia do "game" (20 estados, 2 absorventes), p = P(servidor ganha um ponto)
PARTE B  Extensão: set (jogos alternando serviço + tie-break) e melhor de 3 sets
PARTE C  (opcional) Validação com dados reais ATP (Jeff Sackmann, tennis_atp no GitHub):
         python tenis_markov.py --atp atp_matches_2023.csv

Uso:  pip install numpy pandas matplotlib   &&   python tenis_markov.py
"""
import argparse, itertools
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=150)

# ----------------------------------------------------------------- PARTE A
PTS = ["0", "15", "30", "40"]
TRANS = [f"{PTS[a]}-{PTS[b]}" for a in range(4) for b in range(4) if (a, b) != (3, 3)]  # 15 estados
TRANS += ["Iguais(40-40)", "Vant.Servidor", "Vant.Rececao"]                                # +3 = 18
ABS = ["Game Servidor", "Game Rececao"]
NAMES = TRANS + ABS            # 20 estados; os 18 primeiros transientes, os 2 últimos absorventes
IDX = {n: i for i, n in enumerate(NAMES)}


def matriz_game(p):
    """Matriz de transição 20x20. p = P(servidor ganha o ponto), q = 1-p."""
    q = 1 - p
    P = np.zeros((20, 20))
    for a in range(4):
        for b in range(4):
            if (a, b) == (3, 3):
                continue
            s = IDX[f"{PTS[a]}-{PTS[b]}"]
            win = IDX["Game Servidor"] if a == 3 else (IDX["Iguais(40-40)"] if (a + 1, b) == (3, 3) else IDX[f"{PTS[a+1]}-{PTS[b]}"])
            lose = IDX["Game Rececao"] if b == 3 else (IDX["Iguais(40-40)"] if (a, b + 1) == (3, 3) else IDX[f"{PTS[a]}-{PTS[b+1]}"])
            P[s, win] += p; P[s, lose] += q
    d, vs, vr = IDX["Iguais(40-40)"], IDX["Vant.Servidor"], IDX["Vant.Rececao"]
    P[d, vs] = p;  P[d, vr] = q
    P[vs, IDX["Game Servidor"]] = p;  P[vs, d] = q
    P[vr, IDX["Game Rececao"]] = q;   P[vr, d] = p
    P[18, 18] = 1; P[19, 19] = 1
    return P


def game_formula(p):
    """P(servidor ganha game) em forma fechada."""
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q**2) + 20 * p**3 * q**3 * p**2 / (p**2 + q**2)


def analise_absorvente(P):
    Q, R = P[:18, :18], P[:18, 18:]
    N = np.linalg.inv(np.eye(18) - Q)           # matriz fundamental
    B = N @ R                                   # prob. de absorção
    t = N @ np.ones(18)                         # nº esperado de pontos até ao fim
    var = (2 * N - np.eye(18)) @ t - t**2       # variância
    return N, B, t, var


def parte_A(p=0.64):
    print(f"\n=========== PARTE A: game com p = {p} ===========")
    P = matriz_game(p)
    print("Soma das linhas = 1 ?", np.allclose(P.sum(axis=1), 1), "| nº de estados:", len(P))
    N, B, t, var = analise_absorvente(P)
    i0 = IDX["0-0"]
    print(f"P(servidor ganha o game | 0-0) pela cadeia   = {B[i0,0]:.6f}")
    print(f"P(servidor ganha o game)       fórmula fechada = {game_formula(p):.6f}")
    print(f"Nº esperado de pontos de um game = {t[i0]:.3f}  (desvio-padrão {np.sqrt(var[i0]):.3f})")

    print("\nProbabilidade de ganhar o game a partir de cada score (distribuição inicial = delta no score):")
    for n in ("0-0", "15-0", "0-15", "30-30", "40-0", "0-40", "Iguais(40-40)"):
        print(f"   {n:14s} -> {B[IDX[n],0]:.4f}   (pontos esperados restantes: {t[IDX[n]]:.2f})")

    # n passos: distribuição do nº de pontos do game
    v = np.zeros(20); v[i0] = 1; acab = []
    for n in range(1, 31):
        v = v @ P; acab.append(v[18] + v[19])
    pm = np.diff([0] + acab)
    print("\nP(game acaba exatamente ao ponto n):  n=4:", pm[3].round(4), " n=5:", pm[4].round(4),
          " n=6:", pm[5].round(4), " n=7:", pm[6].round(4), "| P(dura >10 pontos) =", round(1 - acab[9], 4))

    # estacionária vs limite
    print("\nEstacionária vs limite:")
    print("  Estados absorventes: 'Game Servidor' e 'Game Rececao' (cada um é uma classe fechada).")
    print("  Qualquer pi = (1-a)*delta_GS + a*delta_GR (0<=a<=1) é estacionária -> NÃO é única.")
    Pn = np.linalg.matrix_power(P, 200)
    print("  lim P^n a partir de 0-0:", Pn[i0, 18:].round(4), "-> depende do estado inicial; existe mas não é uma estacionária única.")
    print("  Classes: 18 transientes (comunicam apenas 40-40<->Vantagens), 2 recorrentes absorventes (período 1).")

    # sensibilidade
    ps = np.linspace(0.3, 0.8, 11)
    print("\nAmplificação (p ponto -> P(game)):")
    for pp in (0.50, 0.55, 0.60, 0.64, 0.70):
        print(f"   p={pp:.2f} -> P(game)={game_formula(pp):.4f}")
    return P


# ----------------------------------------------------------------- PARTE B
@lru_cache(None)
def _tb(a, b, pa, pb, cap=120):
    """P(A ganha o tie-break) a partir de (a,b) pontos. A serve o 1.º ponto; depois alternam de 2 em 2."""
    if (a >= 7 or b >= 7) and abs(a - b) >= 2:
        return 1.0 if a > b else 0.0
    k = a + b
    if k >= cap:
        return 0.5
    servidor_A = (k == 0) or (((k + 1) // 2) % 2 == 0)
    pw = pa if servidor_A else 1 - pb          # P(A ganha este ponto)
    return pw * _tb(a + 1, b, pa, pb) + (1 - pw) * _tb(a, b + 1, pa, pb)


def prob_set(pa, pb):
    """P(A ganha o set). pa = P(A ganha ponto ao serviço), pb = P(B ganha ponto ao serviço)."""
    hA, hB = game_formula(pa), game_formula(pb)

    @lru_cache(None)
    def S(gA, gB):
        if gA >= 6 and gA - gB >= 2: return 1.0
        if gB >= 6 and gB - gA >= 2: return 0.0
        if gA == 7: return 1.0
        if gB == 7: return 0.0
        if gA == 6 and gB == 6: return _tb(0, 0, pa, pb)
        k = gA + gB
        pg = hA if k % 2 == 0 else 1 - hB      # P(A ganha este jogo): A serve nos jogos pares
        return pg * S(gA + 1, gB) + (1 - pg) * S(gA, gB + 1)
    return S(0, 0)


def parte_B(pa=0.66, pb=0.62):
    print(f"\n=========== PARTE B: set e match (pA serviço={pa}, pB serviço={pb}) ===========")
    hA, hB = game_formula(pa), game_formula(pb)
    s = prob_set(pa, pb)
    m3 = s**2 * (3 - 2 * s)
    m5 = s**3 * (10 - 15 * s + 6 * s**2)
    print(f"P(A segura serviço)={hA:.3f}   P(B segura serviço)={hB:.3f}")
    print(f"P(A ganha um set)={s:.4f}   P(A ganha melhor de 3)={m3:.4f}   P(melhor de 5)={m5:.4f}")
    print("Efeito de pequenas vantagens por ponto (pA=pB+delta, pB=0.62):")
    for d in (0.00, 0.01, 0.02, 0.04):
        ss = prob_set(0.62 + d, 0.62)
        print(f"   delta={d:.2f}: P(set)={ss:.4f}   P(melhor de 3)={ss**2*(3-2*ss):.4f}")


# ----------------------------------------------------------------- PARTE C
def parte_C(csv):
    import pandas as pd
    df = pd.read_csv(csv)
    linhas = []
    for pre in ("w_", "l_"):
        d = df[[pre+"svpt", pre+"1stWon", pre+"2ndWon", pre+"SvGms", pre+"bpSaved", pre+"bpFaced"]].dropna()
        d.columns = ["svpt", "w1", "w2", "svgms", "saved", "faced"]
        linhas.append(d)
    d = pd.concat(linhas); d = d[(d.svpt > 0) & (d.svgms > 0)]
    d["p"] = (d.w1 + d.w2) / d.svpt
    d["hold_obs"] = 1 - (d.faced - d.saved) / d.svgms
    d["hold_mod"] = game_formula(d["p"].values)
    print(f"\n=========== PARTE C: {len(d)} (jogador, jogo) de {csv} ===========")
    print("p médio ao serviço:", round(d.p.mean(), 4))
    print("P(hold) observada média:", round(d.hold_obs.mean(), 4), "| prevista pela cadeia (com p do jogador):", round(d.hold_mod.mean(), 4))
    d["bin"] = np.digitize(d.p, [0.55, 0.60, 0.65, 0.70, 0.75])
    print(d.groupby("bin")[["p", "hold_obs", "hold_mod"]].mean().round(4))
    return d


def figuras(P):
    ps = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ps, ps, "--", c="gray", label="y = p"); ax[0].plot(ps, game_formula(ps), label="P(ganhar game)")
    ax[0].set_xlabel("p = P(ganhar um ponto)"); ax[0].legend(); ax[0].set_title("Amplificação: ponto → game")
    ax[1].imshow(matriz_game(0.64)[:18, :], cmap="Blues"); ax[1].set_title("Matriz P (p=0.64), 18 transientes")
    plt.tight_layout(); plt.savefig("figuras_tenis.png", dpi=150)
    print("Figura guardada em figuras_tenis.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--atp"); ap.add_argument("--p", type=float, default=0.64)
    a = ap.parse_args()
    P = parte_A(a.p); parte_B(); figuras(P)
    if a.atp: parte_C(a.atp)
