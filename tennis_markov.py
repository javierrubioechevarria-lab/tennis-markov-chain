"""
Tennis as an absorbing Markov chain.

PART A  The service game as a chain with 20 states (2 absorbing); p = P(server wins a point).
PART B  Extension: a set (games alternating serve, plus tie-break) and best-of-3 / best-of-5 matches.
PART C  (optional) Quick check against ATP match statistics (Jeff Sackmann, tennis_atp on GitHub):
        python tennis_markov.py --atp atp_matches_2023.csv

Usage:  python -m pip install -r requirements.txt   &&   python tennis_markov.py --p 0.642
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=150)
FIGURES = Path(__file__).resolve().parent / "figures"

# ----------------------------------------------------------------- PART A
PTS = ["0", "15", "30", "40"]
TRANS = [f"{PTS[a]}-{PTS[b]}" for a in range(4) for b in range(4) if (a, b) != (3, 3)]  # 15 states
TRANS += ["Deuce", "Adv. server", "Adv. receiver"]                                     # +3 = 18
ABS = ["Game server", "Game receiver"]
NAMES = TRANS + ABS            # 20 states: the first 18 are transient, the last 2 absorbing
IDX = {n: i for i, n in enumerate(NAMES)}


def game_matrix(p):
    """20 x 20 transition matrix. p = P(server wins the point), q = 1 - p."""
    q = 1 - p
    P = np.zeros((20, 20))
    for a in range(4):
        for b in range(4):
            if (a, b) == (3, 3):
                continue
            s = IDX[f"{PTS[a]}-{PTS[b]}"]
            win = IDX["Game server"] if a == 3 else (IDX["Deuce"] if (a + 1, b) == (3, 3) else IDX[f"{PTS[a+1]}-{PTS[b]}"])
            lose = IDX["Game receiver"] if b == 3 else (IDX["Deuce"] if (a, b + 1) == (3, 3) else IDX[f"{PTS[a]}-{PTS[b+1]}"])
            P[s, win] += p; P[s, lose] += q
    d, vs, vr = IDX["Deuce"], IDX["Adv. server"], IDX["Adv. receiver"]
    P[d, vs] = p;  P[d, vr] = q
    P[vs, IDX["Game server"]] = p;  P[vs, d] = q
    P[vr, IDX["Game receiver"]] = q;  P[vr, d] = p
    P[18, 18] = 1; P[19, 19] = 1
    return P


def game_formula(p):
    """P(server wins the game) in closed form."""
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q**2) + 20 * p**3 * q**3 * p**2 / (p**2 + q**2)


def absorbing_analysis(P):
    Q, R = P[:18, :18], P[:18, 18:]
    N = np.linalg.inv(np.eye(18) - Q)           # fundamental matrix
    B = N @ R                                   # absorption probabilities
    t = N @ np.ones(18)                         # expected number of points until the end
    var = (2 * N - np.eye(18)) @ t - t**2       # variance of that number
    return N, B, t, var


def part_A(p=0.64):
    print(f"\n=========== PART A: service game with p = {p} ===========")
    P = game_matrix(p)
    print("Rows sum to 1?", np.allclose(P.sum(axis=1), 1), "| number of states:", len(P))
    N, B, t, var = absorbing_analysis(P)
    i0 = IDX["0-0"]
    print(f"P(server wins the game | 0-0), from the chain  = {B[i0,0]:.6f}")
    print(f"P(server wins the game), closed formula        = {game_formula(p):.6f}")
    print(f"Expected number of points in a game = {t[i0]:.3f}  (standard deviation {np.sqrt(var[i0]):.3f})")

    print("\nProbability of winning the game from each score:")
    for n in ("0-0", "15-0", "0-15", "30-30", "40-0", "0-40", "Deuce"):
        print(f"   {n:14s} -> {B[IDX[n],0]:.4f}   (expected points left: {t[IDX[n]]:.2f})")

    # n steps: distribution of the number of points in a game
    v = np.zeros(20); v[i0] = 1; ended = []
    for n in range(1, 31):
        v = v @ P; ended.append(v[18] + v[19])
    pm = np.diff([0] + ended)
    print("\nP(game ends exactly at point n):  n=4:", pm[3].round(4), " n=5:", pm[4].round(4),
          " n=6:", pm[5].round(4), " n=7:", pm[6].round(4), "| P(lasts > 10 points) =", round(1 - ended[9], 4))

    # stationary vs limiting distribution
    print("\nStationary vs limiting distribution:")
    print("  Absorbing states: 'Game server' and 'Game receiver' (each one is a closed class).")
    print("  Any pi = (1-a)*delta_GS + a*delta_GR (0<=a<=1) is stationary -> it is NOT unique.")
    Pn = np.linalg.matrix_power(P, 200)
    print("  lim P^n from 0-0:", Pn[i0, 18:].round(4), "-> it depends on the starting state; it exists but is not a unique stationary distribution.")
    print("  Classes: 18 transient states (only Deuce <-> Advantages communicate), 2 absorbing recurrent states (period 1).")

    # sensitivity
    print("\nAmplification (p per point -> P(game)):")
    for pp in (0.50, 0.55, 0.60, 0.64, 0.70):
        print(f"   p={pp:.2f} -> P(game)={game_formula(pp):.4f}")
    return P


# ----------------------------------------------------------------- PART B
@lru_cache(None)
def _tb(a, b, pa, pb, cap=120):
    """P(A wins the tie-break) from (a, b) points. A serves the first point; then serve alternates every two points."""
    if (a >= 7 or b >= 7) and abs(a - b) >= 2:
        return 1.0 if a > b else 0.0
    k = a + b
    if k >= cap:
        return 0.5
    a_serves = (k == 0) or (((k + 1) // 2) % 2 == 0)
    pw = pa if a_serves else 1 - pb          # P(A wins this point)
    return pw * _tb(a + 1, b, pa, pb) + (1 - pw) * _tb(a, b + 1, pa, pb)


def set_probability(pa, pb):
    """P(A wins the set). pa = P(A wins a point on serve), pb = P(B wins a point on serve)."""
    hA, hB = game_formula(pa), game_formula(pb)

    @lru_cache(None)
    def S(gA, gB):
        if gA >= 6 and gA - gB >= 2: return 1.0
        if gB >= 6 and gB - gA >= 2: return 0.0
        if gA == 7: return 1.0
        if gB == 7: return 0.0
        if gA == 6 and gB == 6: return _tb(0, 0, pa, pb)
        k = gA + gB
        pg = hA if k % 2 == 0 else 1 - hB      # P(A wins this game): A serves the even-numbered games
        return pg * S(gA + 1, gB) + (1 - pg) * S(gA, gB + 1)
    return S(0, 0)


def part_B(pa=0.66, pb=0.62):
    print(f"\n=========== PART B: set and match (A on serve={pa}, B on serve={pb}) ===========")
    hA, hB = game_formula(pa), game_formula(pb)
    s = set_probability(pa, pb)
    m3 = s**2 * (3 - 2 * s)
    m5 = s**3 * (10 - 15 * s + 6 * s**2)
    print(f"P(A holds serve)={hA:.3f}   P(B holds serve)={hB:.3f}")
    print(f"P(A wins a set)={s:.4f}   P(A wins best of 3)={m3:.4f}   P(best of 5)={m5:.4f}")
    print("Effect of small edges per point (pA = pB + delta, pB = 0.62):")
    for d in (0.00, 0.01, 0.02, 0.04):
        ss = set_probability(0.62 + d, 0.62)
        print(f"   delta={d:.2f}: P(set)={ss:.4f}   P(best of 3)={ss**2*(3-2*ss):.4f}")


# ----------------------------------------------------------------- PART C
def part_C(csv):
    import pandas as pd
    df = pd.read_csv(csv)
    rows = []
    for pre in ("w_", "l_"):
        d = df[[pre+"svpt", pre+"1stWon", pre+"2ndWon", pre+"SvGms", pre+"bpSaved", pre+"bpFaced"]].dropna()
        d.columns = ["svpt", "w1", "w2", "svgms", "saved", "faced"]
        rows.append(d)
    d = pd.concat(rows); d = d[(d.svpt > 0) & (d.svgms > 0)]
    d["p"] = (d.w1 + d.w2) / d.svpt
    d["hold_obs"] = 1 - (d.faced - d.saved) / d.svgms
    d["hold_model"] = game_formula(d["p"].values)
    print(f"\n=========== PART C: {len(d)} (player, match) rows from {csv} ===========")
    print("Mean p on serve:", round(d.p.mean(), 4))
    print("Mean observed P(hold):", round(d.hold_obs.mean(), 4), "| predicted by the chain (with each player's p):", round(d.hold_model.mean(), 4))
    d["bin"] = np.digitize(d.p, [0.55, 0.60, 0.65, 0.70, 0.75])
    print(d.groupby("bin")[["p", "hold_obs", "hold_model"]].mean().round(4))
    return d


def figures(P):
    FIGURES.mkdir(exist_ok=True)
    ps = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ps, ps, "--", c="gray", label="y = p"); ax[0].plot(ps, game_formula(ps), label="P(win the game)")
    ax[0].set_xlabel("p = P(win a point)"); ax[0].legend(); ax[0].set_title("Amplification: point → game")
    ax[1].imshow(game_matrix(0.64)[:18, :], cmap="Blues"); ax[1].set_title("Matrix P (p = 0.64), 18 transient rows")
    plt.tight_layout(); plt.savefig(FIGURES / "amplification_and_matrix.png", dpi=150)
    print("Figure saved to figures/amplification_and_matrix.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--atp"); ap.add_argument("--p", type=float, default=0.64)
    a = ap.parse_args()
    P = part_A(a.p); part_B(); figures(P)
    if a.atp: part_C(a.atp)
