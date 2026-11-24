"""
The Markov chain of a tennis service game tested against real point-by-point data
(Match Charting Project, Jeff Sackmann; licence CC BY-NC-SA 4.0 - cite the source).

Usage (after running download_charting_data.py):
    python charting_analysis.py

Needs numpy, matplotlib and tennis_markov.py in the same folder. Does NOT use pandas or scipy.

What it does
  1. Rebuilds every game from its points and checks that it follows the rules of the chain (20 states).
     Tie-breaks and incomplete or inconsistent games are left out (and counted).
  2. Estimates p (probability that the server wins a point) and compares the real share of games
     held with the share the chain predicts.
  3. Tests the homogeneous-chain assumption: is p the same at every score? (with and without
     adjusting for differences between players)
  4. Estimates the real transition matrix (p per state) and compares it with the theoretical chain.
  5. Compares the distribution of the number of points in a game (real vs chain).
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tennis_markov import game_matrix, game_formula, absorbing_analysis, IDX, FIGURES

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "charting_data"
POINT_FILES = [("charting-m-points-2010s.csv", "2010s"), ("charting-m-points-2020s.csv", "2020s")]
MATCH_FILE = "charting-m-matches.csv"
csv.field_size_limit(10**7)

TOK = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
PT = ["0", "15", "30", "40"]
STATES = [(a, b) for a in range(4) for b in range(4)] + [(4, 3), (3, 4)]      # 18 transient states
SIDX = {s: i for i, s in enumerate(STATES)}


def label(s):
    a, b = s
    if s == (4, 3): return "AD-40"
    if s == (3, 4): return "40-AD"
    return f"{PT[a]}-{PT[b]}"


def chain_name(s):                        # state names used in tennis_markov.py
    if s == (3, 3): return "Deuce"
    if s == (4, 3): return "Adv. server"
    if s == (3, 4): return "Adv. receiver"
    return f"{PT[s[0]]}-{PT[s[1]]}"


def read_score(txt):
    """'30-15' -> (2, 1); returns None if it is not a normal game score (e.g. tie-break '3-3')."""
    try:
        x, y = txt.strip().split("-")
        a, b = TOK[x], TOK[y]
    except (ValueError, KeyError, AttributeError):
        return None
    if (a == 4 and b != 3) or (b == 4 and a != 3) or (a == 4 and b == 4):
        return None
    return (a, b)


def step(s, server_wins):
    """Chain rule: returns the next state, or 'S' / 'R' (game to the server / to the receiver)."""
    a, b = s
    if server_wins:
        if a == 4: return "S"
        if b == 4: return (3, 3)
        if (a, b) == (3, 3): return (4, 3)
        return "S" if a == 3 else (a + 1, b)
    if b == 4: return "R"
    if a == 4: return (3, 3)
    if (a, b) == (3, 3): return (3, 4)
    return "R" if b == 3 else (a, b + 1)


def chi2_sf(x, df):
    """P(chi2_df > x) without scipy (regularised incomplete gamma function)."""
    if x <= 0: return 1.0
    a, z = df / 2.0, x / 2.0
    if z < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1; term *= z / n; total += term
            if abs(term) < abs(total) * 1e-14: break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b = z + 1 - a; c = 1e300; d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-14: break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


# ------------------------------------------------------------------ loading
def load():
    surface = {}
    if (DATA / MATCH_FILE).exists():
        with open(DATA / MATCH_FILE, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                if r.get("Surface") in ("Hard", "Clay", "Grass"):
                    surface[r["match_id"]] = r["Surface"]

    games, count = [], Counter()          # valid games: dict(unit, era, seq, result)
    for name, era in POINT_FILES:
        path = DATA / name
        if not path.exists():
            print(f"(warning) missing file: {path}")
            continue
        raw = defaultdict(list)
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                try:
                    pt = int(r["Pt"])
                except (ValueError, TypeError, KeyError):
                    continue
                gm = (r.get("Gm#") or "").split()[0] if r.get("Gm#") else ""
                raw[(r["match_id"], gm)].append((pt, r["Pts"], r["Svr"], r["PtWinner"]))
        for (mid, gm), rows in raw.items():
            rows.sort()
            count["games_total"] += 1
            if len({x[2] for x in rows}) != 1 or rows[0][2] not in ("1", "2"):
                count["inconsistent_server"] += 1; continue
            states = [read_score(x[1]) for x in rows]
            if any(e is None for e in states):
                count["tiebreak_or_invalid_score"] += 1; continue
            if states[0] != (0, 0):
                count["starts_mid_game"] += 1; continue
            seq, ok, end = [], True, None
            for i, (e, x) in enumerate(zip(states, rows)):
                if x[3] not in ("1", "2"):
                    ok = False; break
                sw = (x[3] == x[2])
                nxt = step(e, sw)
                if i < len(rows) - 1:
                    if nxt != states[i + 1]:
                        ok = False; break
                else:
                    if nxt not in ("S", "R"):
                        count["incomplete_at_end"] += 1; ok = None; break
                    end = nxt
                seq.append((SIDX[e], sw))
            if ok is None: continue
            if not ok:
                count["inconsistent_sequence"] += 1; continue
            count["valid_games"] += 1
            games.append(dict(unit=(mid, rows[0][2]), era=era, seq=seq, S=(end == "S"),
                              surf=surface.get(mid, "?")))
    return games, count


# ------------------------------------------------------------------ analysis
def main():
    games, count = load()
    if not games:
        print("No data. Run first: python download_charting_data.py"); return
    print("=" * 78); print("1. DATA QUALITY"); print("=" * 78)
    for k, v in count.items():
        print(f"   {k:28s} {v:8d}  ({v / count['games_total']:.1%})")
    unit_pts = defaultdict(lambda: [0, 0, 0, 0])               # unit -> [points won, points, games, holds]
    for g in games:
        u = unit_pts[g["unit"]]
        u[0] += sum(1 for _, sw in g["seq"] if sw); u[1] += len(g["seq"]); u[2] += 1; u[3] += g["S"]
    # partly charted matches have few points: player-match units with < 40 points are left out
    valid = {k for k, v in unit_pts.items() if v[1] >= 40}
    print(f"   Player-match units (one server in one match): {len(unit_pts)}; excluded for having < 40 points: {len(unit_pts) - len(valid)}")
    games = [g for g in games if g["unit"] in valid]
    unit_pts = {k: v for k, v in unit_pts.items() if k in valid}
    print(f"   Remaining: {len(unit_pts)} units, {len(games)} games, {len({k[0] for k in unit_pts})} distinct matches")

    W = sum(u[0] for u in unit_pts.values()); N = sum(u[1] for u in unit_pts.values())
    G = len(games); H = sum(g["S"] for g in games)
    p_hat = W / N; hold_obs = H / G
    pm = {k: v[0] / v[1] for k, v in unit_pts.items()}
    pred_het = sum(game_formula(pm[k]) * unit_pts[k][2] for k in unit_pts) / G

    print("\n" + "=" * 78); print("2. p AND SHARE OF GAMES HELD (chain vs reality)"); print("=" * 78)
    print(f"   Points analysed: {N}   Complete games (no tie-breaks): {G}")
    print(f"   Estimated p (server wins the point)           p^ = {p_hat:.4f}")
    print(f"   Games held, OBSERVED                               = {hold_obs:.4f}")
    print(f"   Homogeneous chain with the global p^               = {game_formula(p_hat):.4f}")
    print(f"   Chain with each player-match's own p (mean)        = {pred_het:.4f}")

    # --- by surface and era
    print("\n   By surface / era:          p^      hold obs   hold chain(p^)    games")
    for key, name in (("surf", "surface"), ("era", "era")):
        groups = defaultdict(list)
        for g in games: groups[g[key]].append(g)
        for k, gs in sorted(groups.items()):
            if k == "?" or len(gs) < 200: continue
            w = sum(sum(1 for _, s in g["seq"] if s) for g in gs); n = sum(len(g["seq"]) for g in gs)
            print(f"   {name:10s} {k:8s}     {w / n:.4f}   {sum(g['S'] for g in gs) / len(gs):.4f}      {game_formula(w / n):.4f}        {len(gs)}")

    # --- calibration by player-match p
    print("\n   Calibration: player-match units grouped by their p")
    print("   p range          units      mean p    hold OBS   hold CHAIN    difference (z)")
    lim = [0, 0.55, 0.60, 0.65, 0.70, 1.01]
    cal = []
    for lo, hi in zip(lim[:-1], lim[1:]):
        ks = [k for k in unit_pts if lo <= pm[k] < hi]
        if not ks: continue
        g_tot = sum(unit_pts[k][2] for k in ks)
        obs = sum(unit_pts[k][3] for k in ks) / g_tot
        pred = sum(game_formula(pm[k]) * unit_pts[k][2] for k in ks) / g_tot
        var = sum(game_formula(pm[k]) * (1 - game_formula(pm[k])) * unit_pts[k][2] for k in ks) / g_tot ** 2
        z = (obs - pred) / math.sqrt(var)
        pmean = sum(pm[k] * unit_pts[k][1] for k in ks) / sum(unit_pts[k][1] for k in ks)
        cal.append((pmean, obs, pred, math.sqrt(var)))
        print(f"   [{lo:.2f}, {min(hi, 1):.2f})   {len(ks):6d}    {pmean:.3f}    {obs:.4f}     {pred:.4f}       {obs - pred:+.4f} ({z:+.1f})")

    # --- by state
    print("\n" + "=" * 78); print("3. IS p THE SAME AT EVERY SCORE?  (homogeneous-chain assumption)"); print("=" * 78)
    n_s = np.zeros(18); w_s = np.zeros(18); e_s = np.zeros(18); v_s = np.zeros(18)
    for g in games:
        u = unit_pts[g["unit"]]
        wg, ng = sum(sw for _, sw in g["seq"]), len(g["seq"])
        # expected p = p of the same player-match computed WITHOUT this game (so the path that led
        # to this score does not enter the estimate, which would bias the comparison)
        p_u = (u[0] - wg) / (u[1] - ng) if u[1] - ng >= 10 else p_hat
        for i, sw in g["seq"]:
            n_s[i] += 1; w_s[i] += sw; e_s[i] += p_u; v_s[i] += p_u * (1 - p_u)
    ps = w_s / n_s
    se = np.sqrt(ps * (1 - ps) / n_s)
    z_adj = (w_s - e_s) / np.sqrt(v_s)
    print("   State     visits    p^_s    95% CI           expected*   deviation  z(adj.)")
    for i, s in enumerate(STATES):
        print(f"   {label(s):7s} {int(n_s[i]):8d}   {ps[i]:.4f}  [{ps[i]-1.96*se[i]:.3f},{ps[i]+1.96*se[i]:.3f}]   {e_s[i]/n_s[i]:.4f}    {ps[i]-e_s[i]/n_s[i]:+.4f}   {z_adj[i]:+6.2f}")
    print("   * expected = mean, over the visits to that score, of the same player's p in the same match computed")
    print("     from the OTHER games (what we would expect if each player had a constant p). Better players reach")
    print("     favourable scores more often, so the fair comparison is with this column, not with the global p^.")
    chi_raw = float(np.sum((w_s - n_s * p_hat) ** 2 / (n_s * p_hat * (1 - p_hat))))
    chi_adj = float(np.sum(z_adj ** 2))
    print(f"\n   Test (a): same p at every score, unadjusted:        chi2 = {chi_raw:9.1f}, df = 17, p-value = {chi2_sf(chi_raw, 17):.3g}")
    print(f"   Test (b): with each player-match's p (adjusted):    chi2 = {chi_adj:9.1f}, df = 18, p-value = {chi2_sf(chi_adj, 18):.3g}")
    print("   With this many points any tiny deviation is 'significant': look at the SIZE of the deviation.")
    bp = [SIDX[s] for s in [(0, 3), (1, 3), (2, 3), (3, 4)]]       # 0-40, 15-40, 30-40, 40-AD (break points)
    bpn, bpw, bpe = n_s[bp].sum(), w_s[bp].sum(), e_s[bp].sum()
    print(f"   Break points (server wins the point): observed {bpw / bpn:.4f}  vs expected {bpe / bpn:.4f}  (n = {int(bpn)})")

    # --- empirical chain
    print("\n" + "=" * 78); print("4. CHAIN ESTIMATED FROM THE DATA (matrix with p per state)"); print("=" * 78)
    Pemp = np.zeros((20, 20)); Pemp[18, 18] = Pemp[19, 19] = 1
    for i, s in enumerate(STATES):
        for sw, p in ((True, ps[i]), (False, 1 - ps[i])):
            nx = step(s, sw)
            j = 18 if nx == "S" else 19 if nx == "R" else IDX[chain_name(nx)]
            Pemp[IDX[chain_name(s)], j] += p
    # the matrix above uses the order of tennis_markov.py (IDX); no reordering needed
    N_e, B_e, t_e, var_e = absorbing_analysis(Pemp)
    Ph = game_matrix(p_hat); N_h, B_h, t_h, var_h = absorbing_analysis(Ph)
    i0 = IDX["0-0"]
    lengths = Counter(len(g["seq"]) for g in games)
    mean_obs = sum(k * v for k, v in lengths.items()) / G
    var_obs = sum((k - mean_obs) ** 2 * v for k, v in lengths.items()) / G
    print("                                        P(server wins game)      expected points (sd)")
    print(f"   Observed in the data                  {hold_obs:.4f}                   {mean_obs:.3f}  ({math.sqrt(var_obs):.3f})")
    print(f"   Homogeneous chain (p^ = {p_hat:.3f})        {B_h[i0, 0]:.4f}                   {t_h[i0]:.3f}  ({math.sqrt(var_h[i0]):.3f})")
    print(f"   Empirical chain (p^_s per state)      {B_e[i0, 0]:.4f}                   {t_e[i0]:.3f}  ({math.sqrt(var_e[i0]):.3f})")

    # --- game length distribution
    print("\n" + "=" * 78); print("5. GAME LENGTH: number of points (observed vs homogeneous chain)"); print("=" * 78)
    v = np.zeros(20); v[i0] = 1; ended = [0.0]
    for _ in range(40):
        v = v @ Ph; ended.append(v[18] + v[19])
    pred_len = np.diff(ended)                     # pred_len[n-1] = P(T = n)
    print("   points     observed    chain(p^)")
    ns = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    for n in ns:
        print(f"   {n:5d}      {lengths.get(n, 0) / G:.4f}      {pred_len[n - 1]:.4f}")
    rest = 1 - sum(lengths.get(n, 0) for n in ns) / G
    print(f"   >=13       {rest:.4f}      {1 - sum(pred_len[n - 1] for n in ns):.4f}")
    print(f"   Games with an odd number of points >= 7 (impossible according to the chain): {sum(v_ for k, v_ in lengths.items() if k >= 7 and k % 2 == 1)}")

    # --- figure
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(18)
    ax[0].errorbar(x, ps, yerr=1.96 * se, fmt="o", c="#2563eb", label="observed (95% CI)", capsize=3)
    ax[0].plot(x, e_s / n_s, "s", c="#ea580c", ms=5, label="expected with each player's p")
    ax[0].axhline(p_hat, ls="--", c="gray", label=f"global p^ = {p_hat:.3f}")
    ax[0].set_xticks(x); ax[0].set_xticklabels([label(s) for s in STATES], rotation=90, fontsize=8)
    ax[0].set_ylabel("P(server wins the point)"); ax[0].set_title("p by score"); ax[0].legend(fontsize=8)
    if cal:
        pm_, ob_, pr_, se_ = zip(*cal)
        ax[1].errorbar(pm_, ob_, yerr=1.96 * np.array(se_), fmt="o", c="#2563eb", capsize=3, label="observed")
        xs = np.linspace(0.45, 0.80, 100); ax[1].plot(xs, game_formula(xs), c="#16a34a", label="chain")
        ax[1].set_xlabel("player-match p"); ax[1].set_ylabel("P(hold serve)")
        ax[1].set_title("Chain vs reality"); ax[1].legend()
    ns2 = list(range(4, 17))
    ax[2].bar(np.array(ns2) - 0.2, [lengths.get(n, 0) / G for n in ns2], 0.4, label="observed", color="#2563eb")
    ax[2].bar(np.array(ns2) + 0.2, [pred_len[n - 1] for n in ns2], 0.4, label="chain (p^)", color="#f59e0b")
    ax[2].set_xlabel("points in the game"); ax[2].set_title("Game length"); ax[2].legend()
    plt.tight_layout(); plt.savefig(FIGURES / "charting_results.png", dpi=160)
    print(f"\nFigure saved to: {FIGURES / 'charting_results.png'}")


if __name__ == "__main__":
    main()
