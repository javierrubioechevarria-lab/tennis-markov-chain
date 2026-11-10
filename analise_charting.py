"""
Parte 3: a cadeia de Markov do game de ténis confrontada com dados reais ponto a ponto
(Match Charting Project, Jeff Sackmann; licenca CC BY-NC-SA 4.0 - citar a fonte).

Uso (no terminal do VS Code, dentro da pasta Pyton, depois de correr obter_dados_charting.py):
    python analise_charting.py

Precisa de: numpy, matplotlib (ja instalados) e do ficheiro tenis_markov.py na mesma pasta.
NAO usa pandas nem scipy.

O que faz
  1. Reconstroi cada game a partir dos pontos e verifica que cumpre as regras da cadeia (20 estados).
     Tie-breaks e games incompletos ou incoerentes ficam de fora (e sao contados).
  2. Estima p (probabilidade de o servidor ganhar um ponto) e compara a taxa real de games
     mantidos com a que a cadeia preve.
  3. Testa a hipotese da cadeia homogenea: p e igual em todos os scores? (com e sem ajuste
     a heterogeneidade entre jogadores)
  4. Estima a matriz de transicao real (p por estado) e compara com a cadeia teorica.
  5. Compara a distribuicao do numero de pontos de um game (real contra cadeia).
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tenis_markov import matriz_game, game_formula, analise_absorvente, IDX

PASTA = Path(__file__).resolve().parent
DADOS = PASTA / "dados_charting"
FICH_PONTOS = [("charting-m-points-2010s.csv", "2010s"), ("charting-m-points-2020s.csv", "2020s")]
FICH_JOGOS = "charting-m-matches.csv"
csv.field_size_limit(10**7)

TOK = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
PT = ["0", "15", "30", "40"]
ESTADOS = [(a, b) for a in range(4) for b in range(4)] + [(4, 3), (3, 4)]      # 18 estados transientes
EIDX = {s: i for i, s in enumerate(ESTADOS)}


def rotulo(s):
    a, b = s
    if s == (4, 3): return "AD-40"
    if s == (3, 4): return "40-AD"
    return f"{PT[a]}-{PT[b]}"


def nome_cadeia(s):                       # nomes usados em tenis_markov.py
    if s == (3, 3): return "Iguais(40-40)"
    if s == (4, 3): return "Vant.Servidor"
    if s == (3, 4): return "Vant.Rececao"
    return f"{PT[s[0]]}-{PT[s[1]]}"


def ler_score(txt):
    """'30-15' -> (2, 1); devolve None se nao for um score normal de game (ex.: tie-break '3-3')."""
    try:
        x, y = txt.strip().split("-")
        a, b = TOK[x], TOK[y]
    except (ValueError, KeyError, AttributeError):
        return None
    if (a == 4 and b != 3) or (b == 4 and a != 3) or (a == 4 and b == 4):
        return None
    return (a, b)


def passo(s, servidor_ganha):
    """Regra da cadeia: devolve o proximo estado ou 'S' / 'R' (game para o servidor / para o recetor)."""
    a, b = s
    if servidor_ganha:
        if a == 4: return "S"
        if b == 4: return (3, 3)
        if (a, b) == (3, 3): return (4, 3)
        return "S" if a == 3 else (a + 1, b)
    if b == 4: return "R"
    if a == 4: return (3, 3)
    if (a, b) == (3, 3): return (3, 4)
    return "R" if b == 3 else (a, b + 1)


def chi2_sf(x, df):
    """P(chi2_df > x) sem scipy (gama incompleta regularizada)."""
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


# ------------------------------------------------------------------ leitura
def carregar():
    superficie = {}
    if (DADOS / FICH_JOGOS).exists():
        with open(DADOS / FICH_JOGOS, newline="", encoding="utf-8", errors="replace") as f:
            for l in csv.DictReader(f):
                if l.get("Surface") in ("Hard", "Clay", "Grass"):
                    superficie[l["match_id"]] = l["Surface"]

    games, cont = [], Counter()          # games validos: dict(unidade, era, seq, resultado)
    for nome, era in FICH_PONTOS:
        caminho = DADOS / nome
        if not caminho.exists():
            print(f"(aviso) ficheiro em falta: {caminho}")
            continue
        bruto = defaultdict(list)
        with open(caminho, newline="", encoding="utf-8", errors="replace") as f:
            for l in csv.DictReader(f):
                try:
                    pt = int(l["Pt"])
                except (ValueError, TypeError, KeyError):
                    continue
                gm = (l.get("Gm#") or "").split()[0] if l.get("Gm#") else ""
                bruto[(l["match_id"], gm)].append((pt, l["Pts"], l["Svr"], l["PtWinner"]))
        for (mid, gm), linhas in bruto.items():
            linhas.sort()
            cont["games_total"] += 1
            if len({x[2] for x in linhas}) != 1 or linhas[0][2] not in ("1", "2"):
                cont["servidor_incoerente"] += 1; continue
            estados = [ler_score(x[1]) for x in linhas]
            if any(e is None for e in estados):
                cont["tiebreak_ou_score_invalido"] += 1; continue
            if estados[0] != (0, 0):
                cont["comeca_a_meio_do_game"] += 1; continue
            seq, ok, fim = [], True, None
            for i, (e, x) in enumerate(zip(estados, linhas)):
                if x[3] not in ("1", "2"):
                    ok = False; break
                sw = (x[3] == x[2])
                prox = passo(e, sw)
                if i < len(linhas) - 1:
                    if prox != estados[i + 1]:
                        ok = False; break
                else:
                    if prox not in ("S", "R"):
                        cont["incompleto_no_fim"] += 1; ok = None; break
                    fim = prox
                seq.append((EIDX[e], sw))
            if ok is None: continue
            if not ok:
                cont["sequencia_incoerente"] += 1; continue
            cont["games_validos"] += 1
            games.append(dict(unid=(mid, linhas[0][2]), era=era, seq=seq, S=(fim == "S"),
                              sup=superficie.get(mid, "?")))
    return games, cont


# ------------------------------------------------------------------ analise
def main():
    games, cont = carregar()
    if not games:
        print("Sem dados. Corre primeiro: python obter_dados_charting.py"); return
    print("=" * 78); print("1. QUALIDADE DOS DADOS"); print("=" * 78)
    for k, v in cont.items():
        print(f"   {k:28s} {v:8d}  ({v / cont['games_total']:.1%})")
    unid_pts = defaultdict(lambda: [0, 0, 0, 0])               # unidade -> [pts ganhos, pts, games, holds]
    for g in games:
        u = unid_pts[g["unid"]]
        u[0] += sum(1 for _, sw in g["seq"] if sw); u[1] += len(g["seq"]); u[2] += 1; u[3] += g["S"]
    # jogos registados so em parte tem poucos pontos: ficam de fora as unidades jogador-jogo com < 40 pontos
    validas = {k for k, v in unid_pts.items() if v[1] >= 40}
    print(f"   Unidades jogador-jogo (um servidor num jogo): {len(unid_pts)}; excluídas por terem < 40 pontos: {len(unid_pts) - len(validas)}")
    games = [g for g in games if g["unid"] in validas]
    unid_pts = {k: v for k, v in unid_pts.items() if k in validas}
    print(f"   Ficam: {len(unid_pts)} unidades, {len(games)} games, {len({k[0] for k in unid_pts})} jogos distintos")

    W = sum(u[0] for u in unid_pts.values()); N = sum(u[1] for u in unid_pts.values())
    G = len(games); H = sum(g["S"] for g in games)
    p_hat = W / N; hold_obs = H / G
    pm = {k: v[0] / v[1] for k, v in unid_pts.items()}
    prev_het = sum(game_formula(pm[k]) * unid_pts[k][2] for k in unid_pts) / G

    print("\n" + "=" * 78); print("2. p E TAXA DE GAMES MANTIDOS (cadeia contra realidade)"); print("=" * 78)
    print(f"   Pontos analisados: {N}   Games completos (sem tie-break): {G}")
    print(f"   p estimado (servidor ganha o ponto)           p^ = {p_hat:.4f}")
    print(f"   Games mantidos OBSERVADOS                          = {hold_obs:.4f}")
    print(f"   Cadeia homogénea com p^ global                     = {game_formula(p_hat):.4f}")
    print(f"   Cadeia com o p de cada jogador-jogo (média)        = {prev_het:.4f}")

    # --- por superficie e era
    print("\n   Por superfície / era:      p^      hold obs   hold cadeia(p^)   nº games")
    for chave, nome in (("sup", "superfície"), ("era", "era")):
        grupos = defaultdict(list)
        for g in games: grupos[g[chave]].append(g)
        for k, gs in sorted(grupos.items()):
            if k == "?" or len(gs) < 200: continue
            w = sum(sum(1 for _, s in g["seq"] if s) for g in gs); n = sum(len(g["seq"]) for g in gs)
            print(f"   {nome:10s} {k:8s}     {w / n:.4f}   {sum(g['S'] for g in gs) / len(gs):.4f}      {game_formula(w / n):.4f}        {len(gs)}")

    # --- calibracao por p do jogador-jogo
    print("\n   Calibração: jogador-jogo agrupados pelo seu p")
    print("   Intervalo p      nº unid.   p médio   hold OBS   hold CADEIA   diferença (z)")
    lim = [0, 0.55, 0.60, 0.65, 0.70, 1.01]
    cal = []
    for lo, hi in zip(lim[:-1], lim[1:]):
        ks = [k for k in unid_pts if lo <= pm[k] < hi]
        if not ks: continue
        g_tot = sum(unid_pts[k][2] for k in ks)
        obs = sum(unid_pts[k][3] for k in ks) / g_tot
        prev = sum(game_formula(pm[k]) * unid_pts[k][2] for k in ks) / g_tot
        var = sum(game_formula(pm[k]) * (1 - game_formula(pm[k])) * unid_pts[k][2] for k in ks) / g_tot ** 2
        z = (obs - prev) / math.sqrt(var)
        pmed = sum(pm[k] * unid_pts[k][1] for k in ks) / sum(unid_pts[k][1] for k in ks)
        cal.append((pmed, obs, prev, math.sqrt(var)))
        print(f"   [{lo:.2f}, {min(hi, 1):.2f})   {len(ks):6d}    {pmed:.3f}    {obs:.4f}     {prev:.4f}       {obs - prev:+.4f} ({z:+.1f})")

    # --- por estado
    print("\n" + "=" * 78); print("3. A PROBABILIDADE p É A MESMA EM TODOS OS SCORES?  (hipótese da cadeia homogénea)"); print("=" * 78)
    n_s = np.zeros(18); w_s = np.zeros(18); e_s = np.zeros(18); v_s = np.zeros(18)
    for g in games:
        u = unid_pts[g["unid"]]
        wg, ng = sum(sw for _, sw in g["seq"]), len(g["seq"])
        # p esperado = p do mesmo jogador-jogo calculado SEM este game (assim o caminho que levou a
        # este score nao entra na estimativa, o que enviesaria a comparacao)
        p_u = (u[0] - wg) / (u[1] - ng) if u[1] - ng >= 10 else p_hat
        for i, sw in g["seq"]:
            n_s[i] += 1; w_s[i] += sw; e_s[i] += p_u; v_s[i] += p_u * (1 - p_u)
    ps = w_s / n_s
    se = np.sqrt(ps * (1 - ps) / n_s)
    z_ajust = (w_s - e_s) / np.sqrt(v_s)
    print("   Estado    visitas   p^_s    IC95%            esperado*   desvio    z(ajust.)")
    for i, s in enumerate(ESTADOS):
        print(f"   {rotulo(s):7s} {int(n_s[i]):8d}   {ps[i]:.4f}  [{ps[i]-1.96*se[i]:.3f},{ps[i]+1.96*se[i]:.3f}]   {e_s[i]/n_s[i]:.4f}    {ps[i]-e_s[i]/n_s[i]:+.4f}   {z_ajust[i]:+6.2f}")
    print("   * esperado = média, nas visitas a esse score, do p do mesmo jogador no mesmo jogo calculado com os")
    print("     OUTROS games (o que se esperaria se cada jogador tivesse p constante). Jogadores melhores chegam")
    print("     mais vezes a scores favoráveis, por isso a comparação justa é com esta coluna, não com o p^ global.")
    chi_nao = float(np.sum((w_s - n_s * p_hat) ** 2 / (n_s * p_hat * (1 - p_hat))))
    chi_aj = float(np.sum(z_ajust ** 2))
    print(f"\n   Teste (a): p igual em todos os scores, sem ajuste:    chi2 = {chi_nao:9.1f}, gl = 17, p-valor = {chi2_sf(chi_nao, 17):.3g}")
    print(f"   Teste (b): com p de cada jogador-jogo (ajustado):      chi2 = {chi_aj:9.1f}, gl = 18, p-valor = {chi2_sf(chi_aj, 18):.3g}")
    print("   Com tantos pontos, qualquer desvio minúsculo é 'significativo': olhem para o TAMANHO do desvio.")
    bp = [EIDX[s] for s in [(0, 3), (1, 3), (2, 3), (3, 4)]]       # 0-40, 15-40, 30-40, 40-AD (break points)
    bpn, bpw, bpe = n_s[bp].sum(), w_s[bp].sum(), e_s[bp].sum()
    print(f"   Break points (servidor ganha o ponto): observado {bpw / bpn:.4f}  vs esperado {bpe / bpn:.4f}  (n = {int(bpn)})")

    # --- cadeia empirica
    print("\n" + "=" * 78); print("4. CADEIA ESTIMADA A PARTIR DOS DADOS (matriz com p por estado)"); print("=" * 78)
    Pemp = np.zeros((20, 20)); Pemp[18, 18] = Pemp[19, 19] = 1
    for i, s in enumerate(ESTADOS):
        for sw, p in ((True, ps[i]), (False, 1 - ps[i])):
            nx = passo(s, sw)
            j = 18 if nx == "S" else 19 if nx == "R" else IDX[nome_cadeia(nx)]
            Pemp[IDX[nome_cadeia(s)], j] += p
    # a matriz acima tem a ordem de tenis_markov.py (IDX); reordenar nao e preciso
    N_e, B_e, t_e, var_e = analise_absorvente(Pemp)
    Ph = matriz_game(p_hat); N_h, B_h, t_h, var_h = analise_absorvente(Ph)
    i0 = IDX["0-0"]
    comp = Counter(len(g["seq"]) for g in games)
    media_obs = sum(k * v for k, v in comp.items()) / G
    var_obs = sum((k - media_obs) ** 2 * v for k, v in comp.items()) / G
    print("                                        P(servidor ganha game)   nº esperado de pontos (desvio)")
    print(f"   Observado nos dados                   {hold_obs:.4f}                   {media_obs:.3f}  ({math.sqrt(var_obs):.3f})")
    print(f"   Cadeia homogénea (p^ = {p_hat:.3f})         {B_h[i0, 0]:.4f}                   {t_h[i0]:.3f}  ({math.sqrt(var_h[i0]):.3f})")
    print(f"   Cadeia empírica (p^_s por estado)     {B_e[i0, 0]:.4f}                   {t_e[i0]:.3f}  ({math.sqrt(var_e[i0]):.3f})")
    print("   (compara com p = 0,60 do Javier: hold 0.7357, 6.48 pontos; e com p = 0,64: hold 0.8126, 6.25 pontos)")

    # --- distribuicao da duracao
    print("\n" + "=" * 78); print("5. DURAÇÃO DO GAME: nº de pontos (observado contra cadeia homogénea)"); print("=" * 78)
    v = np.zeros(20); v[i0] = 1; acab = [0.0]
    for _ in range(40):
        v = v @ Ph; acab.append(v[18] + v[19])
    prev_d = np.diff(acab)                       # prev_d[n-1] = P(T = n)
    print("   n pontos   observado   cadeia(p^)")
    ns = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    for n in ns:
        print(f"   {n:5d}      {comp.get(n, 0) / G:.4f}      {prev_d[n - 1]:.4f}")
    resto = 1 - sum(comp.get(n, 0) for n in ns) / G
    print(f"   >=13       {resto:.4f}      {1 - sum(prev_d[n - 1] for n in ns):.4f}")
    print(f"   Games com número ímpar de pontos >= 7 (a cadeia diz que é impossível): {sum(v_ for k, v_ in comp.items() if k >= 7 and k % 2 == 1)}")

    # --- figura
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(18)
    ax[0].errorbar(x, ps, yerr=1.96 * se, fmt="o", c="#2563eb", label="observado (IC 95%)", capsize=3)
    ax[0].plot(x, e_s / n_s, "s", c="#ea580c", ms=5, label="esperado c/ p de cada jogador")
    ax[0].axhline(p_hat, ls="--", c="gray", label=f"p^ global = {p_hat:.3f}")
    ax[0].set_xticks(x); ax[0].set_xticklabels([rotulo(s) for s in ESTADOS], rotation=90, fontsize=8)
    ax[0].set_ylabel("P(servidor ganha o ponto)"); ax[0].set_title("p por score"); ax[0].legend(fontsize=8)
    if cal:
        pm_, ob_, pr_, se_ = zip(*cal)
        ax[1].errorbar(pm_, ob_, yerr=1.96 * np.array(se_), fmt="o", c="#2563eb", capsize=3, label="observado")
        xs = np.linspace(0.45, 0.80, 100); ax[1].plot(xs, game_formula(xs), c="#16a34a", label="cadeia")
        ax[1].set_xlabel("p do jogador-jogo"); ax[1].set_ylabel("P(manter o serviço)")
        ax[1].set_title("Cadeia contra realidade"); ax[1].legend()
    ns2 = list(range(4, 17))
    ax[2].bar(np.array(ns2) - 0.2, [comp.get(n, 0) / G for n in ns2], 0.4, label="observado", color="#2563eb")
    ax[2].bar(np.array(ns2) + 0.2, [prev_d[n - 1] for n in ns2], 0.4, label="cadeia (p^)", color="#f59e0b")
    ax[2].set_xlabel("nº de pontos do game"); ax[2].set_title("Duração do game"); ax[2].legend()
    plt.tight_layout(); plt.savefig(PASTA / "resultados_charting.png", dpi=160)
    print(f"\nFigura guardada em: {PASTA / 'resultados_charting.png'}")
    print("\nCopia TODO este texto e envia-mo para eu te ajudar a interpretar os resultados.")


if __name__ == "__main__":
    main()
