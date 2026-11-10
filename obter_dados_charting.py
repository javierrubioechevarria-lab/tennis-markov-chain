"""
Parte 2 (versao ponto a ponto): descarrega os ficheiros do Match Charting Project (Jeff Sackmann)
para a subpasta 'dados_charting' e mostra um resumo para vermos o formato real dos dados.

Uso (no terminal do VS Code, dentro da pasta):   python obter_dados_charting.py
Nao precisa de pandas: so da biblioteca padrao do Python.

Fonte: Jeff Sackmann / Match Charting Project, github.com/JeffSackmann/tennis_MatchChartingProject
Licenca: CC BY-NC-SA 4.0 (atribuicao obrigatoria, uso nao comercial) - citar no trabalho.
"""
import csv
import sys
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/{}"
FICHEIROS = ["charting-m-matches.csv", "charting-m-points-2010s.csv", "charting-m-points-2020s.csv"]
PASTA = Path(__file__).resolve().parent / "dados_charting"
csv.field_size_limit(10**7)


def descarregar(nome):
    destino = PASTA / nome
    if destino.exists() and destino.stat().st_size > 0:
        print(f"  {nome}: ja existe ({destino.stat().st_size / 1e6:.1f} MB), a saltar")
        return True
    url = BASE.format(nome)
    print(f"  {nome}: a descarregar ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r, open(destino, "wb") as f:
            total, lido, ultimo = int(r.headers.get("Content-Length") or 0), 0, 0
            while True:
                bloco = r.read(1 << 20)
                if not bloco:
                    break
                f.write(bloco); lido += len(bloco)
                if lido - ultimo >= 10 << 20:
                    ultimo = lido
                    print(f"     {lido / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else ""))
        print(f"  {nome}: OK ({destino.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        if destino.exists():
            destino.unlink()
        print(f"  {nome}: FALHOU ({type(e).__name__}: {e})")
        return False


def abrir(nome):
    return open(PASTA / nome, newline="", encoding="utf-8", errors="replace")


def resumo_matches():
    print("\n--- charting-m-matches.csv ---")
    with abrir("charting-m-matches.csv") as f:
        leitor = csv.DictReader(f)
        linhas = list(leitor)
    print("Colunas:", leitor.fieldnames)
    print("Nº de jogos:", len(linhas))
    print("1.a linha:", dict(list(linhas[0].items())))
    for col in ("Surface", "Best of", "Round", "Tournament"):
        if col in linhas[0]:
            print(f"{col}:", Counter(l[col] for l in linhas).most_common(6))


def resumo_pontos(nome):
    print(f"\n--- {nome} ---")
    n, jogos = 0, set()
    pts, svr, win, svwin, gmn, tb = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    with abrir(nome) as f:
        leitor = csv.DictReader(f)
        print("Colunas:", leitor.fieldnames)
        for i, l in enumerate(leitor):
            n += 1
            jogos.add(l.get("match_id"))
            pts[l.get("Pts")] += 1; svr[l.get("Svr")] += 1; win[l.get("PtWinner")] += 1
            svwin[l.get("isSvrWinner")] += 1; tb[l.get("TB?")] += 1
            if i < 3:
                print("exemplo de linha", i + 1, ":", {k: l[k] for k in ("match_id", "Pt", "Gm#", "Pts", "Svr", "PtWinner", "isSvrWinner", "TB?") if k in l})
            if i < 30:
                gmn[l.get("Gm#")] += 1
    print("Nº de pontos:", n, "| nº de jogos distintos:", len(jogos))
    print("Valores de 'Pts' (os 30 mais comuns):", pts.most_common(30))
    print("Svr:", dict(svr), "| PtWinner:", dict(win), "| isSvrWinner:", dict(svwin), "| TB?:", dict(tb))
    print("Exemplos de 'Gm#':", list(gmn)[:8])
    if svwin.get("1") or svwin.get("0"):
        a, b = svwin.get("1", 0), svwin.get("0", 0)
        print(f"Proporção de pontos ganhos pelo servidor (global, inclui tie-breaks): {a / (a + b):.4f}")


def main():
    PASTA.mkdir(exist_ok=True)
    print("A descarregar para:", PASTA)
    ok = [descarregar(n) for n in FICHEIROS]
    if not all(ok):
        print("\nAlguns ficheiros falharam. Plano B (manual): no browser abre")
        for n, o in zip(FICHEIROS, ok):
            if not o:
                print("  ", BASE.format(n))
        print("e guarda cada ficheiro (Ctrl+S) na pasta:", PASTA, "  (com o mesmo nome).")
    if (PASTA / FICHEIROS[0]).exists():
        resumo_matches()
    for n in FICHEIROS[1:]:
        if (PASTA / n).exists():
            resumo_pontos(n)
    print("\nCopia TODO este texto e envia-mo: com isto escrevo a Parte 3.")


if __name__ == "__main__":
    main()
