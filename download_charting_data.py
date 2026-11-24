"""
Downloads the Match Charting Project files (Jeff Sackmann) into the 'charting_data' subfolder
and prints a short summary of the real data format.

Usage:   python download_charting_data.py
Only needs the Python standard library (no pandas).

Source: Jeff Sackmann / Match Charting Project, github.com/JeffSackmann/tennis_MatchChartingProject
Licence: CC BY-NC-SA 4.0 (attribution required, non-commercial use).
"""
import csv
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/{}"
FILES = ["charting-m-matches.csv", "charting-m-points-2010s.csv", "charting-m-points-2020s.csv"]
FOLDER = Path(__file__).resolve().parent / "charting_data"
csv.field_size_limit(10**7)


def download(name):
    target = FOLDER / name
    if target.exists() and target.stat().st_size > 0:
        print(f"  {name}: already there ({target.stat().st_size / 1e6:.1f} MB), skipping")
        return True
    url = BASE.format(name)
    print(f"  {name}: downloading ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as f:
            total, read, last = int(r.headers.get("Content-Length") or 0), 0, 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block); read += len(block)
                if read - last >= 10 << 20:
                    last = read
                    print(f"     {read / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else ""))
        print(f"  {name}: OK ({target.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        if target.exists():
            target.unlink()
        print(f"  {name}: FAILED ({type(e).__name__}: {e})")
        return False


def open_file(name):
    return open(FOLDER / name, newline="", encoding="utf-8", errors="replace")


def summary_matches():
    print("\n--- charting-m-matches.csv ---")
    with open_file("charting-m-matches.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print("Columns:", reader.fieldnames)
    print("Number of matches:", len(rows))
    print("First row:", dict(list(rows[0].items())))
    for col in ("Surface", "Best of", "Round", "Tournament"):
        if col in rows[0]:
            print(f"{col}:", Counter(r[col] for r in rows).most_common(6))


def summary_points(name):
    print(f"\n--- {name} ---")
    n, matches = 0, set()
    pts, svr, win, svwin, gmn, tb = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    with open_file(name) as f:
        reader = csv.DictReader(f)
        print("Columns:", reader.fieldnames)
        for i, r in enumerate(reader):
            n += 1
            matches.add(r.get("match_id"))
            pts[r.get("Pts")] += 1; svr[r.get("Svr")] += 1; win[r.get("PtWinner")] += 1
            svwin[r.get("isSvrWinner")] += 1; tb[r.get("TB?")] += 1
            if i < 3:
                print("example row", i + 1, ":", {k: r[k] for k in ("match_id", "Pt", "Gm#", "Pts", "Svr", "PtWinner", "isSvrWinner", "TB?") if k in r})
            if i < 30:
                gmn[r.get("Gm#")] += 1
    print("Number of points:", n, "| distinct matches:", len(matches))
    print("Values of 'Pts' (30 most common):", pts.most_common(30))
    print("Svr:", dict(svr), "| PtWinner:", dict(win), "| isSvrWinner:", dict(svwin), "| TB?:", dict(tb))
    print("Examples of 'Gm#':", list(gmn)[:8])
    if svwin.get("1") or svwin.get("0"):
        a, b = svwin.get("1", 0), svwin.get("0", 0)
        print(f"Share of points won by the server (overall, tie-breaks included): {a / (a + b):.4f}")


def main():
    FOLDER.mkdir(exist_ok=True)
    print("Downloading to:", FOLDER)
    ok = [download(n) for n in FILES]
    if not all(ok):
        print("\nSome files failed. Manual fallback: open in the browser")
        for n, o in zip(FILES, ok):
            if not o:
                print("  ", BASE.format(n))
        print("and save each file (Ctrl+S) into:", FOLDER, "  (keeping the same name).")
    if (FOLDER / FILES[0]).exists():
        summary_matches()
    for n in FILES[1:]:
        if (FOLDER / n).exists():
            summary_points(n)


if __name__ == "__main__":
    main()
