# recalcul_scores.py
import json
import re
from pathlib import Path

def extraire_score(analyse):
    match = re.search(
        r"SCORE(?:[^:\n]*)?\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        analyse, re.IGNORECASE
    )
    if match:
        brut = float(match.group(1))
        return round(max(0.0, min(10.0, brut)) / 10.0, 2)
    return None  # None plutôt que 0.5 pour distinguer "non parsé" de "score nul"

chemin = "resultats/enrich.json"
data = json.loads(Path(chemin).read_text(encoding="utf-8"))

corriges = 0
non_parses = 0
for para in data["paragraphes"]:
    ancien = para.get("score", None)
    nouveau = extraire_score(para.get("analyse", ""))
    if nouveau is None:
        print(f"  ⚠ score non parsé : {para.get('id', '?')}")
        non_parses += 1
        continue  # on laisse l'ancien score intact plutôt que d'écraser avec 0.5
    if nouveau != ancien:
        para["score"] = nouveau
        corriges += 1

# Recalcul du score moyen en excluant les paragraphes sans score
scores = [p["score"] for p in data["paragraphes"] if p.get("score") is not None]
if scores:
    data["run"]["score_moyen"] = round(sum(scores) / len(scores), 3)
else:
    print("  ⚠ aucun score valide — score_moyen non recalculé")

Path(chemin).write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"✓ {corriges} scores corrigés")
if non_parses:
    print(f"  ⚠ {non_parses} paragraphe(s) non parsés (scores inchangés)")
print(f"  Score moyen : {data['run'].get('score_moyen', 'n/a')}")
