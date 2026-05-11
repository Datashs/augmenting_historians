#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D_synthese_croisee.py
=====================
Synthèse croisée des analyses A (Toulmin/Walton), B (Perelman) et C (RST)
sur le même segment textuel.

RÔLE
─────
Ce script lit les JSON produits par A, B et C sur le même segment, puis
demande au LLM de produire une synthèse intégrée qui met en dialogue les
trois cadres théoriques.

La valeur de la synthèse croisée est précisément là où les cadres
DIVERGENT — pas où ils convergent. Une convergence (A voit un argument
faible, B voit une rhétorique faible, C voit des relations de preuve
absentes) confirme le diagnostic. Une divergence (A voit un argument
faible, mais B voit une rhétorique forte et C voit un noyau central bien
soutenu) est bien plus intéressante : elle signale souvent que l'auteur
argumente par PRÉSUPPOSITION ou par ACCUMULATION NARRATIVE plutôt que
par démonstration explicite — stratégie légitime en histoire.

CE QUE PRODUIT CE SCRIPT
──────────────────────────
  1. Tableau de convergence/divergence des scores des trois analyses
  2. Identification des zones d'accord et de désaccord entre A, B, C
  3. Analyse LLM de ce que la divergence signifie pour ce segment
  4. Recommandations concrètes pour l'historien (révision, consolidation)
  5. Un MD intégré lisible

DÉTECTION AUTOMATIQUE
──────────────────────
Au lancement, le script cherche automatiquement dans OUTPUT_DIR les JSON
les plus récents produits par A, B et C. Si tous trois sont présents,
il propose de les utiliser directement. Sinon, il demande les chemins.

ESTIMATION DES COÛTS
─────────────────────
Contexte = les trois synthèses (extraites des JSON) + un prompt de synthèse.
  Tokens entrée  : ~3000-4500
  Tokens sortie  : ~2000
  gpt-4.1-mini   : ~0.004-0.007 $ par synthèse

UTILISATION
───────────
  python D_synthese_croisee.py
  python D_synthese_croisee.py --toulmin resultats/toulmin_X.json \\
                                --perelman resultats/perelman_seg_X.json \\
                                --rst resultats/rst_X.json
  python D_synthese_croisee.py --no-confirm
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OPENAI_LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE       = 0.15
MAX_TOKENS        = 5000
# le nombre de token est calculé de façon à optimiser coût et qualité de la réponse
# lorsque le LLM a consommé ces tokens il arrête l'analyse
# si le passage fourni est long il est possible que la réponse soit tronquée
# il faut alors augmenter le nombre de tokens alloués. 
OUTPUT_DIR        = "resultats"

COUT_INPUT_PER_1K  = 0.00040
COUT_OUTPUT_PER_1K = 0.00160

# =============================================================================
# IMPORTS
# =============================================================================

import json
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# CHARGEMENT ET DÉTECTION
# =============================================================================

SYSTEM_PROMPT = (
    "Tu es un expert en argumentation et rhétorique académique. Tu maîtrises "
    "les cadres de Toulmin/Walton (1958/1996), la Nouvelle Rhétorique de "
    "Perelman (1958) et la RST de Mann & Thompson (1988). "
    "Tu produis des synthèses croisées qui mettent en dialogue ces trois "
    "cadres pour éclairer la pratique de l'historien. "
    "Tu valorises les zones de divergence entre cadres autant que les "
    "zones de convergence — c'est là que se cachent les informations les "
    "plus utiles sur le style argumentatif. Tu réponds en français."
)


def trouver_json_recent(output_dir: Path, pattern: str) -> Path | None:
    """Retourne le JSON le plus récent correspondant au pattern, ou None."""
    fichiers = sorted(output_dir.glob(pattern), reverse=True)
    return fichiers[0] if fichiers else None


def detecter_json_disponibles(output_dir: Path) -> dict:
    """Détecte les JSON A/B/C les plus récents dans output_dir."""
    return {
        "A": trouver_json_recent(output_dir, "toulmin_*.json"),
        "B": trouver_json_recent(output_dir, "perelman_seg_*.json"),
        "C": trouver_json_recent(output_dir, "rst_*.json"),
    }


def demander_chemin(lettre: str, pattern: str, output_dir: Path) -> Path:
    """Demande interactivement le chemin d'un JSON manquant."""
    print(f"\n  JSON du script {lettre} introuvable dans {output_dir}.")
    print(f"  Entrez le chemin du fichier ({pattern}) ou 'ignorer' pour continuer sans.")
    chemin_str = input(f"  Chemin [{lettre}] : ").strip()
    if chemin_str.lower() in ("ignorer", "skip", ""):
        return None
    p = Path(chemin_str)
    if not p.is_absolute():
        p = output_dir / p
    if not p.exists():
        print(f"  ⚠ Introuvable : {p} — ignoré.")
        return None
    return p


def charger_json(chemin: Path) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def verifier_coherence(data_a: dict | None, data_b: dict | None, data_c: dict | None) -> str | None:
    """
    Vérifie que A, B, C portent bien sur le même segment (même titre).
    Retourne un avertissement si les titres divergent, None sinon.
    """
    titres = {}
    for lettre, data in [("A", data_a), ("B", data_b), ("C", data_c)]:
        if data:
            titres[lettre] = data.get("titre", "?")
    if len(set(titres.values())) > 1:
        return (
            "⚠ Les trois analyses ne semblent pas porter sur le même segment :\n"
            + "\n".join(f"  {k} : {v}" for k, v in titres.items())
        )
    return None


# =============================================================================
# CONSTRUCTION DU CONTEXTE
# =============================================================================

def formater_scores_a(data: dict) -> str:
    s = data.get("scores", {})
    return (
        f"  Robustesse globale : {s.get('score_robustesse_globale', '?')}\n"
        f"  Cohérence warrant  : {s.get('score_coherence_warrant', '?')}\n"
        f"  Charge probatoire  : {s.get('score_charge_probatoire', '?')}\n"
        f"  Risque sophisme    : {s.get('score_risque_sophisme', '?')}\n"
        f"  Schème dominant    : {data.get('schema_walton_dominant', '?')}\n"
        f"  Sophismes          : {', '.join(data.get('sophismes_detectes', [])) or 'aucun'}\n"
        f"  Synthèse A         : {data.get('synthese', '')[:400]}"
    )


def formater_scores_b(data: dict) -> str:
    s = data.get("scores", {})
    techs = data.get("techniques_detectees", {})
    return (
        f"  Profil argumentatif : {s.get('score_profil_argumentatif', '?')}\n"
        f"  Force persuasive    : {s.get('score_force_persuasive', '?')}\n"
        f"  Ancrage auditoire   : {s.get('score_ancrage_auditoire', '?')}\n"
        f"  Cohérence valeurs   : {s.get('score_coherence_valeurs', '?')}\n"
        f"  Risque sophistique  : {s.get('score_risque_sophistique_rhethorique', '?')}\n"
        f"  Technique dominante : {data.get('technique_dominante', '?')}\n"
        f"  Auditoire           : {data.get('type_auditoire', '?')}\n"
        f"  Valeurs mobilisées  : {', '.join(data.get('valeurs_mobilisees', []))}\n"
        f"  Mouvement rhét.     : {data.get('mouvement_rhetorique', '')[:200]}\n"
        f"  Synthèse B          : {data.get('synthese', '')[:400]}"
    )


def formater_scores_c(data: dict) -> str:
    rels = data.get("relations", [])
    n_basse = sum(1 for r in rels if r.get("confiance") == "basse")
    rel_counts: dict = {}
    for r in rels:
        rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
    rel_dom = max(rel_counts, key=rel_counts.get) if rel_counts else "—"
    return (
        f"  Phrases identifiées     : {len(data.get('phrases', []))}\n"
        f"  Relations identifiées   : {len(rels)}\n"
        f"  Relation dominante      : {rel_dom}\n"
        f"  Noyau central           : {data.get('noyau', '?')}\n"
        f"  Relations faible conf.  : {n_basse}/{len(rels)}\n"
        f"  Structure d'ensemble    : {data.get('structure', '')[:300]}\n"
        f"  Synthèse C              : {data.get('synthese', '')[:400]}"
    )


def construire_prompt(
    data_a: dict | None,
    data_b: dict | None,
    data_c: dict | None,
) -> str:
    titre = next(
        (d.get("titre", "Segment sans titre")
         for d in [data_a, data_b, data_c] if d),
        "Segment sans titre"
    )
    question = next(
        (d.get("question", "")
         for d in [data_a, data_b, data_c] if d and d.get("question")),
        ""
    )

    blocs = []
    if data_a:
        blocs.append(
            "══ ANALYSE A — TOULMIN / WALTON ══\n" + formater_scores_a(data_a)
        )
    else:
        blocs.append("══ ANALYSE A — TOULMIN / WALTON ══\n  (non disponible)")

    if data_b:
        blocs.append(
            "══ ANALYSE B — PERELMAN ══\n" + formater_scores_b(data_b)
        )
    else:
        blocs.append("══ ANALYSE B — PERELMAN ══\n  (non disponible)")

    if data_c:
        blocs.append(
            "══ ANALYSE C — RST ══\n" + formater_scores_c(data_c)
        )
    else:
        blocs.append("══ ANALYSE C — RST ══\n  (non disponible)")

    question_bloc = (
        f"\n══ QUESTION DE L'HISTORIEN ══\n{question}\n"
        if question else ""
    )

    return f"""Voici les résultats de trois analyses complémentaires sur le même segment
de texte historique. Produis une synthèse croisée qui met en dialogue les trois cadres.

SEGMENT ANALYSÉ : {titre}
{chr(10).join(blocs)}
{question_bloc}
══════════════════════════════════════════
SYNTHÈSE CROISÉE DEMANDÉE
══════════════════════════════════════════

━━━ 1. TABLEAU DE CONVERGENCE ━━━
Pour chaque dimension, indique si A, B et C convergent ou divergent :
  - Solidité argumentative (A robustesse vs B profil vs C evidence)
  - Construction des preuves (A charge probatoire vs B ancrage vs C evidence/justify)
  - Cohérence d'ensemble (A warrant vs B valeurs vs C noyau central)
  - Risques rhétoriques (A sophismes vs B sophistiques vs C relations basse conf.)

━━━ 2. ZONES DE DIVERGENCE SIGNIFICATIVES ━━━
Identifie les cas où deux cadres ou plus donnent des lectures opposées.
Explique ce que chaque divergence révèle sur le style argumentatif de l'auteur.
Rappel : une divergence est souvent plus informative qu'une convergence.

━━━ 3. CE QUE LES TROIS CADRES ENSEMBLE RÉVÈLENT ━━━
Quelle image de l'argumentation de ce segment émerge de la lecture croisée ?
Comment l'auteur argumente-t-il dans ce segment ? (Par démonstration explicite,
par accumulation narrative, par autorité, par dissociation conceptuelle, par
cohérence rhétorique sans fondement logique apparent, autre ?)

━━━ 4. RECOMMANDATIONS POUR L'HISTORIEN ━━━
3-5 recommandations concrètes, classées par priorité :
  - Ce qui mérite d'être consolidé (points faibles détectés)
  - Ce qui fonctionne bien et ne doit pas être modifié
  - Une suggestion de révision prioritaire si le segment présente des failles

━━━ 5. RÉPONSE À LA QUESTION (si posée) ━━━
Réponse directe et précise à la question de l'historien, en mobilisant
les trois analyses.

━━━ 6. SYNTHÈSE FINALE ━━━
En 4-5 phrases : portrait argumentatif du segment tel qu'il émerge
du croisement des trois cadres. C'est ce texte qui sera affiché
en premier dans le rapport."""


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def _barre(v: float, w: int = 16) -> str:
    if not isinstance(v, (int, float)):
        return "N/A"
    r = round(v * w)
    return f"[{'█'*r}{'░'*(w-r)}] {v:.2f}"


def generer_rapport_md(
    data_a: dict | None,
    data_b: dict | None,
    data_c: dict | None,
    raw: str,
    timestamp: str,
    output_dir: Path,
) -> Path:
    titre = next(
        (d.get("titre", "Segment sans titre")
         for d in [data_a, data_b, data_c] if d),
        "Segment sans titre"
    )
    question = next(
        (d.get("question", "")
         for d in [data_a, data_b, data_c] if d and d.get("question")),
        ""
    )

    # Extraire la synthèse finale du LLM
    # Pattern tolérant : accepte ━━━, ---, ### ou toute ligne de séparation
    # suivi du numéro de section (optionnel) et du titre
    m_synth = re.search(
        r"(?:━+|[-]{3,}|#{1,3})[^\n]*(?:6\.?\s*)?SYNTHÈSE FINALE[^\n]*\n(.*?)(?=(?:━+|[-]{3,})\s*\Z|\Z)",
        raw, re.IGNORECASE | re.DOTALL
    )
    if not m_synth:
        # Fallback : chercher juste le titre sans séparateur
        m_synth = re.search(
            r"SYNTHÈSE FINALE[^\n]*\n(.*?)$",
            raw, re.IGNORECASE | re.DOTALL
        )
    synthese_finale = m_synth.group(1).strip() if m_synth else raw.strip()[-800:]

    lignes = []
    lignes += [
        "# Synthèse croisée — Toulmin · Perelman · RST",
        "",
        f"**Segment** : {titre}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {OPENAI_LLM_MODEL}  ",
    ]
    if question:
        lignes.append(f"**Question posée** : {question}  ")

    disponibles = []
    if data_a: disponibles.append("A (Toulmin)")
    if data_b: disponibles.append("B (Perelman)")
    if data_c: disponibles.append("C (RST)")
    lignes += [
        f"**Analyses intégrées** : {' · '.join(disponibles)}  ",
        "",
        "---",
        "",
        "## Synthèse finale",
        "",
    ]
    for l in synthese_finale.split("\n"):
        l = l.strip()
        if l:
            lignes.append(l)
    lignes += ["", "---", ""]

    # Tableau comparatif des scores
    lignes += [
        "## Tableau comparatif des scores",
        "",
        "| Dimension | Script A (Toulmin) | Script B (Perelman) | Script C (RST) |",
        "|---|---|---|---|",
    ]
    sa = (data_a or {}).get("scores", {})
    sb = (data_b or {}).get("scores", {})
    sc_rels = (data_c or {}).get("relations", [])
    sc_n_ev = sum(1 for r in sc_rels if r.get("relation") == "evidence")

    def fmt(v):
        return f"`{v:.2f}`" if isinstance(v, (int, float)) else "*N/D*"

    lignes += [
        f"| Solidité / profil  "
        f"| {fmt(sa.get('score_robustesse_globale','N/D'))} robustesse "
        f"| {fmt(sb.get('score_profil_argumentatif','N/D'))} profil "
        f"| {sc_n_ev} relation(s) evidence |",
        f"| Charge probatoire  "
        f"| {fmt(sa.get('score_charge_probatoire','N/D'))} "
        f"| {fmt(sb.get('score_ancrage_auditoire','N/D'))} ancrage "
        f"| — |",
        f"| Cohérence          "
        f"| {fmt(sa.get('score_coherence_warrant','N/D'))} warrant "
        f"| {fmt(sb.get('score_coherence_valeurs','N/D'))} valeurs "
        f"| noyau : `{(data_c or {}).get('noyau','?')}` |",
        f"| Risques            "
        f"| {fmt(sa.get('score_risque_sophisme','N/D'))} sophisme "
        f"| {fmt(sb.get('score_risque_sophistique_rhethorique','N/D'))} sophistique "
        f"| {sum(1 for r in sc_rels if r.get('confiance')=='basse')} rel. basse conf. |",
    ]
    lignes += ["", "---", ""]

    # Sections extraites du texte LLM
    def extraire_section_llm(raw: str, mots_cles: list[str], stop: list[str]) -> str:
        """
        Extrait une section de la réponse LLM de façon tolérante.

        Cherche une ligne contenant l'un des mots-clés (sans se préoccuper
        du type de séparateur — ━━━, ---, ###, **, numéro de section, etc.),
        puis capture tout le texte jusqu'à la prochaine section ou la fin.

        Args:
            raw      : Réponse LLM brute complète.
            mots_cles: Liste de termes qui identifient cette section.
            stop     : Liste de termes qui marquent la fin de la section.

        Returns:
            Contenu extrait et nettoyé, ou chaîne vide si non trouvé.
        """
        # Construire un pattern qui trouve la ligne d'en-tête
        mots_pat = "|".join(re.escape(m) for m in mots_cles)
        stop_pat = "|".join(re.escape(s) for s in stop) if stop else "ZZZNOMATCH"

        # Chercher la ligne contenant les mots-clés (avec ou sans séparateur)
        m = re.search(
            rf"^[^\n]*(?:{mots_pat})[^\n]*\n(.*?)(?=^[^\n]*(?:{stop_pat})|\Z)",
            raw, re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        if m:
            return m.group(1).strip()

        # Fallback : cherche simplement après le mot-clé sur n'importe quelle ligne
        for mot in mots_cles:
            idx = raw.lower().find(mot.lower())
            if idx != -1:
                # Avancer jusqu'à la fin de la ligne du titre
                fin_ligne = raw.find("\n", idx)
                if fin_ligne != -1:
                    suite = raw[fin_ligne + 1:].strip()
                    # Couper à la prochaine ligne de séparation
                    for stop_mot in stop:
                        idx_stop = suite.lower().find(stop_mot.lower())
                        if idx_stop != -1:
                            suite = suite[:idx_stop].strip()
                    return suite[:2000]  # limite de sécurité
        return ""

    SECTIONS_D = [
        ("Tableau de convergence",
         ["TABLEAU DE CONVERGENCE", "convergence"],
         ["ZONES DE DIVERGENCE", "divergence significative"]),
        ("Zones de divergence",
         ["ZONES DE DIVERGENCE", "divergence significative"],
         ["CE QUE LES TROIS", "trois cadres"]),
        ("Ce que les trois cadres révèlent",
         ["CE QUE LES TROIS", "trois cadres ensemble", "révèlent"],
         ["RECOMMANDATIONS", "recommandation"]),
        ("Recommandations",
         ["RECOMMANDATIONS POUR L", "recommandations"],
         ["RÉPONSE", "réponse à la question", "SYNTHÈSE FINALE"]),
    ]
    if question:
        SECTIONS_D.append(
            ("Réponse à la question",
             ["RÉPONSE À LA QUESTION", "réponse à"],
             ["SYNTHÈSE FINALE", "synthèse"])
        )

    for titre_section, mots_cles, stop in SECTIONS_D:
        contenu_section = extraire_section_llm(raw, mots_cles, stop)
        if not contenu_section:
            contenu_section = "*Non disponible — le LLM n'a pas produit cette section.*"
        lignes += [f"## {titre_section}", "", contenu_section, "", "---", ""]

    lignes += [
        "",
        "*Rapport généré par `D_synthese_croisee.py`.*  ",
        "*Cadres intégrés : Toulmin (1958) · Perelman (1958) · RST Mann & Thompson (1988).*",
    ]

    contenu = "\n".join(lignes)
    chemin  = output_dir / f"synthese_croisee_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="D_synthese_croisee — Synthèse intégrée A+B+C."
    )
    parser.add_argument("--toulmin",  type=str, default=None,
                        help="JSON produit par A_toulmin_segment.py.")
    parser.add_argument("--perelman", type=str, default=None,
                        help="JSON produit par B_perelman_segment.py.")
    parser.add_argument("--rst",      type=str, default=None,
                        help="JSON produit par C_rst_segment.py.")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  SYNTHÈSE CROISÉE — Toulmin · Perelman · RST")
    print("═"*60)

    # ── Résolution des chemins ────────────────────────────────────────────────
    chemins = {"A": None, "B": None, "C": None}

    if args.toulmin:
        chemins["A"] = Path(args.toulmin)
    if args.perelman:
        chemins["B"] = Path(args.perelman)
    if args.rst:
        chemins["C"] = Path(args.rst)

    # Détection automatique pour les non-fournis
    detectes = detecter_json_disponibles(output_dir)
    for lettre in ("A", "B", "C"):
        if chemins[lettre] is None:
            if detectes[lettre]:
                chemins[lettre] = detectes[lettre]
                print(f"  Détecté automatiquement [{lettre}] : {detectes[lettre].name}")
            else:
                patterns = {"A": "toulmin_*.json", "B": "perelman_seg_*.json", "C": "rst_*.json"}
                chemins[lettre] = demander_chemin(lettre, patterns[lettre], output_dir)

    # Chargement
    data = {}
    for lettre, chemin in chemins.items():
        if chemin and chemin.exists():
            data[lettre] = charger_json(chemin)
            print(f"  Chargé [{lettre}] : {chemin.name}")
        else:
            data[lettre] = None
            print(f"  Ignoré [{lettre}] : non disponible")

    if not any(data.values()):
        print("\n❌ Aucune analyse disponible. Lancez d'abord A, B ou C.")
        sys.exit(1)

    # Vérification cohérence
    avert = verifier_coherence(data["A"], data["B"], data["C"])
    if avert:
        print(f"\n{avert}")
        r = input("  Continuer quand même ? [o/N] : ").strip().lower()
        if r != "o":
            sys.exit(0)

    # Estimation coût
    prompt = construire_prompt(data["A"], data["B"], data["C"])
    tokens_in  = len(prompt) // 4
    tokens_out = MAX_TOKENS
    cout = (tokens_in * COUT_INPUT_PER_1K / 1000
            + tokens_out * COUT_OUTPUT_PER_1K / 1000)
    print(f"\n  Estimation : ~{tokens_in + tokens_out} tokens | ~{cout:.4f} $ ({OPENAI_LLM_MODEL})")
    if not args.no_confirm:
        r = input("  Lancer la synthèse ? [O/n] : ").strip().lower()
        if r == "n":
            print("  Annulé.")
            sys.exit(0)

    print(f"\n  Synthèse croisée en cours ({OPENAI_LLM_MODEL}, max_tokens={MAX_TOKENS})…")
    client = OpenAI()

    try:
        raw = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        ).choices[0].message.content
    except Exception as e:
        print(f"❌ Erreur API : {e}")
        sys.exit(1)

    if not raw:
        print("❌ Réponse vide.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md = generer_rapport_md(
        data["A"], data["B"], data["C"], raw, timestamp, output_dir
    )

    print(f"\n{'═'*60}")
    print(f"✅ Synthèse croisée générée.")
    print(f"   MD : {chemin_md}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
