#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E_recommandations_openai.py
===========================
Recommandations actionnables à partir de la synthèse croisée D.

RÔLE
─────
Ce script lit le rapport Markdown produit par D_synthese_croisee.py
(synthèse croisée Toulmin + Perelman + RST) et demande au LLM de
transformer cette synthèse en recommandations concrètes, hiérarchisées
et directement actionnables pour l'historien.

DIFFÉRENCE AVEC D
──────────────────
D produit une synthèse analytique croisée — un portrait argumentatif
du segment. Ce script va plus loin : il traduit ce portrait en actions
de révision concrètes, phrase par phrase ou section par section.

FORMAT DES RECOMMANDATIONS
────────────────────────────
Pour chaque recommandation :
  - Priorité : HAUTE / MOYENNE / FAIBLE
  - Problème identifié (d'après A, B ou C)
  - Localisation dans le texte (phrase ou section concernée)
  - Nature de la révision : légère (reformulation) | structurelle (déplacement, ajout)
  - Suggestion concrète de révision

ESTIMATION DES COÛTS
─────────────────────
Entrée : synthèse D (~1500-2000 tokens)
Sortie : recommandations (~1000-1500 tokens)
gpt-4.1-mini : ~0.002-0.003 $ par analyse

UTILISATION
───────────
  python E_recommandations_openai.py
  python E_recommandations_openai.py --synthese resultats/synthese_croisee_X.md
  python E_recommandations_openai.py --no-confirm
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OPENAI_LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE       = 0.1       # basse : recommandations précises, pas créatives
MAX_TOKENS        = 3500      # suffisant pour 5-7 recommandations détaillées
                               # augmenter si recommandations tronquées (max : 4096)

OUTPUT_DIR        = "resultats"

COUT_INPUT_PER_1K  = 0.00040   # gpt-4.1-mini input
COUT_OUTPUT_PER_1K = 0.00160   # gpt-4.1-mini output

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import argparse
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique et argumentation académique, spécialisé dans "
    "l'accompagnement des historiens dans la révision de leurs textes. "
    "À partir d'une synthèse croisée (Toulmin, Perelman, RST), tu produis des "
    "recommandations concrètes, hiérarchisées et actionnables. "
    "Tu distingues les révisions légères (reformulation d'une phrase) des révisions "
    "structurelles (déplacement, ajout, suppression). "
    "Tu cites toujours la phrase ou la section concernée. "
    "Tu réponds en français avec précision et concision."
)

USER_TEMPLATE = """Voici la synthèse croisée (Toulmin + Perelman + RST) d'un segment textuel :

{synthese}

À partir de cette synthèse, produis des recommandations actionnables pour l'historien.

FORMAT STRICT — pour chaque recommandation :

━━━ RECOMMANDATION N — [HAUTE / MOYENNE / FAIBLE] ━━━
PROBLÈME : [Ce que la synthèse croisée révèle comme point faible ou opportunité]
CADRE(S) : [A-Toulmin / B-Perelman / C-RST — quel(s) cadre(s) signale(nt) ce problème]
LOCALISATION : [Phrase ou section concernée dans le texte original]
NATURE : [légère — reformulation / structurelle — déplacement ou ajout]
SUGGESTION : [Proposition concrète de révision en 2-3 phrases. Si possible,
              proposer une reformulation directement utilisable.]

Produis 3 à 5 recommandations, de la plus urgente à la moins urgente.
Termine par :

━━━ CE QUI FONCTIONNE BIEN ━━━
[1-2 points forts à ne pas modifier, identifiés par la synthèse croisée]"""

# =============================================================================
# CHARGEMENT
# =============================================================================

def trouver_derniere_synthese(output_dir: Path) -> Path | None:
    """Retourne le fichier synthese_croisee_*.md le plus récent."""
    fichiers = sorted(output_dir.glob("synthese_croisee_*.md"), reverse=True)
    return fichiers[0] if fichiers else None


def charger_synthese(chemin: Path) -> str:
    """
    Charge le rapport Markdown de D et en extrait le contenu pertinent.

    On garde la synthèse finale, le tableau des scores et les sections
    analytiques — on supprime les en-têtes administratifs pour économiser
    des tokens.

    Args:
        chemin : Chemin vers le fichier synthese_croisee_*.md.

    Returns:
        Contenu nettoyé, prêt à être inséré dans le prompt.

    Raises:
        FileNotFoundError : Si le fichier est introuvable.
        ValueError        : Si le fichier est vide.
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin.resolve()}")

    contenu = chemin.read_text(encoding="utf-8").strip()
    if not contenu:
        raise ValueError(f"Le fichier {chemin} est vide.")

    # Supprimer le pied de page (économie de tokens)
    contenu = re.sub(
        r"\*Rapport généré par.*$", "", contenu,
        flags=re.DOTALL | re.MULTILINE
    ).strip()

    return contenu


# =============================================================================
# ESTIMATION DU COÛT
# =============================================================================

def estimer_cout(synthese: str) -> tuple[int, float]:
    tokens_in  = (len(synthese) + len(USER_TEMPLATE) + len(SYSTEM_PROMPT)) // 4
    tokens_out = MAX_TOKENS
    cout = (tokens_in  * COUT_INPUT_PER_1K  / 1000
          + tokens_out * COUT_OUTPUT_PER_1K / 1000)
    return tokens_in + tokens_out, cout


def confirmer_cout(tokens: int, cout: float, no_confirm: bool) -> None:
    print(f"\n  Estimation : ~{tokens} tokens | ~{cout:.4f} $ ({OPENAI_LLM_MODEL})")
    if not no_confirm:
        r = input("  Lancer l'analyse ? [O/n] : ").strip().lower()
        if r == "n":
            print("  Annulé.")
            sys.exit(0)


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def generer_rapport_md(
    raw: str,
    titre_segment: str,
    chemin_synthese: str,
    timestamp: str,
    output_dir: Path,
) -> Path:
    """
    Génère recommandations_{timestamp}.md — rapport de recommandations actionnables.

    Args:
        raw             : Réponse brute du LLM.
        titre_segment   : Titre du segment analysé (extrait du MD de D).
        chemin_synthese : Chemin du fichier synthèse D source.
        timestamp       : Horodatage du run.
        output_dir      : Dossier de sortie.

    Returns:
        Chemin du fichier Markdown produit.
    """
    lignes = [
        "# Recommandations actionnables",
        "",
        f"**Segment** : {titre_segment}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {OPENAI_LLM_MODEL}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Source** : `{chemin_synthese}`  ",
        "",
        "> *Recommandations produites à partir de la synthèse croisée*",
        "> *Toulmin (A) · Perelman (B) · RST (C) produite par D_synthese_croisee.py*",
        "",
        "---",
        "",
    ]

    # Le LLM produit directement des sections ━━━ RECOMMANDATION N ━━━
    # On les reformate en Markdown propre
    sections = re.split(r"━+\s*", raw)
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # En-tête de recommandation
        if section.upper().startswith("RECOMMANDATION"):
            # Extraire le titre (ex: "RECOMMANDATION 1 — HAUTE")
            premiere_ligne = section.split("\n")[0].strip()
            reste = "\n".join(section.split("\n")[1:]).strip()

            # Déterminer la priorité pour le badge
            if "HAUTE" in premiere_ligne.upper():
                badge = "🔴 HAUTE"
            elif "MOYENNE" in premiere_ligne.upper():
                badge = "🟡 MOYENNE"
            else:
                badge = "🟢 FAIBLE"

            lignes += [f"## {premiere_ligne}", ""]

            # Formatter les champs PROBLÈME, CADRE, LOCALISATION, NATURE, SUGGESTION
            for champ in ["PROBLÈME", "CADRE(S)", "LOCALISATION", "NATURE", "SUGGESTION"]:
                pattern = rf"{champ}\s*[:(]\s*(.+?)(?=\n(?:PROBLÈME|CADRE|LOCALISATION|NATURE|SUGGESTION|$))"
                m = re.search(pattern, reste, re.DOTALL | re.IGNORECASE)
                if m:
                    valeur = m.group(1).strip()
                    if champ == "SUGGESTION":
                        lignes += [f"**{champ}**", ""]
                        for l in valeur.split("\n"):
                            l = l.strip()
                            if l:
                                lignes.append(f"> {l}")
                        lignes.append("")
                    else:
                        lignes.append(f"**{champ}** : {valeur}")

            lignes += ["", "---", ""]

        elif "FONCTIONNE BIEN" in section.upper():
            lignes += ["## ✅ Ce qui fonctionne bien", ""]
            for l in section.split("\n")[1:]:
                l = l.strip()
                if l:
                    lignes.append(l)
            lignes += ["", "---", ""]

        else:
            # Contenu libre (intro ou outro du LLM)
            for l in section.split("\n"):
                l = l.strip()
                if l:
                    lignes.append(l)
            lignes.append("")

    lignes += [
        "",
        "*Rapport généré par `E_recommandations_openai.py`.*  ",
        "*Cadres sources : Toulmin (1958) · Perelman (1958) · RST Mann & Thompson (1988).*",
    ]

    contenu  = "\n".join(lignes)
    chemin   = output_dir / f"recommandations_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E_recommandations_openai — Recommandations actionnables depuis la synthèse D."
    )
    parser.add_argument(
        "--synthese", type=str, default=None, metavar="FICHIER",
        help="Fichier synthese_croisee_*.md produit par D. "
             "Si absent : prend le plus récent dans OUTPUT_DIR."
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Ne pas demander confirmation du coût."
    )
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  E — RECOMMANDATIONS ACTIONNABLES (OpenAI)")
    print("═"*60)

    # ── Chargement de la synthèse D ───────────────────────────────────────────
    if args.synthese:
        chemin_synthese = Path(args.synthese)
    else:
        chemin_synthese = trouver_derniere_synthese(output_dir)
        if not chemin_synthese:
            print(f"❌ Aucun fichier synthese_croisee_*.md dans {output_dir}.")
            print("   Lancez d'abord D_synthese_croisee.py.")
            sys.exit(1)
        print(f"  Détecté automatiquement : {chemin_synthese.name}")

    try:
        synthese = charger_synthese(chemin_synthese)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"  Synthèse chargée : {len(synthese)} caractères")

    # Extraire le titre du segment depuis le MD de D
    m_titre = re.search(r"\*\*Segment\*\*\s*:\s*(.+?)  ", synthese)
    titre_segment = m_titre.group(1).strip() if m_titre else "Segment sans titre"
    print(f"  Segment : {titre_segment[:60]}")

    # ── Estimation du coût ────────────────────────────────────────────────────
    tokens, cout = estimer_cout(synthese)
    confirmer_cout(tokens, cout, args.no_confirm)

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    print(f"\n  Génération des recommandations ({OPENAI_LLM_MODEL}, "
          f"T°={TEMPERATURE}, max_tokens={MAX_TOKENS})…")

    client = OpenAI()
    prompt = USER_TEMPLATE.format(synthese=synthese)

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

    # ── Rapport Markdown ──────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md  = generer_rapport_md(
        raw, titre_segment, chemin_synthese.name, timestamp, output_dir
    )

    # ── Bilan console ─────────────────────────────────────────────────────────
    # Compter les recommandations produites
    nb_reco = len(re.findall(r"RECOMMANDATION\s+\d", raw, re.IGNORECASE))

    print(f"\n{'═'*60}")
    print(f"✅ {nb_reco} recommandation(s) générée(s).")
    print(f"   MD : {chemin_md}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
