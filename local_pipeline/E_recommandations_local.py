#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E_recommandations_local.py
==========================
Recommandations actionnables à partir de la synthèse croisée D.
VERSION 100 % LOCALE — Ollama / qwen2.5:14b.

RÔLE
─────
Ce script lit le rapport Markdown produit par D_synthese_croisee.py
(synthèse croisée Toulmin + Perelman + RST) et demande à qwen de
transformer cette synthèse en recommandations concrètes, hiérarchisées
et directement actionnables pour l'historien.

DIFFÉRENCES AVEC LA VERSION OPENAI
────────────────────────────────────
- Pas de clé API, pas de coût, fonctionne hors ligne.
- Prompt légèrement simplifié pour rester dans les capacités de qwen2.5:14b.
- MAX_TOKENS réduit à 2000 (qwen est plus lent et moins verbeux qu'OpenAI).
- Pas d'estimation de coût (gratuit).
- Résultats légèrement moins nuancés mais utilisables.

LIMITES EN LOCAL
─────────────────
- qwen peut avoir du mal à localiser précisément les phrases du texte
  original (il n'y a pas accès directement, seulement via la synthèse D).
- Les recommandations structurelles sont moins bien formulées qu'avec OpenAI.


UTILISATION
───────────
  ollama serve   # dans un terminal séparé
  python E_recommandations_local.py
  python E_recommandations_local.py --synthese outputs/synthese_croisee_X.md
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OLLAMA_URL   = "http://localhost:11434/api/generate"
LLM_MODEL    = "qwen2.5:14b"
TEMPERATURE  = 0.1        # basse : recommandations précises, pas créatives
MAX_TOKENS   = 2400       # suffisant pour 3-5 recommandations avec qwen
                           # augmenter si recommandations tronquées

OUTPUT_DIR   = "outputs"

# Longueur maximale de la synthèse D soumise à qwen (caractères).
# La synthèse D peut être longue — on la tronque pour rester dans
# la fenêtre de contexte de qwen (32k tokens, mais attention aux
# performances en fin de contexte).
MAX_SYNTHESE_CHARS = 6000

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import json
import argparse
import re
import requests
from datetime import datetime
from pathlib import Path

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique et argumentation académique. "
    "À partir d'une synthèse croisée (Toulmin, Perelman, RST), tu produis "
    "des recommandations concrètes et actionnables pour aider un historien "
    "à améliorer son texte. Tu cites toujours la phrase ou la section concernée. "
    "Tu distingues les révisions légères (reformulation) des révisions "
    "structurelles (déplacement, ajout, suppression). "
    "Tu réponds en français avec précision et concision."
)

USER_TEMPLATE = """Voici la synthèse croisée (Toulmin + Perelman + RST) d'un segment textuel :

{synthese}

Produis 3 à 5 recommandations actionnables pour l'historien, de la plus urgente
à la moins urgente. Pour chaque recommandation, utilise ce format exact :

RECOMMANDATION 1 — [HAUTE / MOYENNE / FAIBLE]
PROBLÈME : [Ce que la synthèse révèle comme point faible]
CADRE(S) : [A-Toulmin / B-Perelman / C-RST]
LOCALISATION : [Phrase ou section concernée]
NATURE : [légère — reformulation / structurelle — déplacement ou ajout]
SUGGESTION : [Proposition concrète en 2-3 phrases]

RECOMMANDATION 2 — [HAUTE / MOYENNE / FAIBLE]
[...]

CE QUI FONCTIONNE BIEN :
[1-2 points forts à ne pas modifier]"""

# =============================================================================
# CHARGEMENT
# =============================================================================

def trouver_derniere_synthese(output_dir: Path) -> Path | None:
    """Retourne le fichier synthese_croisee_*.md le plus récent."""
    fichiers = sorted(output_dir.glob("synthese_croisee_*.md"), reverse=True)
    return fichiers[0] if fichiers else None


def charger_synthese(chemin: Path, max_chars: int) -> str:
    """
    Charge le rapport Markdown de D et le tronque si nécessaire.

    La synthèse D peut être longue. On tronque à MAX_SYNTHESE_CHARS
    pour rester dans les capacités de qwen sans dégradation de qualité.
    Les sections les plus importantes (synthèse finale, divergences,
    recommandations de D) sont en début de fichier — la troncature
    en fin de document est donc acceptable.

    Args:
        chemin    : Chemin vers le fichier synthese_croisee_*.md.
        max_chars : Longueur maximale à conserver.

    Returns:
        Contenu nettoyé et tronqué si nécessaire.
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin.resolve()}")

    contenu = chemin.read_text(encoding="utf-8").strip()
    if not contenu:
        raise ValueError(f"Le fichier {chemin} est vide.")

    # Supprimer le pied de page
    contenu = re.sub(
        r"\*Rapport généré par.*$", "", contenu,
        flags=re.DOTALL | re.MULTILINE
    ).strip()

    # Troncature si nécessaire
    if len(contenu) > max_chars:
        contenu = contenu[:max_chars]
        contenu += "\n\n[... synthèse tronquée pour respecter la fenêtre de contexte ...]"
        print(f"  ⚠ Synthèse tronquée à {max_chars} caractères.")

    return contenu


# =============================================================================
# APPEL OLLAMA
# =============================================================================

def call_ollama(prompt: str) -> str:
    """
    Soumet le prompt à Ollama sans streaming.

    Sans streaming pour récupérer la réponse complète d'un coup
    et faciliter le parsing des recommandations.

    Args:
        prompt : Prompt complet (système + synthèse D + instructions).

    Returns:
        Réponse brute du LLM.

    Raises:
        SystemExit : Si Ollama n'est pas joignable.
    """
    payload = {
        "model"       : LLM_MODEL,
        "prompt"      : f"{SYSTEM_PROMPT}\n\n{prompt}",
        "temperature" : TEMPERATURE,
        "num_predict" : MAX_TOKENS,
        "stream"      : False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print(
            f"\n❌ Ollama inaccessible. Lancez : ollama serve\n"
            f"   URL : {OLLAMA_URL}"
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Timeout — qwen met trop de temps à répondre.")
        print("   Essayez de réduire MAX_SYNTHESE_CHARS ou MAX_TOKENS.")
        sys.exit(1)


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
    Génère recommandations_local_{timestamp}.md.

    Format identique à la version OpenAI pour comparabilité.

    Args:
        raw             : Réponse brute du LLM.
        titre_segment   : Titre du segment analysé.
        chemin_synthese : Nom du fichier synthèse D source.
        timestamp       : Horodatage du run.
        output_dir      : Dossier de sortie.

    Returns:
        Chemin du fichier Markdown produit.
    """
    lignes = [
        "# Recommandations actionnables (local)",
        "",
        f"**Segment** : {titre_segment}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {LLM_MODEL}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Source** : `{chemin_synthese}`  ",
        "",
        "> *Recommandations produites à partir de la synthèse croisée*",
        "> *Toulmin (A) · Perelman (B) · RST (C) produite par D_synthese_croisee.py*",
        "> *Version locale — qwen2.5:14b via Ollama*",
        "",
        "---",
        "",
    ]

    # Découper la réponse en recommandations
    # qwen utilise le format "RECOMMANDATION N —" sans les ━━━
    blocs = re.split(r"(?=RECOMMANDATION\s+\d|CE QUI FONCTIONNE)", raw, flags=re.IGNORECASE)

    for bloc in blocs:
        bloc = bloc.strip()
        if not bloc:
            continue

        if re.match(r"RECOMMANDATION\s+\d", bloc, re.IGNORECASE):
            premiere_ligne = bloc.split("\n")[0].strip()
            reste = "\n".join(bloc.split("\n")[1:]).strip()

            if "HAUTE" in premiere_ligne.upper():
                badge = "🔴"
            elif "MOYENNE" in premiere_ligne.upper():
                badge = "🟡"
            else:
                badge = "🟢"

            lignes += [f"## {badge} {premiere_ligne}", ""]

            # Formatter les champs
            champs = ["PROBLÈME", "CADRE", "LOCALISATION", "NATURE", "SUGGESTION"]
            for champ in champs:
                pattern = rf"{champ}(?:\(S\))?\s*:\s*(.+?)(?=\n(?:PROBLÈME|CADRE|LOCALISATION|NATURE|SUGGESTION|$))"
                m = re.search(pattern, reste, re.DOTALL | re.IGNORECASE)
                if m:
                    valeur = m.group(1).strip()
                    if champ == "SUGGESTION":
                        lignes += [f"**{champ}** :", ""]
                        for l in valeur.split("\n"):
                            l = l.strip()
                            if l:
                                lignes.append(f"> {l}")
                        lignes.append("")
                    else:
                        lignes.append(f"**{champ}** : {valeur}")

            lignes += ["", "---", ""]

        elif "FONCTIONNE BIEN" in bloc.upper():
            lignes += ["## ✅ Ce qui fonctionne bien", ""]
            for l in bloc.split("\n")[1:]:
                l = l.strip()
                if l:
                    lignes.append(l)
            lignes += ["", "---", ""]

        else:
            for l in bloc.split("\n"):
                l = l.strip()
                if l:
                    lignes.append(l)
            lignes.append("")

    lignes += [
        "",
        "*Rapport généré par `E_recommandations_local.py`.*  ",
        "*Cadres sources : Toulmin (1958) · Perelman (1958) · RST Mann & Thompson (1988).*  ",
        f"*Modèle local : {LLM_MODEL} via Ollama.*",
    ]

    contenu  = "\n".join(lignes)
    chemin   = output_dir / f"recommandations_local_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E_recommandations_local — Recommandations actionnables (Ollama/qwen)."
    )
    parser.add_argument(
        "--synthese", type=str, default=None, metavar="FICHIER",
        help="Fichier synthese_croisee_*.md produit par D. "
             "Si absent : prend le plus récent dans OUTPUT_DIR."
    )
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  E — RECOMMANDATIONS ACTIONNABLES (Local/qwen)")
    print("═"*60)
    print(f"  Modèle : {LLM_MODEL} | T° : {TEMPERATURE} | max_tokens : {MAX_TOKENS}")

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
        synthese = charger_synthese(chemin_synthese, MAX_SYNTHESE_CHARS)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"  Synthèse chargée : {len(synthese)} caractères")

    # Extraire le titre du segment
    m_titre = re.search(r"\*\*Segment\*\*\s*:\s*(.+?)  ", synthese)
    titre_segment = m_titre.group(1).strip() if m_titre else "Segment sans titre"
    print(f"  Segment : {titre_segment[:60]}\n")

    # ── Appel Ollama ──────────────────────────────────────────────────────────
    print(f"  Génération des recommandations en cours…")
    print(f"  (peut prendre 1-3 minutes selon la longueur de la synthèse)\n")

    prompt = USER_TEMPLATE.format(synthese=synthese)
    raw    = call_ollama(prompt)

    if not raw:
        print("❌ Réponse vide de qwen.")
        sys.exit(1)

    # ── Rapport Markdown ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md = generer_rapport_md(
        raw, titre_segment, chemin_synthese.name, timestamp, output_dir
    )

    # ── Bilan console ─────────────────────────────────────────────────────────
    nb_reco = len(re.findall(r"RECOMMANDATION\s+\d", raw, re.IGNORECASE))

    print(f"\n{'═'*60}")
    print(f"✅ {nb_reco} recommandation(s) générée(s).")
    print(f"   MD : {chemin_md}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
