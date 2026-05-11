#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_toulmin_segment.py
====================
Analyse argumentative approfondie (Toulmin / Adam / Walton) sur un segment
textuel délimité par l'historien.

CE SCRIPT EST DIFFÉRENT DU 09
──────────────────────────────
Le script 09 analyse le manuscrit entier paragraphe par paragraphe en mode
batch. Ses scores individuels sont des signaux d'orientation, pas des métriques
fiables, en raison de la taille de contexte (un paragraphe = trop court pour
que Toulmin soit applicable rigoureusement) et de la polyphonie du texte
historique (discours rapporté vs discours de l'auteur).

Ce script-ci opère différemment :
  - L'historien délimite lui-même le segment à analyser (quelques paragraphes,
    une section, un développement complet).
  - Il fournit un titre et une question facultative qui cadrent l'analyse.
  - Le LLM dispose d'une fenêtre cohérente et d'un contexte intentionnel.
  - L'analyse Toulmin/Adam/Walton est menée sur cet ensemble, pas atome par atome.

C'est sur ce script que repose la responsabilité de l'analyse argumentative
rigoureuse. Le 09 désigne les zones ; ce script les analyse.

CADRES THÉORIQUES
─────────────────
  TOULMIN (1958) — The Uses of Argument, Cambridge UP.
    Six composantes : claim | grounds | warrant | backing | qualifier | rebuttal
    Limite assumée : conçu pour des arguments en première personne.
    Pour le texte historique, le warrant est souvent implicite et le rebuttal
    absent. C'est une information, pas un défaut.

  ADAM (1992) — Les textes : types et prototypes, Nathan.
    Quatre moments de la séquence argumentative :
    thèse antérieure | données/argument | conclusion/thèse | restriction
    Avantage sur Toulmin : intègre la thèse antérieure, ce qui correspond
    au mode de travail de l'historien (toujours en dialogue avec une
    historiographie existante).

  WALTON (1996) — Argumentation Schemes for Presumptive Reasoning, Erlbaum.
    Schèmes légitimes : autorité | analogie | causal | pragmatique
    Schèmes fallacieux : pente_glissante | généralisation | ad_hominem |
                         homme_de_paille
    Note : un schème "fallacieux" n'est pas nécessairement un défaut dans un
    texte historique — il peut être une stratégie rhétorique consciente.
    Le script le signale simplement.

LIMITES 
────────────────────────────────
A POLYPHONIE 
    Le texte historique rapporte, cite et reconstruit des argumentaires qui ne
sont pas ceux de l'auteur. "Victor Hugo défend Hartmann" : le claim analysé
est-il la position de l'auteur ou la position rapportée ? Le LLM peut
confondre les deux. Fournir un titre précis ("Analyse de la stratégie
rhétorique de Hugo dans la pétition de 1880") aide à cadrer.
B TAILLE 
    Un LLm commercial aujourd'hui (en avril 2026) 
peut mener à bien ce travail sur des corpus d'assez grande taille, 
mais au delà de 1500 à 2000 mots il est très probable que la qualité 
de l'analyse se dégradera sensiblement.



ESTIMATION DES COÛTS
─────────────────────
Un segment de 5-8 paragraphes (~1500 mots) :
  Tokens entrée  : ~2500  (texte + prompt système + passages corpus optionnels)
  Tokens sortie  : ~2000
  gpt-4.1-mini   : ~0.002 $ par analyse
  gpt-4.1        : ~0.016 $ par analyse
Le script affiche une estimation avant de lancer l'appel API.

UTILISATION
───────────
  python A_toulmin_segment.py
  python A_toulmin_segment.py --fichier mon_passage.txt
  python A_toulmin_segment.py --no-confirm   (sans demande de confirmation coût)
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OPENAI_LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE       = 0.1       # basse : on veut de la cohérence, pas de la créativité
MAX_TOKENS        = 4500      # tokens max en sortie
# le nombre de token est calculé de façon à optimiser coût et qualité de la réponse
# lorsque le LLM a consommé ces tokens il arrête l'analyse
# si le passage fourni est long il est possible que la réponse soit tronquée
# il faut alors augmenter le nombre de tokens alloués. 

OUTPUT_DIR        = "resultats"

# Coûts indicatifs en $ par 1000 tokens (input / output) — à mettre à jour
# selon la grille tarifaire OpenAI en vigueur
COUT_INPUT_PER_1K  = 0.00040   # gpt-4.1-mini input
COUT_OUTPUT_PER_1K = 0.00160   # gpt-4.1-mini output

# Longueur minimale d'un segment pour que l'analyse soit pertinente (caractères)
MIN_SEGMENT_CHARS = 300

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
from config_00 import OPENAI_LLM_MODEL as _MODEL_CONFIG  # fallback si besoin

load_dotenv()

# =============================================================================
# DONNÉES DE RÉFÉRENCE
# =============================================================================

SCHEMES_WALTON = {
    "autorité"        : "Argument from Expert Opinion",
    "analogie"        : "Argument from Analogy",
    "causal"          : "Argument from Cause to Effect",
    "pragmatique"     : "Pragmatic Argument",
    "pente_glissante" : "Slippery Slope",
    "generalisation"  : "Hasty Generalization",
    "ad_hominem"      : "Ad Hominem",
    "homme_de_paille" : "Straw Man",
}
SCHEMES_FALLACIEUX = {"pente_glissante", "generalisation", "ad_hominem", "homme_de_paille"}

SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique et logique argumentative, spécialisé dans "
    "l'analyse des textes académiques et historiques. Tu maîtrises les cadres "
    "théoriques de Toulmin, Adam et Walton. Tu travailles sur des segments "
    "textuels délimités par l'historien — pas sur des paragraphes isolés. "
    "Tu distingues le discours de l'auteur du discours rapporté. "
    "Tu réponds en français avec précision et rigueur."
)

# =============================================================================
# SAISIE INTERACTIVE
# =============================================================================

def saisir_segment() -> tuple[str, str, str]:
    """
    Guide l'utilisateur dans la saisie du segment à analyser.

    Retourne :
        titre   : Titre ou description du segment (obligatoire)
        texte   : Texte du segment
        question: Question facultative de l'historien
    """
    print("\n" + "═"*60)
    print("  ANALYSE ARGUMENTATIVE SUR SEGMENT (Toulmin / Adam / Walton)")
    print("═"*60)
    print()
    print("  Ce script analyse un segment textuel délimité par vous.")
    print("  Un segment = quelques paragraphes à une section entière.")
    print("  Fournissez un titre précis pour cadrer l'analyse.")
    print()

    # ── Titre (obligatoire) ───────────────────────────────────────────────────
    print("─"*60)
    print("  TITRE DU SEGMENT (obligatoire)")
    print("  Exemples :")
    print("    'Argument central du chapitre 3 sur l'extradition politique'")
    print("    'Stratégie rhétorique de Victor Hugo dans la pétition de 1880'")
    print("    'Section sur la politisation des migrants 1920-1930'")
    print()
    titre = ""
    while not titre.strip():
        titre = input("  Titre : ").strip()
        if not titre:
            print("  ⚠ Le titre est obligatoire.")

    # ── Texte (fichier ou saisie) ─────────────────────────────────────────────
    print()
    print("─"*60)
    print("  SEGMENT À ANALYSER")
    print("  [1] Coller le texte directement dans la console")
    print("  [2] Indiquer un fichier .txt")
    print()
    choix = ""
    while choix not in ("1", "2"):
        choix = input("  Choix [1/2] : ").strip()

    if choix == "2":
        chemin = input("  Chemin du fichier .txt : ").strip()
        p = Path(chemin)
        if not p.exists():
            print(f"  ❌ Fichier introuvable : {p.resolve()}")
            sys.exit(1)
        texte = p.read_text(encoding="utf-8").strip()
        print(f"  ✔ {len(texte)} caractères chargés depuis {p.name}")
    else:
        print()
        print("  Collez le texte puis appuyez sur Entrée,")
        print("  puis tapez '###FIN###' sur une ligne seule et appuyez sur Entrée.")
        print()
        lignes = []
        while True:
            ligne = input()
            if ligne.strip() == "###FIN###":
                break
            lignes.append(ligne)
        texte = "\n".join(lignes).strip()
        print(f"  ✔ {len(texte)} caractères saisis.")

    if len(texte) < MIN_SEGMENT_CHARS:
        print(f"\n  ⚠ Segment trop court ({len(texte)} caractères).")
        print(f"    Minimum recommandé : {MIN_SEGMENT_CHARS} caractères.")
        r = input("  Continuer quand même ? [o/N] : ").strip().lower()
        if r != "o":
            sys.exit(0)

    # ── Question facultative ──────────────────────────────────────────────────
    print()
    print("─"*60)
    print("  QUESTION FACULTATIVE (appuyez sur Entrée pour ignorer)")
    print("  Exemples :")
    print("    'Mon argument tient-il sans les sources primaires ?'")
    print("    'Est-ce que je distingue bien ma thèse du discours rapporté ?'")
    print("    'Quelles sont les faiblesses probatoires de ce passage ?'")
    print()
    question = input("  Question : ").strip()

    return titre, texte, question


# =============================================================================
# ESTIMATION DES COÛTS
# =============================================================================

def estimer_cout(texte: str, titre: str, question: str) -> tuple[int, float]:
    """
    Estime le nombre de tokens et le coût avant l'appel API.

    Méthode approximative : 1 token ≈ 4 caractères (règle heuristique
    pour le français). La vraie valeur dépend du tokenizer OpenAI.
    L'estimation est volontairement légèrement supérieure à la réalité.
    """
    prompt_sys_chars  = len(SYSTEM_PROMPT)
    prompt_user_chars = len(texte) + len(titre) + len(question) + 500  # overhead prompt
    tokens_input_est  = (prompt_sys_chars + prompt_user_chars) // 4
    tokens_output_est = MAX_TOKENS
    cout = (tokens_input_est * COUT_INPUT_PER_1K / 1000
            + tokens_output_est * COUT_OUTPUT_PER_1K / 1000)
    return tokens_input_est + tokens_output_est, round(cout, 4)


def confirmer_cout(tokens: int, cout: float, no_confirm: bool) -> None:
    """Affiche l'estimation et demande confirmation."""
    print()
    print(f"  Estimation : ~{tokens} tokens | ~{cout:.4f} $ ({OPENAI_LLM_MODEL})")
    if no_confirm:
        return
    r = input("  Lancer l'analyse ? [O/n] : ").strip().lower()
    if r == "n":
        print("  Annulé.")
        sys.exit(0)


# =============================================================================
# PROMPT
# =============================================================================

def construire_prompt(titre: str, texte: str, question: str) -> str:
    question_bloc = (
        f"\n══════════════════════════════════════════\n"
        f"QUESTION DE L'HISTORIEN\n"
        f"══════════════════════════════════════════\n"
        f"{question}\n"
    ) if question else ""

    return f"""Analyse le segment suivant selon les cadres Toulmin, Adam et Walton.
Ce segment a été délimité par l'historien — c'est une unité argumentative cohérente,
pas un paragraphe isolé.

══════════════════════════════════════════
SEGMENT : {titre}
══════════════════════════════════════════
{texte}
{question_bloc}
══════════════════════════════════════════
ANALYSE DEMANDÉE
══════════════════════════════════════════

━━━ 1. DISCOURS DE L'AUTEUR VS DISCOURS RAPPORTÉ ━━━
Identifie les passages où l'auteur rapporte ou reconstruit des argumentaires
qui ne sont pas les siens (citations, résumés de positions adverses, discours
d'époque). Ces passages sont exclus de l'analyse Toulmin.
DISCOURS_RAPPORTE: <liste ou AUCUN>

━━━ 2. ANALYSE TOULMIN (sur le discours de l'auteur uniquement) ━━━
CLAIM     : thèse ou conclusion principale de l'auteur
GROUNDS   : données, faits, sources invoquées par l'auteur
WARRANT   : règle implicite ou explicite autorisant le passage GROUNDS→CLAIM
BACKING   : autorité ou preuve légitimant le WARRANT
QUALIFIER : nuances ou modalités de la CLAIM
REBUTTAL  : cas d'exception ou objections anticipées par l'auteur
(Si une composante est absente : indiquer ABSENT et expliquer en une phrase
pourquoi son absence est notable ou normale dans ce type de texte.)

━━━ 3. SÉQUENCE ADAM ━━━
THÈSE ANTÉRIEURE : position historiographique de départ ou thèse adverse
DONNÉES          : arguments et preuves mobilisés dans le segment
CONCLUSION/THÈSE : nouvelle thèse défendue par l'auteur
RESTRICTION      : concessions, limites, nuances explicites

━━━ 4. SCHÈMES WALTON ━━━
Schèmes présents parmi :
  Légitimes  : autorité | analogie | causal | pragmatique
  Fallacieux : pente_glissante | generalisation | ad_hominem | homme_de_paille
Pour chaque schème : cite le passage exact du segment qui l'illustre.
Note : un schème "fallacieux" peut être une stratégie rhétorique consciente —
signale-le sans le condamner si c'est le cas.
SCHEMA_DOMINANT: <schème>
SOPHISMES: <schème1>, <schème2>  (ou AUCUN)

━━━ 5. ANALYSE DE LA POLYPHONIE ━━━
L'auteur distingue-t-il clairement sa voix de celle qu'il rapporte ?
Y a-t-il des passages ambigus où cette distinction est floue ?
Comment cela affecte-t-il la structure argumentative d'ensemble ?

━━━ 6. SCORES (format strict) ━━━
SCORE_COMPLETUDE_TOULMIN: X/10
SCORE_COHERENCE_WARRANT: X/10
SCORE_RISQUE_SOPHISME: X/10
SCORE_CHARGE_PROBATOIRE: X/10
SCORE_ROBUSTESSE_GLOBALE: X/10

━━━ 7. SYNTHÈSE ━━━
En 4-5 phrases : évaluation argumentative globale du segment, point fort,
point faible principal, et — si une question a été posée — réponse directe
à cette question."""


# =============================================================================
# EXTRACTION
# =============================================================================

NOMS_COURTS_SCORES = {
    "score_completude_toulmin": ["completude", "complétude", "completude_toulmin"],
    "score_coherence_warrant" : ["warrant", "coherence_warrant", "cohérence_warrant"],
    "score_risque_sophisme"   : ["sophisme", "risque_sophisme", "risque"],
    "score_charge_probatoire" : ["probatoire", "charge_probatoire", "charge"],
    "score_robustesse_globale": ["robustesse", "robustesse_globale"],
}

PATTERNS_SCORES = {
    "score_completude_toulmin": r"SCORE_COMPLETUDE_TOULMIN\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_coherence_warrant" : r"SCORE_COHERENCE_WARRANT\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_risque_sophisme"   : r"SCORE_RISQUE_SOPHISME\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_charge_probatoire" : r"SCORE_CHARGE_PROBATOIRE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_robustesse_globale": r"SCORE_ROBUSTESSE_GLOBALE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
}

VALEUR_NEUTRE = 0.5


def _normaliser_score(v: float, avec_denominateur: bool) -> float:
    """
    Normalise une valeur brute LLM vers [0, 1].

    avec_denominateur=True  : valeur lue comme N/10 → diviser par 10.
    avec_denominateur=False : ambigu — règle :
        - entier (1, 8, 10…) → sur 10, diviser par 10.
        - décimal > 1 (8.1, 7.5…) → sur 10, diviser par 10.
        - décimal ≤ 1 (0.8, 0.75…) → déjà normalisé, garder tel quel.
    Cette règle est conservatrice : un LLM qui écrit "1" sans /10 voulait
    presque certainement dire "1/10", pas "score parfait de 1.0".
    """
    if avec_denominateur or v == int(v) or v > 1.0:
        return round(max(0.0, min(10.0, v)) / 10.0, 3)
    return round(max(0.0, min(1.0, v)), 3)


def extraire_scores(texte_llm: str) -> dict:
    """
    Extrait les cinq scores Toulmin depuis la réponse LLM brute.

    Quatre niveaux de robustesse, du plus strict au plus permissif :

    Niveau 1 — Format strict : SCORE_X: N/10
        Accepte point ou virgule comme séparateur décimal.
        Ex : SCORE_COMPLETUDE_TOULMIN: 8/10  |  SCORE_RISQUE_SOPHISME: 7,5/10

    Niveau 2 — Espaces autour du / : SCORE_X: N / 10
        Ex : SCORE_COHERENCE_WARRANT: 9 / 10

    Niveau 3 — Nom court sans préfixe SCORE_
        Le LLM peut écrire "complétude : 8" ou "robustesse = 7.5"
        sans /10. Normalisation : entier ou décimal > 1 → sur 10 ;
        décimal ≤ 1 → déjà normalisé.

    Niveau 4 — Tableau Markdown
        | label | valeur | — même règle de normalisation que le niveau 3.

    Si aucun niveau ne trouve le score : retourne VALEUR_NEUTRE (0.5)
    avec un avertissement stderr. 0.5 est une valeur neutre explicite,
    pas un score — à ne pas interpréter comme "argument moyen".
    """
    scores = {}

    for nom, pattern in PATTERNS_SCORES.items():
        # Niveau 1 + 2 — format strict (avec /10 explicite)
        m = re.search(pattern, texte_llm, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", "."))
            scores[nom] = _normaliser_score(v, avec_denominateur=True)
            continue

        # Niveau 3 — nom court, avec ou sans /10
        trouve = False
        for nom_court in NOMS_COURTS_SCORES.get(nom, []):
            pat3 = (rf"(?:^|[\s|])(?:{re.escape(nom_court)})"
                    rf"\s*[:=]\s*(\d+(?:[.,]\d+)?)\s*(?:/\s*10)?")
            m3 = re.search(pat3, texte_llm, re.IGNORECASE | re.MULTILINE)
            if m3:
                v = float(m3.group(1).replace(",", "."))
                avec_denom = bool(re.search(
                    r"/\s*10", texte_llm[m3.start():m3.end() + 10]
                ))
                scores[nom] = _normaliser_score(v, avec_denominateur=avec_denom)
                trouve = True
                break

        if trouve:
            continue

        # Niveau 4 — tableau Markdown
        for nom_court in NOMS_COURTS_SCORES.get(nom, []):
            pat4 = (rf"\|\s*[^|]*{re.escape(nom_court)}[^|]*\|"
                    rf"\s*(\d+(?:[.,]\d+)?)\s*\|")
            m4 = re.search(pat4, texte_llm, re.IGNORECASE)
            if m4:
                v = float(m4.group(1).replace(",", "."))
                scores[nom] = _normaliser_score(v, avec_denominateur=False)
                break
        else:
            print(f"  ⚠ Score non trouvé : {nom} → {VALEUR_NEUTRE}",
                  file=sys.stderr)
            scores[nom] = VALEUR_NEUTRE

    return scores


def extraire_composante(raw: str, comp: str) -> str:
    """Extrait une composante Toulmin en s'arrêtant à la suivante."""
    suivantes = ["CLAIM","GROUNDS","WARRANT","BACKING","QUALIFIER","REBUTTAL",
                 "THÈSE","DONNÉES","CONCLUSION","RESTRICTION","SCHEMA","SCORE","SYNTHÈSE"]
    suivantes_pat = "|".join(s for s in suivantes if s != comp)
    m = re.search(
        rf"{comp}\s*:\s*(.+?)(?=\n\s*(?:{suivantes_pat})|━|$)",
        raw, re.IGNORECASE | re.DOTALL
    )
    if m:
        val = m.group(1).strip()
        return val if val.upper() != "ABSENT" else "ABSENT"
    return "ABSENT"


def extraire_section(raw: str, marqueur: str, suivant: str = "━") -> str:
    """Extrait un bloc de texte entre deux marqueurs."""
    m = re.search(
        rf"━+\s*{re.escape(marqueur)}.*?━+\n(.*?)(?=━+\s*\d|\Z)",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else ""


def extraire_synthese(raw: str) -> str:
    m = re.search(
        r"(?:SYNTHÈSE|synthèse)[^\n]*\n(.+?)$",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else raw.strip()[-800:]


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def _barre(v: float, w: int = 20) -> str:
    r = round(v * w)
    return f"[{'█'*r}{'░'*(w-r)}] {v:.2f}"


def generer_rapport_md(
    titre: str,
    texte: str,
    question: str,
    raw: str,
    scores: dict,
    timestamp: str,
    output_dir: Path,
) -> Path:
    """Génère toulmin_{timestamp}.md — rapport lisible de l'analyse."""

    comp = {k: extraire_composante(raw, k.upper())
            for k in ["claim","grounds","warrant","backing","qualifier","rebuttal"]}

    adam = {}
    for cle, pat in [
        ("these_anterieure", r"THÈSE ANTÉRIEURE\s*:\s*(.+?)(?=\n\s*DONNÉES|\n\s*CONCLUSION|\n\s*RESTRICTION|━|$)"),
        ("donnees",          r"DONNÉES\s*:\s*(.+?)(?=\n\s*CONCLUSION|\n\s*RESTRICTION|━|$)"),
        ("conclusion",       r"CONCLUSION(?:/THÈSE)?\s*:\s*(.+?)(?=\n\s*RESTRICTION|━|$)"),
        ("restriction",      r"RESTRICTION\s*:\s*(.+?)(?=━|$)"),
    ]:
        m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
        adam[cle] = m.group(1).strip() if m else "non identifié"

    m_schema = re.search(r"SCHEMA_DOMINANT\s*:\s*(\w+)", raw, re.IGNORECASE)
    schema = m_schema.group(1).lower() if m_schema else "aucun"
    if schema not in SCHEMES_WALTON:
        schema = "aucun"

    m_soph = re.search(r"SOPHISMES\s*:\s*(.+)", raw, re.IGNORECASE)
    sophismes = []
    if m_soph and m_soph.group(1).strip().upper() != "AUCUN":
        sophismes = [s.strip().lower() for s in re.split(r"[,;]", m_soph.group(1))
                     if s.strip().lower() in SCHEMES_FALLACIEUX]

    m_disc = re.search(r"DISCOURS_RAPPORTE\s*:\s*(.+?)(?=━|$)", raw, re.IGNORECASE | re.DOTALL)
    discours_rapporte = m_disc.group(1).strip() if m_disc else "non identifié"

    synthese = extraire_synthese(raw)

    lignes = []
    lignes += [
        "# Analyse argumentative — Toulmin / Adam / Walton",
        "",
        f"**Segment** : {titre}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {OPENAI_LLM_MODEL}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Longueur du segment** : {len(texte)} caractères  ",
    ]
    if question:
        lignes.append(f"**Question posée** : {question}  ")
    lignes += ["", "---", ""]

    # Extrait du segment
    lignes += [
        "## Segment analysé",
        "",
        f"> {texte[:400].replace(chr(10), ' ')}{'…' if len(texte) > 400 else ''}",
        "",
        "---",
        "",
    ]

    # Discours rapporté
    lignes += [
        "## Discours rapporté identifié",
        "",
        discours_rapporte if discours_rapporte.upper() != "AUCUN" else "*Aucun discours rapporté identifié.*",
        "",
        "---",
        "",
        "## Analyse Toulmin",
        "",
        "> *Appliquée au discours de l'auteur uniquement.*",
        "",
    ]
    etiquettes = {
        "claim"    : "CLAIM     — thèse ou conclusion",
        "grounds"  : "GROUNDS   — données / preuves",
        "warrant"  : "WARRANT   — loi de passage",
        "backing"  : "BACKING   — appui du warrant",
        "qualifier": "QUALIFIER — nuances / modalités",
        "rebuttal" : "REBUTTAL  — objections anticipées",
    }
    for cle, lbl in etiquettes.items():
        val = comp[cle]
        sym = "✗" if val == "ABSENT" else "✔"
        lignes.append(f"- **{sym} {lbl}** : {val}")
    lignes += ["", "---", "", "## Séquence Adam", ""]
    adam_lbl = {
        "these_anterieure": "Thèse antérieure",
        "donnees"         : "Données / arguments",
        "conclusion"      : "Conclusion / thèse",
        "restriction"     : "Restriction / concession",
    }
    for cle, lbl in adam_lbl.items():
        lignes.append(f"- **{lbl}** : {adam[cle]}")
    lignes += ["", "---", "", "## Schème Walton et sophismes", ""]
    schema_en = SCHEMES_WALTON.get(schema, "—")
    lignes.append(f"- **Schème dominant** : `{schema}` — *{schema_en}*")
    if sophismes:
        lignes.append(f"- **⚑ Sophismes** : {', '.join(f'`{s}`' for s in sophismes)}")
    else:
        lignes.append("- **Sophismes** : aucun")
    lignes += ["", "---", "", "## Polyphonie", ""]
    bloc_poly = extraire_section(raw, "5. ANALYSE DE LA POLYPHONIE")
    lignes.append(bloc_poly if bloc_poly else "*Non disponible.*")
    lignes += ["", "---", "", "## Scores", ""]
    scores_affich = [
        ("Complétude Toulmin", "score_completude_toulmin"),
        ("Cohérence warrant",  "score_coherence_warrant"),
        ("Charge probatoire",  "score_charge_probatoire"),
        ("Risque sophisme",    "score_risque_sophisme"),
        ("Robustesse globale", "score_robustesse_globale"),
    ]
    for lbl, cle in scores_affich:
        lignes.append(f"- **{lbl}** : `{_barre(scores[cle])}`")
    lignes += ["", "---", "", "## Synthèse", ""]
    for l in synthese.split("\n"):
        l = l.strip()
        if l:
            lignes.append(l)
    lignes += [
        "", "---", "",
        "*Rapport généré par `A_toulmin_segment.py`.*  ",
        "*Cadres : Toulmin (1958), Adam (1992), Walton (1996).*",
    ]

    contenu = "\n".join(lignes)
    chemin  = output_dir / f"toulmin_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


def generer_json(
    titre: str, texte: str, question: str,
    scores: dict, schema: str, sophismes: list,
    raw: str, timestamp: str, output_dir: Path,
) -> Path:
    """JSON léger pour le script D (synthèse croisée)."""
    m_schema = re.search(r"SCHEMA_DOMINANT\s*:\s*(\w+)", raw, re.IGNORECASE)
    schema_val = m_schema.group(1).lower() if m_schema else "aucun"
    data = {
        "script"    : "A_toulmin_segment",
        "timestamp" : timestamp,
        "modele"    : OPENAI_LLM_MODEL,
        "titre"     : titre,
        "question"  : question,
        "nb_chars"  : len(texte),
        "scores"    : scores,
        "schema_walton_dominant" : schema_val,
        "sophismes_detectes"     : sophismes,
        "synthese"  : extraire_synthese(raw),
    }
    chemin = output_dir / f"toulmin_{timestamp}.json"
    chemin.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A_toulmin_segment — Analyse Toulmin/Adam/Walton sur segment."
    )
    parser.add_argument("--fichier", type=str, default=None,
                        help="Fichier .txt contenant le segment. "
                             "Si absent : saisie interactive.")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Ne pas demander confirmation du coût.")
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Saisie ────────────────────────────────────────────────────────────────
    if args.fichier:
        p = Path(args.fichier)
        if not p.exists():
            print(f"❌ Fichier introuvable : {p}")
            sys.exit(1)
        texte = p.read_text(encoding="utf-8").strip()
        print(f"✔ Segment chargé : {p.name} ({len(texte)} caractères)")
        print()
        titre = input("Titre du segment (obligatoire) : ").strip()
        while not titre:
            titre = input("Titre (obligatoire) : ").strip()
        question = input("Question facultative (Entrée pour ignorer) : ").strip()
    else:
        titre, texte, question = saisir_segment()

    # ── Estimation coût ───────────────────────────────────────────────────────
    tokens, cout = estimer_cout(texte, titre, question)
    confirmer_cout(tokens, cout, args.no_confirm)

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    print(f"\n  Analyse en cours ({OPENAI_LLM_MODEL}, T°={TEMPERATURE}, max_tokens={MAX_TOKENS})…")
    client = OpenAI()

    prompt = construire_prompt(titre, texte, question)
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

    # ── Extraction et sauvegarde ───────────────────────────────────────────────
    scores    = extraire_scores(raw)
    m_schema  = re.search(r"SCHEMA_DOMINANT\s*:\s*(\w+)", raw, re.IGNORECASE)
    schema    = m_schema.group(1).lower() if m_schema else "aucun"
    m_soph    = re.search(r"SOPHISMES\s*:\s*(.+)", raw, re.IGNORECASE)
    sophismes = []
    if m_soph and m_soph.group(1).strip().upper() != "AUCUN":
        sophismes = [s.strip().lower() for s in re.split(r"[,;]", m_soph.group(1))
                     if s.strip().lower() in SCHEMES_FALLACIEUX]

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md  = generer_rapport_md(
        titre, texte, question, raw, scores, timestamp, output_dir
    )
    chemin_json = generer_json(
        titre, texte, question, scores, schema, sophismes, raw, timestamp, output_dir
    )

    # ── Bilan console ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  BILAN — {titre[:50]}")
    print(f"{'═'*60}")
    print(f"  Robustesse globale : {scores['score_robustesse_globale']:.3f}")
    print(f"  Cohérence warrant  : {scores['score_coherence_warrant']:.3f}")
    print(f"  Charge probatoire  : {scores['score_charge_probatoire']:.3f}")
    print(f"  Schème dominant    : {schema}")
    if sophismes:
        print(f"  ⚑ Sophismes        : {', '.join(sophismes)}")
    print(f"{'═'*60}")
    print(f"\n✅ MD   : {chemin_md}")
    print(f"   JSON : {chemin_json}")
    print(f"\n   Lancez B_perelman_segment.py et C_rst_segment.py")
    print(f"   sur le même segment pour une analyse croisée complète.\n")


if __name__ == "__main__":
    main()
