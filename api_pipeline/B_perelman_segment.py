#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_perelman_segment.py
=====================
Analyse rhétorique approfondie (Nouvelle Rhétorique de Perelman) sur un
segment textuel délimité par l'historien.

CE SCRIPT EST DIFFÉRENT DU 10
──────────────────────────────
Le script 10 analyse le manuscrit entier paragraphe par paragraphe. Ses
scores individuels sont des signaux d'orientation. Ce script opère sur un
segment cohérent délimité par l'historien, ce qui permet deux choses
supplémentaires impossibles en mode batch :

  1. MOUVEMENT RHÉTORIQUE INTERNE : le LLM peut tracer l'évolution des
     techniques au fil du segment — l'auteur commence-t-il par l'autorité
     pour finir par la dissociation ? Ce mouvement est invisible paragraphe
     par paragraphe.

  2. CONSTRUCTION DE L'AUDITOIRE SUR LE SEGMENT : la construction de
     l'auditoire est un processus discursif qui se déploie sur plusieurs
     paragraphes, pas dans une seule phrase. Le segment donne la fenêtre
     nécessaire pour l'observer.

CADRE THÉORIQUE
───────────────
Chaïm Perelman & Lucie Olbrechts-Tyteca,
Traité de l'argumentation — La Nouvelle Rhétorique.
Bruxelles : Éditions de l'Université de Bruxelles, 1958 (rééd. 1988).

  A. TECHNIQUES D'ASSOCIATION
     A1. Quasi-logiques : incompatibilite | identite | transitif |
         reciprocite | inclusion | comparaison | sacrifice
     A2. Fondées sur la structure du réel : lien_causal |
         argument_pragmatique | argument_autorite | illustration |
         modele | anti_modele | analogie
     A3. Qui fondent la structure du réel : exemple | metaphore

  B. TECHNIQUES DE DISSOCIATION
     Paires philosophiques : apparence/réalité | moyen/fin |
     relatif/absolu | individu/collectif | lettre/esprit |
     subjectif/objectif | acte/personne | théorie/pratique

  C. AUDITOIRE
     Universel : l'argument prétend s'adresser à tout être raisonnable.
     Particulier disciplinaire : communauté des historiens.
     Particulier idéologique : groupe aux valeurs communes.

LIMITES 
─────────────────────────────

    Un LLm commercial aujourd'hui (en avril 2026) 
    peut mener à bien ce travail sur des corpus d'assez grande taille, 
    mais au delà de 1500 à 2000 mots il est très probable que la qualité 
    de l'analyse se dégradera sensiblement.

NOTE SUR L'AUDITOIRE DISCIPLINAIRE
────────────────────────────────────
En histoire académique, l'auditoire est presque toujours "particulier
disciplinaire" — la communauté des historiens, avec ses conventions
épistémiques (rapport aux sources, critique interne/externe, normes de
citation). Un auteur qui revendique l'auditoire universel dans un texte
historique academic fait souvent un usage discutable de Perelman.

ESTIMATION DES COÛTS
─────────────────────
Un segment de 5-8 paragraphes (~1500 mots) :
  Tokens entrée  : ~2500
  Tokens sortie  : ~2500
  gpt-4.1-mini   : ~0.002 $ par analyse
  gpt-4.1        : ~0.018 $ par analyse

UTILISATION
───────────
  python B_perelman_segment.py
  python B_perelman_segment.py --fichier mon_passage.txt
  python B_perelman_segment.py --no-confirm
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OPENAI_LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE       = 0.2       # légèrement supérieure à A : l'analyse rhétorique
                               # tolère une interprétation plus nuancée
MAX_TOKENS        = 4500
# le nombre de token est calculé de façon à optimiser coût et qualité de la réponse
# lorsque le LLM a consommé ces tokens il arrête l'analyse
# si le passage fourni est long il est possible que la réponse soit tronquée
# il faut alors augmenter le nombre de tokens alloués. 
OUTPUT_DIR        = "resultats"

COUT_INPUT_PER_1K  = 0.00040
COUT_OUTPUT_PER_1K = 0.00160

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

load_dotenv()

# =============================================================================
# DONNÉES DE RÉFÉRENCE
# =============================================================================

TECHNIQUES_PERELMAN = {
    "incompatibilite": "A1", "identite": "A1", "transitif": "A1",
    "reciprocite": "A1", "inclusion": "A1", "comparaison": "A1", "sacrifice": "A1",
    "lien_causal": "A2", "argument_pragmatique": "A2", "argument_autorite": "A2",
    "illustration": "A2", "modele": "A2", "anti_modele": "A2", "analogie": "A2",
    "exemple": "A3", "metaphore": "A3",
    "dissociation_apparence_realite": "B", "dissociation_moyen_fin": "B",
    "dissociation_relatif_absolu": "B", "dissociation_individu_collectif": "B",
    "dissociation_lettre_esprit": "B", "dissociation_autre": "B",
}
FAMILLES = {"A1": "quasi_logique", "A2": "structure_reel", "A3": "fonde_reel", "B": "dissociation"}
SOPHISTIQUES = [
    "auditoire_universel_abusif", "metaphore_glissante", "fausse_dichotomie",
    "autorite_non_qualifiee", "petition_principe",
]

SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique académique, spécialisé dans la Nouvelle Rhétorique "
    "de Perelman et Olbrechts-Tyteca (Traité de l'argumentation, 1958). Tu analyses "
    "les techniques argumentatives des textes historiques sur des segments cohérents "
    "délimités par l'historien — pas sur des paragraphes isolés. "
    "Tu traces le mouvement rhétorique interne du segment (évolution des techniques). "
    "Tu réponds en français."
)

# =============================================================================
# SAISIE INTERACTIVE (identique à A — mutualisée)
# =============================================================================

def saisir_segment(nom_script: str = "B — Perelman") -> tuple[str, str, str]:
    print("\n" + "═"*60)
    print(f"  ANALYSE RHÉTORIQUE SUR SEGMENT — PERELMAN")
    print("═"*60)
    print()
    print("  Ce script analyse les stratégies rhétoriques d'un segment")
    print("  textuel délimité par vous (Nouvelle Rhétorique de Perelman).")
    print()

    print("─"*60)
    print("  TITRE DU SEGMENT (obligatoire)")
    print()
    titre = ""
    while not titre.strip():
        titre = input("  Titre : ").strip()
        if not titre:
            print("  ⚠ Le titre est obligatoire.")

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
        print("  Collez le texte puis tapez '###FIN###' sur une ligne seule.")
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
        r = input("  Continuer quand même ? [o/N] : ").strip().lower()
        if r != "o":
            sys.exit(0)

    print()
    print("─"*60)
    print("  QUESTION FACULTATIVE (Entrée pour ignorer)")
    print("  Exemples :")
    print("    'Comment est construit l'auditoire dans ce passage ?'")
    print("    'Y a-t-il un glissement vers l'auditoire universel ?'")
    print("    'Quelle est la technique dominante et est-elle cohérente ?'")
    print()
    question = input("  Question : ").strip()

    return titre, texte, question


# =============================================================================
# ESTIMATION DES COÛTS
# =============================================================================

def estimer_cout(texte: str, titre: str, question: str) -> tuple[int, float]:
    chars = len(SYSTEM_PROMPT) + len(texte) + len(titre) + len(question) + 600
    tokens_in  = chars // 4
    tokens_out = MAX_TOKENS
    cout = (tokens_in * COUT_INPUT_PER_1K / 1000
            + tokens_out * COUT_OUTPUT_PER_1K / 1000)
    return tokens_in + tokens_out, round(cout, 4)


def confirmer_cout(tokens: int, cout: float, no_confirm: bool) -> None:
    print(f"\n  Estimation : ~{tokens} tokens | ~{cout:.4f} $ ({OPENAI_LLM_MODEL})")
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

    return f"""Analyse le segment suivant selon la Nouvelle Rhétorique de Perelman.
Ce segment a été délimité par l'historien — c'est une unité rhétorique cohérente.

══════════════════════════════════════════
SEGMENT : {titre}
══════════════════════════════════════════
{texte}
{question_bloc}
══════════════════════════════════════════
ANALYSE PERELMAN DEMANDÉE
══════════════════════════════════════════

━━━ 1. TECHNIQUES D'ASSOCIATION ━━━

A1 — QUASI-LOGIQUES :
  Parmi : incompatibilite | identite | transitif | reciprocite |
          inclusion | comparaison | sacrifice
  Pour chaque technique détectée : cite le passage exact du segment.

A2 — FONDÉES SUR LA STRUCTURE DU RÉEL :
  Parmi : lien_causal | argument_pragmatique | argument_autorite |
          illustration | modele | anti_modele | analogie
  Pour chaque technique : cite le passage exact.

A3 — QUI FONDENT LA STRUCTURE DU RÉEL :
  Parmi : exemple | metaphore
  Pour chaque technique : cite le passage exact.

TECHNIQUE_DOMINANTE: <nom_technique>

━━━ 2. TECHNIQUES DE DISSOCIATION ━━━
Identifie les paires philosophiques (apparence/réalité, moyen/fin,
relatif/absolu, individu/collectif, lettre/esprit, ou autre).
Pour chaque paire : quelle est sa fonction dans ce segment ?
Quel terme est valorisé, lequel est dévalué ?

━━━ 3. MOUVEMENT RHÉTORIQUE INTERNE ━━━
Comment évoluent les techniques au fil du segment ?
L'auteur commence-t-il par établir des faits (A2) pour conclure par une
dissociation (B) ? Y a-t-il une progression, un retournement, une
accumulation ? Décris le mouvement rhétorique d'ensemble.
Ce mouvement est-il cohérent avec le titre et l'objectif du segment ?

━━━ 4. CONSTRUCTION DE L'AUDITOIRE ━━━
TYPE_AUDITOIRE: universel | particulier_disciplinaire | particulier_ideologique | indetermine
Quels marqueurs textuels révèlent la construction de l'auditoire ?
L'auteur revendique-t-il abusivement l'auditoire universel ?
La construction de l'auditoire évolue-t-elle au fil du segment ?

━━━ 5. VALEURS MOBILISÉES ━━━
VALEURS: <valeur1>, <valeur2>, …
Ces valeurs sont-elles cohérentes entre elles ?
Y a-t-il tension ou contradiction entre les valeurs mobilisées
au début et à la fin du segment ?

━━━ 6. USAGES SOPHISTIQUES ━━━
Parmi : auditoire_universel_abusif | metaphore_glissante | fausse_dichotomie |
        autorite_non_qualifiee | petition_principe
USAGES_SOPHISTIQUES: <usage1>, <usage2>  (ou AUCUN)
Pour chaque usage : cite le passage et explique pourquoi c'est problématique.

━━━ 7. SCORES (format strict) ━━━
SCORE_FORCE_PERSUASIVE: X/10
SCORE_ANCRAGE_AUDITOIRE: X/10
SCORE_COHERENCE_VALEURS: X/10
SCORE_RISQUE_SOPHISTIQUE: X/10
SCORE_PROFIL_ARGUMENTATIF: X/10

━━━ 8. SYNTHÈSE RHÉTORIQUE ━━━
En 4-5 phrases : comment ce segment construit-il sa persuasion ?
Quelles techniques dominent et pourquoi ? Le mouvement rhétorique
est-il efficace ? Quelle révision renforcerait la force persuasive ?
Si une question a été posée : réponse directe."""


# =============================================================================
# EXTRACTION
# =============================================================================

NOMS_COURTS_SCORES_B = {
    "score_force_persuasive"               : ["force", "force_persuasive"],
    "score_ancrage_auditoire"              : ["ancrage", "ancrage_auditoire"],
    "score_coherence_valeurs"              : ["valeurs", "coherence_valeurs", "cohérence_valeurs"],
    "score_risque_sophistique_rhethorique" : ["risque", "sophistique", "risque_sophistique"],
    "score_profil_argumentatif"            : ["profil", "profil_argumentatif"],
}

PATTERNS_SCORES = {
    "score_force_persuasive"               : r"SCORE_FORCE_PERSUASIVE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_ancrage_auditoire"              : r"SCORE_ANCRAGE_AUDITOIRE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_coherence_valeurs"              : r"SCORE_COHERENCE_VALEURS\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_risque_sophistique_rhethorique" : r"SCORE_RISQUE_SOPHISTIQUE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_profil_argumentatif"            : r"SCORE_PROFIL_ARGUMENTATIF\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
}


def _normaliser_score(v: float, avec_denominateur: bool) -> float:
    """Normalise vers [0,1]. Voir extraire_scores() pour la règle complète."""
    if avec_denominateur or v == int(v) or v > 1.0:
        return round(max(0.0, min(10.0, v)) / 10.0, 3)
    return round(max(0.0, min(1.0, v)), 3)


def extraire_scores(raw: str) -> dict:
    """
    Extrait les cinq scores Perelman depuis la réponse LLM brute.

    Quatre niveaux de robustesse (identique à A_toulmin_segment.py) :
    1. Format strict : SCORE_X: N/10  (point ou virgule décimale)
    2. Espaces autour du / : SCORE_X: N / 10
    3. Nom court sans préfixe SCORE_ : force : 8  |  profil = 7.5
    4. Tableau Markdown : | Force persuasive | 8 |

    Règle de normalisation sans /10 : entier → sur 10 ; décimal > 1 → sur 10 ;
    décimal ≤ 1 → déjà normalisé.
    """
    scores = {}

    for nom, pattern in PATTERNS_SCORES.items():
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", "."))
            scores[nom] = _normaliser_score(v, avec_denominateur=True)
            continue

        trouve = False
        for nom_court in NOMS_COURTS_SCORES_B.get(nom, []):
            pat3 = (rf"(?:^|[\s|])(?:{re.escape(nom_court)})"
                    rf"\s*[:=]\s*(\d+(?:[.,]\d+)?)\s*(?:/\s*10)?")
            m3 = re.search(pat3, raw, re.IGNORECASE | re.MULTILINE)
            if m3:
                v = float(m3.group(1).replace(",", "."))
                avec_denom = bool(re.search(
                    r"/\s*10", raw[m3.start():m3.end() + 10]
                ))
                scores[nom] = _normaliser_score(v, avec_denominateur=avec_denom)
                trouve = True
                break

        if trouve:
            continue

        for nom_court in NOMS_COURTS_SCORES_B.get(nom, []):
            pat4 = (rf"\|\s*[^|]*{re.escape(nom_court)}[^|]*\|"
                    rf"\s*(\d+(?:[.,]\d+)?)\s*\|")
            m4 = re.search(pat4, raw, re.IGNORECASE)
            if m4:
                v = float(m4.group(1).replace(",", "."))
                scores[nom] = _normaliser_score(v, avec_denominateur=False)
                break
        else:
            scores[nom] = 0.5

    return scores


def extraire_techniques(raw: str) -> dict:
    par_famille = {f: [] for f in FAMILLES.values()}
    for tech, code in TECHNIQUES_PERELMAN.items():
        nom = FAMILLES[code]
        if re.search(rf"\b{re.escape(tech)}\b", raw, re.IGNORECASE):
            if tech not in par_famille[nom]:
                par_famille[nom].append(tech)
    return par_famille


def extraire_paires(raw: str) -> list[str]:
    paires_ref = [
        "apparence / réalité", "moyen / fin", "relatif / absolu",
        "individu / collectif", "lettre / esprit", "subjectif / objectif",
        "acte / personne", "théorie / pratique",
    ]
    return [p for p in paires_ref
            if re.search(p.replace(" / ", r"\s*/\s*"), raw, re.IGNORECASE)]


def extraire_valeurs(raw: str) -> list[str]:
    m = re.search(r"VALEURS\s*:\s*(.+)", raw, re.IGNORECASE)
    return [v.strip() for v in re.split(r"[,;]", m.group(1)) if v.strip()] if m else []


def extraire_usages(raw: str) -> list[str]:
    m = re.search(r"USAGES_SOPHISTIQUES\s*:\s*(.+)", raw, re.IGNORECASE)
    if m and m.group(1).strip().upper() != "AUCUN":
        return [s.strip().lower() for s in re.split(r"[,;]", m.group(1))
                if s.strip().lower() in SOPHISTIQUES]
    return []


def extraire_auditoire(raw: str) -> str:
    m = re.search(r"TYPE_AUDITOIRE\s*:\s*(\S+)", raw, re.IGNORECASE)
    if m:
        t = m.group(1).lower().strip(".,;")
        valides = {"universel", "particulier_disciplinaire",
                   "particulier_ideologique", "indetermine"}
        return t if t in valides else "indetermine"
    return "indetermine"


def extraire_technique_dominante(raw: str) -> str:
    m = re.search(r"TECHNIQUE_DOMINANTE\s*:\s*(\w+)", raw, re.IGNORECASE)
    if m:
        t = m.group(1).lower()
        return t if t in TECHNIQUES_PERELMAN else "indeterminee"
    return "indeterminee"


def extraire_mouvement(raw: str) -> str:
    m = re.search(
        r"━+\s*3\. MOUVEMENT RHÉTORIQUE.*?━+\n(.*?)(?=━+\s*4\.|\Z)",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else ""


def extraire_synthese(raw: str) -> str:
    m = re.search(
        r"(?:SYNTHÈSE RHÉTORIQUE|synthèse rhétorique|SYNTHÈSE)[^\n]*\n(.+?)$",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else raw.strip()[-800:]


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def _barre(v: float, w: int = 20) -> str:
    r = round(v * w)
    return f"[{'█'*r}{'░'*(w-r)}] {v:.2f}"


FAMILLES_LABEL = {
    "quasi_logique" : "A1 — Quasi-logiques",
    "structure_reel": "A2 — Fondées sur la structure du réel",
    "fonde_reel"    : "A3 — Qui fondent la structure du réel",
    "dissociation"  : "B  — Dissociation",
}
AUD_DESC = {
    "universel"                : "prétend s'adresser à tout être raisonnable",
    "particulier_disciplinaire": "communauté des historiens (conventions épistémiques)",
    "particulier_ideologique"  : "groupe aux valeurs communes",
    "indetermine"              : "auditoire non clairement construit",
}


def generer_rapport_md(
    titre: str, texte: str, question: str, raw: str,
    scores: dict, timestamp: str, output_dir: Path,
) -> Path:
    techniques = extraire_techniques(raw)
    paires     = extraire_paires(raw)
    valeurs    = extraire_valeurs(raw)
    usages     = extraire_usages(raw)
    aud        = extraire_auditoire(raw)
    tech_dom   = extraire_technique_dominante(raw)
    mouvement  = extraire_mouvement(raw)
    synthese   = extraire_synthese(raw)

    lignes = []
    lignes += [
        "# Analyse rhétorique — Nouvelle Rhétorique (Perelman)",
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

    lignes += [
        "## Segment analysé",
        "",
        f"> {texte[:400].replace(chr(10), ' ')}{'…' if len(texte) > 400 else ''}",
        "",
        "---",
        "",
        "## Techniques Perelman détectées",
        "",
    ]
    for fam_key, fam_label in FAMILLES_LABEL.items():
        items = techniques.get(fam_key, [])
        lignes.append(f"**{fam_label}**")
        if items:
            for t in items:
                lignes.append(f"- `{t.replace('_', ' ')}`")
        else:
            lignes.append("*(aucune détectée)*")
        lignes.append("")

    lignes += [
        f"**Technique dominante** : `{tech_dom.replace('_', ' ')}`",
        "",
        "---",
        "",
        "## Mouvement rhétorique interne",
        "",
        mouvement if mouvement else "*Non disponible.*",
        "",
        "---",
        "",
        "## Paires de dissociation",
        "",
    ]
    if paires:
        for p in paires:
            lignes.append(f"- {p}")
    else:
        lignes.append("*Aucune paire identifiée.*")
    lignes += ["", "---", "", "## Construction de l'auditoire", ""]
    aud_desc = AUD_DESC.get(aud, "")
    lignes += [
        f"`{aud}` — {aud_desc}",
        "",
        "---",
        "",
        "## Valeurs mobilisées",
        "",
    ]
    if valeurs:
        lignes.append(", ".join(f"`{v}`" for v in valeurs))
    else:
        lignes.append("*Aucune valeur identifiée.*")
    lignes += ["", "---", "", "## Usages sophistiques", ""]
    if usages:
        for u in usages:
            lignes.append(f"- ⚑ `{u.replace('_', ' ')}`")
    else:
        lignes.append("*Aucun usage sophistique détecté.*")
    lignes += ["", "---", "", "## Scores", ""]
    scores_affich = [
        ("Force persuasive",      "score_force_persuasive"),
        ("Ancrage auditoire",     "score_ancrage_auditoire"),
        ("Cohérence des valeurs", "score_coherence_valeurs"),
        ("Risque sophistique",    "score_risque_sophistique_rhethorique"),
        ("Profil argumentatif",   "score_profil_argumentatif"),
    ]
    for lbl, cle in scores_affich:
        lignes.append(f"- **{lbl}** : `{_barre(scores[cle])}`")
    lignes += ["", "---", "", "## Synthèse rhétorique", ""]
    for l in synthese.split("\n"):
        l = l.strip()
        if l:
            lignes.append(l)
    lignes += [
        "", "---", "",
        "*Rapport généré par `B_perelman_segment.py`.*  ",
        "*Cadre : Perelman & Olbrechts-Tyteca, Traité de l'argumentation (1958).*",
    ]

    contenu = "\n".join(lignes)
    chemin  = output_dir / f"perelman_seg_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


def generer_json(
    titre: str, texte: str, question: str,
    scores: dict, raw: str, timestamp: str, output_dir: Path,
) -> Path:
    data = {
        "script"             : "B_perelman_segment",
        "timestamp"          : timestamp,
        "modele"             : OPENAI_LLM_MODEL,
        "titre"              : titre,
        "question"           : question,
        "nb_chars"           : len(texte),
        "scores"             : scores,
        "technique_dominante": extraire_technique_dominante(raw),
        "type_auditoire"     : extraire_auditoire(raw),
        "techniques_detectees": extraire_techniques(raw),
        "paires_dissociation": extraire_paires(raw),
        "valeurs_mobilisees" : extraire_valeurs(raw),
        "usages_sophistiques": extraire_usages(raw),
        "mouvement_rhetorique": extraire_mouvement(raw),
        "synthese"           : extraire_synthese(raw),
    }
    chemin = output_dir / f"perelman_seg_{timestamp}.json"
    chemin.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="B_perelman_segment — Analyse Nouvelle Rhétorique sur segment."
    )
    parser.add_argument("--fichier", type=str, default=None)
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fichier:
        p = Path(args.fichier)
        if not p.exists():
            print(f"❌ Fichier introuvable : {p}")
            sys.exit(1)
        texte = p.read_text(encoding="utf-8").strip()
        print(f"✔ Segment chargé : {p.name} ({len(texte)} caractères)")
        titre = input("Titre du segment (obligatoire) : ").strip()
        while not titre:
            titre = input("Titre (obligatoire) : ").strip()
        question = input("Question facultative (Entrée pour ignorer) : ").strip()
    else:
        titre, texte, question = saisir_segment()

    tokens, cout = estimer_cout(texte, titre, question)
    confirmer_cout(tokens, cout, args.no_confirm)

    print(f"\n  Analyse en cours ({OPENAI_LLM_MODEL}, T°={TEMPERATURE}, max_tokens={MAX_TOKENS})…")
    client = OpenAI()

    try:
        raw = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": construire_prompt(titre, texte, question)},
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

    scores    = extraire_scores(raw)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md = generer_rapport_md(titre, texte, question, raw, scores, timestamp, output_dir)
    chemin_json = generer_json(titre, texte, question, scores, raw, timestamp, output_dir)

    print(f"\n{'═'*60}")
    print(f"  BILAN — {titre[:50]}")
    print(f"{'═'*60}")
    print(f"  Profil argumentatif : {scores['score_profil_argumentatif']:.3f}")
    print(f"  Force persuasive    : {scores['score_force_persuasive']:.3f}")
    print(f"  Ancrage auditoire   : {scores['score_ancrage_auditoire']:.3f}")
    print(f"  Technique dominante : {extraire_technique_dominante(raw)}")
    print(f"  Auditoire           : {extraire_auditoire(raw)}")
    usages = extraire_usages(raw)
    if usages:
        print(f"  ⚑ Usages sophistiques : {', '.join(usages)}")
    print(f"{'═'*60}")
    print(f"\n✅ MD   : {chemin_md}")
    print(f"   JSON : {chemin_json}")
    print(f"\n   Lancez C_rst_segment.py sur le même segment,")
    print(f"   puis D_synthese_croisee.py pour l'analyse intégrée.\n")


if __name__ == "__main__":
    main()
