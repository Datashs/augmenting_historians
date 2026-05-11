#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_rst_segment.py
================
Analyse des relations rhétoriques (RST — Rhetorical Structure Theory) sur
un segment textuel délimité par l'historien.

RHÉTORICAL STRUCTURE THEORY — PRINCIPES
─────────────────────────────────────────
Mann & Thompson (1988), "Rhetorical Structure Theory: Toward a functional
theory of text organization", Text 8(3), p. 243-281.

RST modélise la cohérence d'un texte en identifiant des relations entre
unités discursives (EDU — Elementary Discourse Units). Chaque relation
connecte un NOYAU (nucleus) et une PÉRIPHÉRIE (satellite) :
  - le noyau est l'unité centrale, celle qui porte le propos
  - le satellite est l'unité qui sert le noyau

RELATIONS RST UTILISÉES DANS CE SCRIPT
────────────────────────────────────────
  Groupe 1 — Relations présentatives (présentent l'information)
    elaboration  : le satellite développe, précise ou illustre le noyau
    preparation  : le satellite prépare le lecteur avant le noyau
    background   : le satellite fournit le contexte nécessaire au noyau
    circumstance : le satellite décrit le contexte situationnel du noyau

  Groupe 2 — Relations sujet-matière (portent sur le contenu)
    cause        : le satellite est la cause du noyau
    result       : le satellite est le résultat du noyau
    evidence     : le satellite est une preuve du noyau
    justify      : le satellite justifie l'acte de communication du noyau

  Groupe 3 — Relations de contraste
    contrast     : noyau et satellite sont en opposition
    concession   : le satellite admet un élément adverse au noyau
    antithesis   : le satellite contredit ou nie le noyau

  Groupe 4 — Relations rhétoriques actives
    motivation   : le satellite incite à l'action présentée dans le noyau
    restatement  : le satellite reformule le noyau
    summary      : le satellite résume le noyau

GRANULARITÉ — CHOIX DÉLIBÉRÉ
──────────────────────────────
Ce script segmente le texte en unités au niveau de la PHRASE (pas de la
proposition). Avantages :
  - Un arbre à 10-20 nœuds est lisible ; un arbre à 60 nœuds ne l'est pas
  - La détection LLM est plus fiable à cette granularité
  - L'historien peut identifier les phrases dans son texte

Inconvénient assumé : on perd les relations intra-phrastiques. Pour un
usage rhétorique et une relecture critique, c'est un compromis acceptable.

STATUT ÉPISTÉMIQUE — À LIRE AVANT D'UTILISER
──────────────────────────────────────────────
RST a été formalisé pour de l'annotation humaine experte. Un annotateur RST
formé passe 20-30 minutes sur un paragraphe. Ce que produit ce script est
une APPROXIMATION heuristique, pas une annotation RST rigoureuse.

Fiabilité par type de relation (estimation) :
  Haute  : elaboration, evidence, contrast, concession (relations claires)
  Moyenne: cause, result, background, restatement
  Basse  : preparation, justify, motivation, circumstance (très contextuelles)

Utilisez ce script comme outil d'exploration, pas de vérification.
Les zones d'incertitude sont signalées dans la sortie.

Un LLm commercial aujourd'hui (en avril 2026) peut mener à bien ce travail 
sur des textes d'assez grande taille, mais au delà de 1500 à 2000 mots 
il est très probable que la qualité  de l'analyse se dégradera sensiblement.

SORTIE GRAPHIQUE — ARBRE MERMAID
──────────────────────────────────
L'arbre RST est produit en syntaxe Mermaid, lisible dans tout éditeur
Markdown qui supporte Mermaid (VS Code, Obsidian, GitHub, Typora…).
Aucune dépendance externe requise — le script génère la syntaxe en Python.

ESTIMATION DES COÛTS
─────────────────────
Un segment de 5-10 phrases :
  Tokens entrée  : ~1500-2500
  Tokens sortie  : ~2000-3000  (l'arbre JSON peut être long)
  gpt-4.1-mini   : ~0.002-0.004 $ par analyse

UTILISATION
───────────
  python C_rst_segment.py
  python C_rst_segment.py --fichier mon_passage.txt
  python C_rst_segment.py --no-confirm
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

OPENAI_LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE       = 0.10      # légèrement supérieure à A : RST demande
                               # une interprétation contextuelle fine
MAX_TOKENS        = 5500      # l'arbre JSON peut être verbeux
# le nombre de token est calculé de façon à optimiser coût et qualité de la réponse
# lorsque le LLM a consommé ces tokens il arrête l'analyse
# si le passage fourni est long il est possible que la réponse soit tronquée
# il faut alors augmenter le nombre de tokens alloués. 
OUTPUT_DIR        = "resultats"

COUT_INPUT_PER_1K  = 0.00040
COUT_OUTPUT_PER_1K = 0.00160

MIN_SEGMENT_CHARS = 200
# Nombre maximum de phrases à segmenter (au-delà l'arbre devient illisible)
MAX_PHRASES       = 20

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
# RELATIONS RST RECONNUES
# =============================================================================

RELATIONS_RST = {
    # Groupe 1 — présentatives
    "elaboration"  : ("présentative", "le satellite développe, précise ou illustre le noyau"),
    "preparation"  : ("présentative", "le satellite prépare le lecteur avant le noyau"),
    "background"   : ("présentative", "le satellite fournit le contexte nécessaire au noyau"),
    "circumstance" : ("présentative", "le satellite décrit le contexte situationnel du noyau"),
    # Groupe 2 — sujet-matière
    "cause"        : ("sujet-matière", "le satellite est la cause du noyau"),
    "result"       : ("sujet-matière", "le satellite est le résultat du noyau"),
    "evidence"     : ("sujet-matière", "le satellite est une preuve du noyau"),
    "justify"      : ("sujet-matière", "le satellite justifie l'acte de communication du noyau"),
    # Groupe 3 — contraste
    "contrast"     : ("contraste", "noyau et satellite sont en opposition"),
    "concession"   : ("contraste", "le satellite admet un élément adverse au noyau"),
    "antithesis"   : ("contraste", "le satellite contredit ou nie le noyau"),
    # Groupe 4 — rhétoriques actives
    "motivation"   : ("rhétorique", "le satellite incite à l'action présentée dans le noyau"),
    "restatement"  : ("rhétorique", "le satellite reformule le noyau"),
    "summary"      : ("rhétorique", "le satellite résume le noyau"),
}

SYSTEM_PROMPT = (
    "Tu es un expert en analyse du discours, spécialisé dans la Rhetorical "
    "Structure Theory (RST) de Mann & Thompson (1988). Tu analyses des segments "
    "textuels délimités par l'historien en identifiant les relations rhétoriques "
    "entre phrases. Tu travailles à la granularité de la phrase. "
    "Tu signales explicitement tes incertitudes. Tu réponds en français."
)

# =============================================================================
# SAISIE INTERACTIVE
# =============================================================================

def saisir_segment() -> tuple[str, str, str]:
    print("\n" + "═"*60)
    print("  ANALYSE RST SUR SEGMENT — RELATIONS RHÉTORIQUES")
    print("═"*60)
    print()
    print("  Ce script identifie les relations rhétoriques entre les")
    print("  phrases d'un segment (RST — Mann & Thompson 1988).")
    print("  Granularité : la phrase. Taille recommandée : 5-15 phrases.")
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

    # Estimation du nombre de phrases
    phrases_est = len(re.findall(r'[.!?]\s', texte)) + 1
    if phrases_est > MAX_PHRASES:
        print(f"\n  ⚠ Le segment contient ~{phrases_est} phrases.")
        print(f"    Au-delà de {MAX_PHRASES} phrases, l'arbre RST devient illisible.")
        print(f"    Recommandation : découpez en sous-segments plus courts.")
        r = input("  Continuer quand même ? [o/N] : ").strip().lower()
        if r != "o":
            sys.exit(0)

    print()
    print("─"*60)
    print("  QUESTION FACULTATIVE (Entrée pour ignorer)")
    print("  Exemples :")
    print("    'Quelle phrase est le noyau central du segment ?'")
    print("    'Y a-t-il des relations de preuve bien construites ?'")
    print("    'Comment les concessions sont-elles articulées ?'")
    print()
    question = input("  Question : ").strip()

    return titre, texte, question


# =============================================================================
# ESTIMATION DES COÛTS
# =============================================================================

def estimer_cout(texte: str, titre: str, question: str) -> tuple[int, float]:
    chars = len(SYSTEM_PROMPT) + len(texte) + len(titre) + len(question) + 800
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
    relations_str = "\n".join(
        f"  {k:<14} : {v[1]}"
        for k, v in RELATIONS_RST.items()
    )
    question_bloc = (
        f"\n══════════════════════════════════════════\n"
        f"QUESTION DE L'HISTORIEN\n"
        f"══════════════════════════════════════════\n"
        f"{question}\n"
    ) if question else ""

    return f"""Analyse le segment suivant selon la Rhetorical Structure Theory (RST).
Ce segment a été délimité par l'historien.

══════════════════════════════════════════
SEGMENT : {titre}
══════════════════════════════════════════
{texte}
{question_bloc}
══════════════════════════════════════════
ANALYSE RST DEMANDÉE
══════════════════════════════════════════

RELATIONS DISPONIBLES (utilise UNIQUEMENT ces noms) :
{relations_str}

━━━ 1. SEGMENTATION EN PHRASES ━━━
Numérotez chaque phrase du segment de P1 à Pn.
Restez fidèle au découpage naturel du texte — ne fusionnez pas et
ne divisez pas les phrases sauf si une proposition subordonnée
constitue clairement une unité rhétorique autonome.

Format strict :
P1: [texte exact de la première phrase]
P2: [texte exact de la deuxième phrase]
...

━━━ 2. NOYAU CENTRAL ━━━
NOYAU: P[numéro]
Identifie la phrase qui porte la thèse ou l'information centrale du
segment — celle dont toutes les autres dépendent rhétoriquement.
Justifie en une phrase.

━━━ 3. RELATIONS RST ━━━
Pour chaque relation identifiée :
RELATION: [P_noyau] --[nom_relation]--> [P_satellite]
CONFIANCE: haute | moyenne | basse
NOTE: [une phrase d'explication, signale les incertitudes]

Identifie TOUTES les relations que tu perçois. Une phrase peut être
à la fois noyau d'une relation et satellite d'une autre.
Signale explicitement les cas ambigus (confiance basse).

━━━ 4. STRUCTURE D'ENSEMBLE ━━━
Décris en 3-4 phrases l'architecture rhétorique du segment :
- Quelle est la relation dominante ?
- Y a-t-il une progression linéaire ou une structure en étoile autour du noyau ?
- Les relations de preuve (evidence) et de concession sont-elles bien construites ?
- Y a-t-il des relations manquantes ou des ruptures de cohérence ?

━━━ 5. ARBRE RST (JSON strict) ━━━
Produis la structure de l'arbre en JSON valide pour la génération
automatique du diagramme Mermaid. Format :
{{
  "noyau_central": "P1",
  "relations": [
    {{"noyau": "P1", "satellite": "P2", "relation": "evidence", "confiance": "haute"}},
    {{"noyau": "P1", "satellite": "P3", "relation": "concession", "confiance": "moyenne"}}
  ]
}}
JSON_ARBRE:
[JSON ici]
FIN_JSON

━━━ 6. SYNTHÈSE RST ━━━
En 3-4 phrases : comment la structure rhétorique du segment sert-elle
l'argumentation ? Quelles sont les forces et faiblesses de la cohérence
discursive ? Si une question a été posée : réponse directe."""


# =============================================================================
# EXTRACTION ET CONSTRUCTION DE L'ARBRE
# =============================================================================

def extraire_phrases(raw: str) -> list[dict]:
    """Extrait les phrases numérotées P1..Pn de la réponse LLM."""
    phrases = []
    for m in re.finditer(r"^P(\d+)\s*:\s*(.+)$", raw, re.MULTILINE):
        phrases.append({"id": f"P{m.group(1)}", "texte": m.group(2).strip()})
    return phrases


def extraire_noyau(raw: str) -> str:
    m = re.search(r"NOYAU\s*:\s*(P\d+)", raw, re.IGNORECASE)
    return m.group(1) if m else "P1"


def extraire_relations(raw: str) -> list[dict]:
    """Extrait d'abord depuis le JSON, puis en fallback depuis le texte libre."""
    # Tentative JSON
    m = re.search(r"JSON_ARBRE:\s*(\{.*?\})\s*FIN_JSON", raw, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("relations", [])
        except json.JSONDecodeError:
            pass

    # Fallback : pattern textuel
    rels = []
    for m in re.finditer(
        r"RELATION\s*:\s*(P\d+)\s*--\s*\[?(\w+)\]?\s*-->\s*(P\d+).*?\n"
        r"CONFIANCE\s*:\s*(\w+)",
        raw, re.IGNORECASE
    ):
        rel = m.group(2).lower()
        if rel in RELATIONS_RST:
            rels.append({
                "noyau"    : m.group(1),
                "satellite": m.group(3),
                "relation" : rel,
                "confiance": m.group(4).lower(),
            })
    return rels


def extraire_synthese(raw: str) -> str:
    m = re.search(
        r"(?:SYNTHÈSE RST|synthèse rst|SYNTHÈSE)[^\n]*\n(.+?)$",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else raw.strip()[-600:]


def extraire_structure(raw: str) -> str:
    m = re.search(
        r"━+\s*4\. STRUCTURE D'ENSEMBLE.*?━+\n(.*?)(?=━+\s*5\.|\Z)",
        raw, re.IGNORECASE | re.DOTALL
    )
    return m.group(1).strip() if m else ""


# =============================================================================
# GÉNÉRATION MERMAID
# =============================================================================

def generer_mermaid(
    phrases: list[dict],
    relations: list[dict],
    noyau: str,
    titre: str,
) -> str:
    """
    Construit la syntaxe Mermaid pour l'arbre RST.

    Convention visuelle :
      - Nœuds rectangulaires arrondis pour toutes les phrases
      - Noyau central en gras (style spécial)
      - Étiquettes des arêtes = nom de la relation
      - Confiance basse → arête en pointillés (style différent)
    """
    lignes = ["```mermaid", "graph TD"]

    # Nœuds
    for p in phrases:
        texte_court = p["texte"][:50].replace('"', "'")
        if len(p["texte"]) > 50:
            texte_court += "…"
        if p["id"] == noyau:
            lignes.append(f'    {p["id"]}["{p["id"]} ★ {texte_court}"]')
        else:
            lignes.append(f'    {p["id"]}("{p["id"]} {texte_court}")')

    # Relations
    for r in relations:
        nod = r.get("noyau", "?")
        sat = r.get("satellite", "?")
        rel = r.get("relation", "?")
        conf = r.get("confiance", "haute")
        if conf == "basse":
            lignes.append(f'    {nod} -. "{rel}" .-> {sat}')
        else:
            lignes.append(f'    {nod} -- "{rel}" --> {sat}')

    # Style noyau
    lignes.append(f"    style {noyau} fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px")
    lignes.append("```")

    return "\n".join(lignes)


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def generer_rapport_md(
    titre: str, texte: str, question: str, raw: str,
    timestamp: str, output_dir: Path,
) -> Path:
    phrases   = extraire_phrases(raw)
    noyau     = extraire_noyau(raw)
    relations = extraire_relations(raw)
    synthese  = extraire_synthese(raw)
    structure = extraire_structure(raw)
    mermaid   = generer_mermaid(phrases, relations, noyau, titre)

    # Statistiques sur les relations
    rel_counts: dict = {}
    for r in relations:
        rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
    rel_dom = max(rel_counts, key=rel_counts.get) if rel_counts else "—"
    n_basse = sum(1 for r in relations if r.get("confiance") == "basse")

    lignes = []
    lignes += [
        "# Analyse RST — Relations rhétoriques",
        "",
        f"**Segment** : {titre}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {OPENAI_LLM_MODEL}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Phrases identifiées** : {len(phrases)}  ",
        f"**Relations identifiées** : {len(relations)}  ",
        f"**Relation dominante** : `{rel_dom}`  ",
        f"**Relations à faible confiance** : {n_basse}  ",
    ]
    if question:
        lignes.append(f"**Question posée** : {question}  ")
    lignes += [
        "",
        "> ⚠ *Ce rapport est une approximation heuristique, non une annotation*  ",
        "> *RST rigoureuse. Les relations à confiance 'basse' sont signalées.*  ",
        "> *Utilisez ce script comme outil d'exploration, pas de vérification.*",
        "",
        "---",
        "",
        "## Segment analysé",
        "",
        f"> {texte[:400].replace(chr(10), ' ')}{'…' if len(texte) > 400 else ''}",
        "",
        "---",
        "",
        "## Segmentation en phrases",
        "",
    ]
    for p in phrases:
        noyau_mark = " ★ *noyau central*" if p["id"] == noyau else ""
        lignes.append(f"- **{p['id']}{noyau_mark}** : {p['texte']}")
    lignes += [
        "",
        f"**Noyau central** : `{noyau}`",
        "",
        "---",
        "",
        "## Relations rhétoriques identifiées",
        "",
        "| Noyau | Relation | Satellite | Confiance |",
        "|---|---|---|---|",
    ]
    for r in relations:
        conf_sym = "🔴" if r.get("confiance") == "basse" else (
                   "🟡" if r.get("confiance") == "moyenne" else "🟢")
        rel_desc = RELATIONS_RST.get(r["relation"], ("?", "?"))[1]
        lignes.append(
            f"| `{r.get('noyau','?')}` | `{r.get('relation','?')}` "
            f"| `{r.get('satellite','?')}` | {conf_sym} {r.get('confiance','?')} |"
        )
    lignes += ["", "---", "", "## Arbre RST", ""]
    lignes.append(mermaid)
    lignes += [
        "",
        "> ★ = noyau central · Arêtes pleines = confiance haute/moyenne · "
        "Arêtes pointillées = confiance basse",
        "",
        "---",
        "",
        "## Structure d'ensemble",
        "",
        structure if structure else "*Non disponible.*",
        "",
        "---",
        "",
        "## Synthèse RST",
        "",
    ]
    for l in synthese.split("\n"):
        l = l.strip()
        if l:
            lignes.append(l)
    lignes += [
        "", "---", "",
        "*Rapport généré par `C_rst_segment.py`.*  ",
        "*Cadre : Mann & Thompson (1988), Rhetorical Structure Theory.*  ",
        "*Granularité : phrase. Statut : approximation heuristique.*",
    ]

    contenu = "\n".join(lignes)
    chemin  = output_dir / f"rst_{timestamp}.md"
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


def generer_json(
    titre: str, texte: str, question: str,
    raw: str, timestamp: str, output_dir: Path,
) -> Path:
    data = {
        "script"     : "C_rst_segment",
        "timestamp"  : timestamp,
        "modele"     : OPENAI_LLM_MODEL,
        "titre"      : titre,
        "question"   : question,
        "nb_chars"   : len(texte),
        "phrases"    : extraire_phrases(raw),
        "noyau"      : extraire_noyau(raw),
        "relations"  : extraire_relations(raw),
        "structure"  : extraire_structure(raw),
        "synthese"   : extraire_synthese(raw),
    }
    chemin = output_dir / f"rst_{timestamp}.json"
    chemin.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="C_rst_segment — Analyse RST (Mann & Thompson) sur segment."
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

    print(f"\n  Analyse RST en cours ({OPENAI_LLM_MODEL}, T°={TEMPERATURE}, max_tokens={MAX_TOKENS})…")
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

    phrases   = extraire_phrases(raw)
    relations = extraire_relations(raw)
    noyau     = extraire_noyau(raw)
    n_basse   = sum(1 for r in relations if r.get("confiance") == "basse")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md   = generer_rapport_md(titre, texte, question, raw, timestamp, output_dir)
    chemin_json = generer_json(titre, texte, question, raw, timestamp, output_dir)

    print(f"\n{'═'*60}")
    print(f"  BILAN RST — {titre[:50]}")
    print(f"{'═'*60}")
    print(f"  Phrases identifiées     : {len(phrases)}")
    print(f"  Relations identifiées   : {len(relations)}")
    print(f"  Noyau central           : {noyau}")
    print(f"  Relations faible conf.  : {n_basse}")
    if n_basse > len(relations) * 0.4:
        print(f"  ⚠ Plus de 40% des relations ont une confiance basse.")
        print(f"    L'analyse est peu fiable sur ce segment.")
    print(f"{'═'*60}")
    print(f"\n✅ MD   : {chemin_md}")
    print(f"   JSON : {chemin_json}")
    print(f"\n   Lancez D_synthese_croisee.py pour l'analyse intégrée A+B+C.\n")


if __name__ == "__main__":
    main()
