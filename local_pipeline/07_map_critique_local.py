"""
07_map_critique_local.py
========================
Étape 7 du pipeline RAG documentaire — VERSION 100 % LOCALE.

Rôle : Cartographie critique. Analyse les relations entre chaque section
du manuscrit et les passages du corpus identifiés par 06_map_enrich_local.py.
Produit un rapport critique structuré en JSON, avec scores et taxonomie.

Pipeline :
    …
    06_map_enrich_local.py   → map_{timestamp}.json
    07_map_critique_local.py → critique_{timestamp}.json (ce script)
    08_visualise.py          → visualisation des scores

TAXONOMIE DES RELATIONS (six catégories) :
┌─────────────────┬──────────────────────────────────────────────────────┐
│ conforte        │ Le passage corpus confirme, corrobore ou renforce     │
│                 │ l'argument du manuscrit avec des preuves convergentes │
├─────────────────┼──────────────────────────────────────────────────────┤
│ contredit       │ Le passage corpus s'oppose directement à l'affirmation│
│                 │ du manuscrit : faits, dates, interprétations opposés  │
├─────────────────┼──────────────────────────────────────────────────────┤
│ nuance          │ Le passage apporte des distinctions ou des cas limites │
│                 │ qui complexifient sans invalider l'argument            │
├─────────────────┼──────────────────────────────────────────────────────┤
│ problématise    │ Le passage soulève une question, une tension ou une   │
│                 │ difficulté théorique que le manuscrit ne traite pas    │
├─────────────────┼──────────────────────────────────────────────────────┤
│ déplace         │ Le passage repositionne l'argument dans un autre cadre │
│                 │ (chronologique, géographique, épistémologique)         │
├─────────────────┼──────────────────────────────────────────────────────┤
│ particularise   │ Le passage illustre, détaille ou exemplifie un aspect  │
│                 │ général de l'argument par un cas spécifique            │
└─────────────────┴──────────────────────────────────────────────────────┘

FORMAT DE SORTIE JSON (trois niveaux) :
    {
      "score_global": 0.72,               ← score pondéré agrégé [0–1]
      "profil_dominant": "conforte",      ← catégorie majoritaire
      "sections": [
        {
          "id": 1,
          "titre": "…",
          "score_section": 0.68,          ← score agrégé de la section [0–1]
          "profil_section": "nuance",     ← catégorie majoritaire de la section
          "scores_taxonomie": {           ← six scores [0–1] pour la section
            "conforte"    : 0.30,
            "contredit"   : 0.05,
            "nuance"      : 0.40,
            "problématise": 0.10,
            "déplace"     : 0.05,
            "particularise": 0.10
          },
          "passages": [
            {
              "source"    : "fichier.pdf",
              "page"      : 3,
              "relation"  : "nuance",
              "score"     : 0.72,         ← confiance [0–1]
              "justification": "Ce passage montre que…",
              "segments"  : ["texte manuscrit concerné", "texte corpus concerné"]
            }
          ]
        }
      ]
    }

NOTE SUR LES SCORES :
    Les scores [0–1] expriment la CONFIANCE du LLM dans sa classification,
    pas une mesure de similarité vectorielle.
    0.0 → relation incertaine ou absente
    0.5 → relation probable mais ambiguë
    1.0 → relation nette et bien documentée
    Ces scores sont subjectifs et dépendent du modèle LLM utilisé.
    Ils doivent être interprétés comme des indicateurs orientateurs,
    pas comme des mesures objectives.

Usage :
    python 07_map_critique_local.py map_20240101_120000.json
    python 07_map_critique_local.py   # utilise le fichier map le plus récent

Prérequis :
    pip install sentence-transformers faiss-cpu numpy requests
    ollama serve && ollama pull qwen2.5:14b
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

from rag_config_local import (
    OLLAMA_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT_CRITIQUE,
)

# Dossier contenant les fichiers map produits par 06_map_enrich_local.py
MAP_DIR    = "outputs"
OUTPUT_DIR = "outputs"

# Température très basse pour la classification : on veut de la cohérence,
# pas de la créativité.
TEMPERATURE_CRITIQUE = 0.05

# Nombre maximum de passages analysés par section.
# Augmenter pour une analyse exhaustive, réduire pour accélérer.
# Les passages au-delà de ce seuil (les moins proches vectoriellement)
# sont inclus dans le JSON mais pas soumis au LLM.
MAX_PASSAGES_PER_SECTION = 8

# Longueur maximale de l'extrait manuscrit soumis au LLM par section (chars)
MAX_SECTION_CHARS = 1200

# Longueur maximale de l'extrait corpus soumis au LLM par passage (chars)
MAX_PASSAGE_CHARS = 800

# Seuil de distance L2 au-delà duquel un passage est jugé hors-sujet
# et exclu de l'analyse critique.
# En local (paraphrase-multilingual-mpnet-base-v2), les scores L2 typiques :
#   < 2.0 : passage très pertinent
#   2.0–3.5 : pertinence acceptable
#   > 3.5 : passage probablement hors-sujet — exclu par défaut
# Mettre à None pour désactiver le filtre.
DISTANCE_THRESHOLD = 3.5

# Pondérations pour le score_global agrégé.
# Ces poids reflètent l'importance accordée à chaque relation dans
# l'évaluation globale du manuscrit par rapport au corpus.
# La somme doit être égale à 1.0.
WEIGHTS_TAXONOMIE = {
    "conforte"     : 0.20,
    "contredit"    : 0.25,   # poids élevé : la contradiction est critique
    "nuance"       : 0.20,
    "problématise" : 0.15,
    "déplace"      : 0.10,
    "particularise": 0.10,
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import sys
import requests
from pathlib import Path
from datetime import datetime

# =============================================================================
# INITIALISATION
# =============================================================================

map_dir    = Path(MAP_DIR)
output_dir = Path(OUTPUT_DIR)

TAXONOMIE = list(WEIGHTS_TAXONOMIE.keys())

# =============================================================================
# FONCTIONS — CHARGEMENT
# =============================================================================

def find_latest_map() -> Path:
    """
    Trouve le fichier map JSON le plus récent dans MAP_DIR.

    Returns:
        Chemin vers le fichier map le plus récent.

    Raises:
        FileNotFoundError : si aucun fichier map n'est trouvé.
    """
    maps = sorted(map_dir.glob("map_*.json"), reverse=True)
    # Exclure les fichiers critique (qui commencent aussi parfois par map_)
    maps = [m for m in maps if "critique" not in m.name]
    if not maps:
        raise FileNotFoundError(
            f"Aucun fichier map_*.json dans {map_dir.resolve()}\n"
            "Lancez d'abord 06_map_enrich_local.py."
        )
    return maps[0]


def load_map(map_file: Path) -> dict:
    """
    Charge un fichier map JSON produit par 06_map_enrich_local.py.

    Args:
        map_file : Chemin vers le fichier JSON.

    Returns:
        Dictionnaire structuré avec "sections" et leurs "passages_corpus".
    """
    if not map_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {map_file.resolve()}")
    return json.loads(map_file.read_text(encoding="utf-8"))


# =============================================================================
# FONCTIONS — ANALYSE CRITIQUE
# =============================================================================

def build_critique_prompt(
    section_titre: str,
    section_texte: str,
    passages: list[dict],
) -> str:
    """
    Construit le prompt de classification critique pour une section.

    Le prompt demande au LLM d'analyser chaque passage et de :
    1. Classer la relation selon la taxonomie des six catégories.
    2. Attribuer un score de confiance [0–1].
    3. Fournir une justification en une phrase.
    4. Identifier les segments concernés.

    Le format de sortie imposé (JSON strict) facilite le parsing automatique.

    Args:
        section_titre : Titre de la section du manuscrit.
        section_texte : Texte de la section (tronqué à MAX_SECTION_CHARS).
        passages      : Liste de dicts {source, page, extrait} à analyser.

    Returns:
        Prompt complet prêt à être soumis à Ollama.
    """
    extraits_corpus = []
    for i, p in enumerate(passages, 1):
        extraits_corpus.append(
            f"Passage {i} [{p['source']}, p. {p['page']}] :\n"
            f"{p['extrait'][:MAX_PASSAGE_CHARS]}"
        )

    extraits_str = "\n\n---\n\n".join(extraits_corpus)

    taxonomie_str = "\n".join(
        f"  - {cat}" for cat in TAXONOMIE
    )

    prompt = f"""{SYSTEM_PROMPT_CRITIQUE}

TAXONOMIE DES RELATIONS (utilise UNIQUEMENT ces termes) :
{taxonomie_str}

SECTION DU MANUSCRIT — « {section_titre} » :
{section_texte[:MAX_SECTION_CHARS]}

PASSAGES DU CORPUS À ANALYSER :
{extraits_str}

TÂCHE EN DEUX PARTIES :

━━━ PARTIE 1 — CLASSIFICATION JSON ━━━
Pour chaque passage numéroté, analyse sa relation avec la section du manuscrit.

RÈGLES POUR LA JUSTIFICATION (impératives) :
- Rédige 2 à 3 phrases en français.
- Intègre obligatoirement une citation directe et complète du corpus entre guillemets
  (1 à 2 phrases, compréhensibles sans contexte, non tronquées).
- Explique ensuite en quoi cette citation établit la relation identifiée avec le manuscrit.
- Ne commence jamais par "Ce passage" — varie les formulations.

Réponds d'abord avec un JSON valide (sans texte avant ni après), sans backticks Markdown :
[
  {{
    "passage_num"   : 1,
    "source"        : "fichier.pdf",
    "page"          : 3,
    "relation"      : "nuance",
    "score"         : 0.75,
    "justification" : "2-3 phrases avec citation directe du corpus intégrée."
  }},
  ...
]

━━━ PARTIE 2 — SYNTHÈSE ━━━
Après le JSON, sur une nouvelle ligne, produis :

SYNTHESE: En 2-3 phrases : quel est le rapport dominant entre cette section et le corpus ?
Quelle est la priorité de révision pour l'historien ? Si les passages sont hors-sujet,
dis-le clairement plutôt que de forcer des relations artificielles.

JSON :"""

    return prompt


def call_ollama_critique(prompt: str) -> str:
    """
    Soumet le prompt critique à Ollama (sans streaming pour parsing JSON).

    Sans streaming : on attend la réponse complète pour pouvoir la parser
    comme du JSON sans interruption.

    Args:
        prompt : Prompt de classification critique.

    Returns:
        Réponse brute du LLM (JSON attendu).

    Raises:
        SystemExit : Si Ollama n'est pas joignable.
    """
    payload = {
        "model"       : LLM_MODEL,
        "prompt"      : prompt,
        "temperature" : TEMPERATURE_CRITIQUE,
        "num_predict" : LLM_MAX_TOKENS,
        "stream"      : False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Ollama inaccessible. Lancez : ollama serve\n   URL : {OLLAMA_URL}")
        sys.exit(1)


def parse_critique_response(raw: str, passages: list[dict]) -> list[dict]:
    """
    Parse la réponse JSON du LLM et valide les champs attendus.

    En cas de réponse mal formée (JSON invalide, champs manquants), produit
    un résultat de repli avec relation "problématise" et score 0.0, plutôt
    que de faire planter le pipeline entier.

    Les scores sont bornés à [0, 1] par précaution.

    Args:
        raw      : Réponse brute du LLM.
        passages : Passages originaux (pour les métadonnées de repli).

    Returns:
        Liste de dicts validés, un par passage analysé.
    """
    # Nettoyage des backticks Markdown éventuels
    raw = raw.strip()
    for marker in ("```json", "```"):
        raw = raw.replace(marker, "")
    raw = raw.strip()

    # Isolation du bloc JSON : on extrait uniquement ce qui se trouve
    # entre le premier "[" et le "]" fermant correspondant, pour ignorer
    # le texte SYNTHESE: … que le LLM ajoute après (cause de "Extra data").
    import re as _re_parse
    # Cherche le dernier "]" avant un éventuel "SYNTHESE"
    synthese_pos = _re_parse.search(r"\n\s*SYNTHESE", raw, _re_parse.IGNORECASE)
    if synthese_pos:
        raw = raw[:synthese_pos.start()].strip()
    # Si le LLM a quand même laissé du texte après le "]" final, on coupe là
    last_bracket = raw.rfind("]")
    if last_bracket != -1:
        raw = raw[:last_bracket + 1].strip()

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("La réponse n'est pas une liste JSON.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠ Parsing JSON échoué ({e}). Repli sur valeurs par défaut.")
        return [
            {
                "source"        : p["source"],
                "page"          : p["page"],
                "relation"      : "problématise",
                "score"         : 0.0,
                "justification" : "Analyse non disponible (erreur de parsing).",
            }
            for p in passages
        ]

    validated = []
    for item in items:
        relation = item.get("relation", "problématise")
        if relation not in TAXONOMIE:
            relation = "problématise"   # valeur de repli si catégorie inconnue

        score = float(item.get("score", 0.0))
        score = max(0.0, min(1.0, score))   # borne à [0, 1]

        validated.append({
            "source"        : item.get("source", "?"),
            "page"          : item.get("page", 0),
            "relation"      : relation,
            "score"         : round(score, 3),
            "justification" : item.get("justification", ""),
        })

    return validated


# =============================================================================
# FONCTIONS — CALCUL DES SCORES AGRÉGÉS
# =============================================================================

def compute_section_scores(passages_analysés: list[dict]) -> dict:
    """
    Calcule les six scores taxonomiques et le score agrégé d'une section.

    Méthode de calcul :
    - Pour chaque catégorie, on somme les scores des passages classés
      dans cette catégorie, puis on normalise par le nombre total de passages.
    - Le score_section est la moyenne pondérée des six scores selon WEIGHTS_TAXONOMIE.

    Ce calcul produit des valeurs [0, 1] interprétables :
    - score_section proche de 1 → le corpus est fortement en relation avec la section
    - profil_dominant → nature dominante de cette relation

    Args:
        passages_analysés : Liste de dicts {relation, score, …}.

    Returns:
        Dict avec "scores_taxonomie", "score_section", "profil_dominant".
    """
    n = len(passages_analysés)
    if n == 0:
        return {
            "scores_taxonomie": {cat: 0.0 for cat in TAXONOMIE},
            "score_section"   : 0.0,
            "profil_dominant" : "indéterminé",
        }

    # Accumulation des scores par catégorie
    accum = {cat: 0.0 for cat in TAXONOMIE}
    for p in passages_analysés:
        cat = p["relation"]
        if cat in accum:
            accum[cat] += p["score"]

    # Normalisation par le nombre de passages
    scores_normalisés = {cat: round(v / n, 3) for cat, v in accum.items()}

    # Score global pondéré de la section
    score_section = sum(
        scores_normalisés[cat] * WEIGHTS_TAXONOMIE[cat]
        for cat in TAXONOMIE
    )
    score_section = round(score_section, 3)

    # Catégorie dominante (score normalisé le plus élevé)
    profil_dominant = max(scores_normalisés, key=lambda k: scores_normalisés[k])

    return {
        "scores_taxonomie": scores_normalisés,
        "score_section"   : score_section,
        "profil_dominant" : profil_dominant,
    }


def compute_global_score(sections_critiques: list[dict]) -> tuple[float, str]:
    """
    Calcule le score global et le profil dominant sur l'ensemble du manuscrit.

    Méthode : moyenne des score_section, puis profil dominant
    comme catégorie dont le score moyen normalisé est le plus élevé.

    Args:
        sections_critiques : Sections avec leurs scores calculés.

    Returns:
        Tuple (score_global, profil_dominant_global).
    """
    if not sections_critiques:
        return 0.0, "indéterminé"

    # Moyenne des scores de sections
    score_global = round(
        sum(s["score_section"] for s in sections_critiques) / len(sections_critiques),
        3,
    )

    # Profil dominant global : moyenne des scores taxonomiques par catégorie
    moyennes = {cat: 0.0 for cat in TAXONOMIE}
    n = len(sections_critiques)
    for sec in sections_critiques:
        for cat in TAXONOMIE:
            moyennes[cat] += sec["scores_taxonomie"].get(cat, 0.0)
    moyennes = {cat: v / n for cat, v in moyennes.items()}

    profil_global = max(moyennes, key=lambda k: moyennes[k])

    return score_global, profil_global



# =============================================================================
# FONCTIONS — RAPPORT MARKDOWN
# =============================================================================

def _barre(valeur: float, largeur: int = 20) -> str:
    """Barre ASCII proportionnelle. Ex : [████████░░░░░░░░░░░░] 0.42"""
    rempli = round(valeur * largeur)
    return f"[{'█' * rempli}{'░' * (largeur - rempli)}] {valeur:.2f}"


# Émojis de relation pour rendre le MD plus scannable visuellement
RELATION_EMOJI = {
    "conforte"     : "✔",
    "contredit"    : "✗",
    "nuance"       : "◑",
    "problématise" : "?",
    "déplace"      : "→",
    "particularise": "◎",
}


def generer_rapport_md(
    map_data: dict,
    sections_critiques: list[dict],
    score_global: float,
    profil_global: str,
    timestamp: str,
) -> Path:
    """
    Génère critique_{timestamp}.md — version lisible du rapport critique.

    Produit en parallèle du JSON par save_critique(). C'est ce fichier qui
    est destiné à la lecture humaine directe ; le JSON reste pour les scripts
    aval (visualisation, analyse croisée).

    Structure par section :
      - Titre et extrait manuscrit
      - Profil dominant + score section
      - Tableau des six scores taxonomiques avec barres ASCII
      - Liste des passages analysés avec :
          relation (émoji + label) | score confiance | source + page
          justification du LLM en clair (c'est la valeur principale)

    TAXONOMIE DES SIX RELATIONS :
      ✔ conforte      — le corpus confirme et renforce l'argument
      ✗ contredit     — le corpus s'oppose directement à l'affirmation
      ◑ nuance        — le corpus complexifie sans invalider
      ? problématise  — le corpus soulève une tension non traitée
      → déplace       — le corpus repositionne l'argument dans un autre cadre
      ◎ particularise — le corpus illustre par un cas spécifique

    Args:
        map_data           : Métadonnées de la carte source (06).
        sections_critiques : Sections avec scores et passages analysés.
        score_global       : Score agrégé global.
        profil_global      : Profil dominant global.
        timestamp          : Horodatage de la session.

    Returns:
        Chemin du fichier MD produit.
    """
    source    = map_data.get("manuscrit_source", "?")
    date_run  = map_data.get("date", "")
    n_sec     = len(sections_critiques)

    lignes = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lignes += [
        "# Rapport critique — manuscrit × corpus",
        "",
        f"**Source manuscrit** : `{source}`  ",
        f"**Date d'analyse** : {date_run}  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle LLM** : {LLM_MODEL}  ",
        f"**Température** : {TEMPERATURE_CRITIQUE}  ",
        f"**Sections analysées** : {n_sec}  ",
        "",
        "---",
        "",
        "## Bilan global",
        "",
        f"| Métrique | Valeur |",
        f"|---|---|",
        f"| Score global | **{score_global:.3f}** |",
        f"| Profil dominant | `{profil_global}` |",
        "",
    ]

    # Tableau récapitulatif
    lignes += [
        "| § | Titre | Score | Profil | Conforte | Contredit | Nuance | Problématise | Déplace | Particularise |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sec in sections_critiques:
        st = sec.get("scores_taxonomie", {})
        lignes.append(
            f"| {sec['id']} "
            f"| {sec['titre'][:40]} "
            f"| **{sec['score_section']:.2f}** "
            f"| `{sec['profil_dominant']}` "
            f"| {st.get('conforte', 0):.2f} "
            f"| {st.get('contredit', 0):.2f} "
            f"| {st.get('nuance', 0):.2f} "
            f"| {st.get('problématise', 0):.2f} "
            f"| {st.get('déplace', 0):.2f} "
            f"| {st.get('particularise', 0):.2f} |"
        )
    lignes += ["", "---", ""]

    # ── Sections détaillées ───────────────────────────────────────────────────
    lignes.append("## Analyse par section\n")

    for sec in sections_critiques:
        st      = sec.get("scores_taxonomie", {})
        profil  = sec.get("profil_dominant", "?")
        score_s = sec.get("score_section", 0)
        emoji   = RELATION_EMOJI.get(profil, "·")
        texte   = sec.get("texte", "")

        lignes += [
            f"### § {sec['id']} — {sec['titre']}",
            "",
        ]

        if texte:
            lignes += [
                "**Extrait manuscrit :**",
                "",
                f"> {texte[:300].replace(chr(10), ' ')}{'…' if len(texte) > 300 else ''}",
                "",
            ]

        # Score section + profil
        lignes += [
            f"**Score section** : `{score_s:.3f}` &nbsp;|&nbsp; "
            f"**Profil dominant** : {emoji} `{profil}`",
            "",
            "**Scores taxonomiques :**",
            "",
        ]

        # Barres ASCII pour les six catégories
        for cat in TAXONOMIE:
            v     = st.get(cat, 0)
            em    = RELATION_EMOJI.get(cat, "·")
            lignes.append(f"- {em} **{cat:<14}** `{_barre(v)}`")
        lignes.append("")

        # Passages analysés
        passages = sec.get("passages", [])
        if not passages:
            lignes += ["*Aucun passage analysé.*", "", "---", ""]
            continue

        lignes += [
            f"**{len(passages)} passage(s) analysé(s) :**",
            "",
        ]

        for i, p in enumerate(passages, 1):
            rel    = p.get("relation", "?")
            score  = p.get("score", 0)
            src    = p.get("source", "?")
            page   = p.get("page", "?")
            justif = p.get("justification", "").strip()
            em     = RELATION_EMOJI.get(rel, "·")

            lignes += [
                f"**{i}.** {em} `{rel}` — score : `{score:.2f}` &nbsp;|&nbsp; "
                f"`{src}` p. {page}",
                "",
            ]
            if justif:
                # Indentation de la justification pour la distinguer visuellement
                for ligne in justif.split("\n"):
                    ligne = ligne.strip()
                    if ligne:
                        lignes.append(f"> {ligne}")
                lignes.append("")

        lignes += ["---", ""]

        # Synthèse de la section
        synthese = sec.get("synthese", "").strip()
        if synthese:
            lignes += ["### Synthèse", ""]
            lignes.append(synthese)
            lignes += ["", "---", ""]

    # ── Pied de page ──────────────────────────────────────────────────────────
    lignes += [
        "",
        "*Rapport généré automatiquement par `07_map_critique_local.py`.*  ",
        "*Taxonomie : conforte · contredit · nuance · problématise · déplace · particularise.*  ",
        f"*Pondérations : contredit×{WEIGHTS_TAXONOMIE['contredit']} "
        f"· conforte×{WEIGHTS_TAXONOMIE['conforte']} "
        f"· nuance×{WEIGHTS_TAXONOMIE['nuance']} "
        f"· problématise×{WEIGHTS_TAXONOMIE['problématise']}*",
    ]

    contenu   = "\n".join(lignes)
    chemin_md = output_dir / f"critique_{timestamp}.md"
    chemin_md.write_text(contenu, encoding="utf-8")
    return chemin_md


# =============================================================================
# FONCTIONS — SAUVEGARDE
# =============================================================================

def save_critique(map_data: dict, sections_critiques: list[dict], score_global: float, profil_global: str) -> tuple:
    """
    Sauvegarde le rapport critique en JSON (trois niveaux).

    Niveau 1 : score_global + profil_dominant (vue d'ensemble du manuscrit)
    Niveau 2 : score_section + profil_section par section
    Niveau 3 : passages individuels avec relation, score, justification, segments

    Args:
        map_data           : Données de la carte source (06).
        sections_critiques : Sections avec scores calculés.
        score_global       : Score agrégé global.
        profil_global      : Profil dominant global.

    Returns:
        Tuple (chemin_json, chemin_md).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        # Niveau 1 — vue d'ensemble
        "score_global"     : score_global,
        "profil_dominant"  : profil_global,
        "manuscrit_source" : map_data.get("manuscrit_source", "?"),
        "date"             : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modele_llm"       : LLM_MODEL,
        "poids_taxonomie"  : WEIGHTS_TAXONOMIE,
        # Niveaux 2 et 3
        "sections"         : [
            {
                "id"              : sec["id"],
                "titre"           : sec["titre"],
                # Niveau 2
                "score_section"   : sec["score_section"],
                "profil_section"  : sec["profil_dominant"],
                "scores_taxonomie": sec["scores_taxonomie"],
                # Niveau 3
                "passages"        : sec["passages"],
                # Synthèse
                "synthese"        : sec.get("synthese", ""),
            }
            for sec in sections_critiques
        ],
    }

    output_path = output_dir / f"critique_{timestamp}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Rapport Markdown lisible en parallèle du JSON
    md_path = generer_rapport_md(
        map_data, sections_critiques, score_global, profil_global, timestamp
    )

    return output_path, md_path


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="07 : cartographie critique du manuscrit (version locale)"
    )
    parser.add_argument(
        "map_file",
        nargs="?",
        help="Fichier map JSON produit par 06_map_enrich_local.py. "
             "Si absent, utilise le fichier le plus récent dans outputs/.",
    )
    args = parser.parse_args()

    # Chargement du fichier map
    if args.map_file:
        map_file = Path(args.map_file)
        if not map_file.is_absolute():
            map_file = map_dir / map_file
    else:
        print("Aucun fichier map spécifié, recherche du plus récent…")
        map_file = find_latest_map()

    print(f"Chargement de la carte : {map_file.name}")
    map_data = load_map(map_file)
    sections = map_data.get("sections", [])
    print(f"  {len(sections)} sections à analyser.\n")
    print(f"Modèle LLM : {LLM_MODEL} | T° : {TEMPERATURE_CRITIQUE}\n")

    # Analyse critique section par section
    sections_critiques = []

    for sec in sections:
        print(f"§ {sec['id']} — {sec['titre']}")

        passages_bruts = sec.get("passages_corpus", [])

        # Filtre par distance L2 — exclut les passages trop éloignés sémantiquement
        if DISTANCE_THRESHOLD is not None:
            passages_bruts = [
                p for p in passages_bruts
                if p.get("score_L2", 999) <= DISTANCE_THRESHOLD
            ]
            if not passages_bruts:
                print(f"  Tous les passages dépassent le seuil L2 ({DISTANCE_THRESHOLD}) — section ignorée.\n")
                continue

        passages_bruts = passages_bruts[:MAX_PASSAGES_PER_SECTION]

        if not passages_bruts:
            print("  Aucun passage corpus — section ignorée.\n")
            continue

        # Classification critique via LLM
        prompt  = build_critique_prompt(sec["titre"], sec["texte"], passages_bruts)
        raw     = call_ollama_critique(prompt)
        analysés = parse_critique_response(raw, passages_bruts)

        # Extraction de la synthèse
        import re as _re
        synthese = ""
        m = _re.search(r"SYNTHESE\s*:\s*(.+?)(?:\n\n|\Z)", raw, _re.DOTALL | _re.IGNORECASE)
        if m:
            synthese = m.group(1).strip()

        # Calcul des scores
        scores = compute_section_scores(analysés)

        print(f"  Score section : {scores['score_section']:.3f} | "
              f"Profil : {scores['profil_dominant']}")
        for cat in TAXONOMIE:
            v = scores["scores_taxonomie"][cat]
            bar = "█" * int(v * 20)
            print(f"    {cat:<14} {v:.3f}  {bar}")
        if synthese:
            print(f"  Synthèse : {synthese[:100]}…")

        sections_critiques.append({
            "id"              : sec["id"],
            "titre"           : sec["titre"],
            "texte"           : sec.get("texte", ""),
            "passages"        : analysés,
            "synthese"        : synthese,
            **scores,
        })
        print()

    if not sections_critiques:
        print("❌ Aucune section analysée. Vérifiez le fichier map.")
        sys.exit(1)

    # Score global
    score_global, profil_global = compute_global_score(sections_critiques)

    print(f"{'═'*60}")
    print(f"SCORE GLOBAL : {score_global:.3f} | PROFIL : {profil_global}")
    print(f"{'═'*60}\n")

    # Sauvegarde
    output_path, md_path = save_critique(map_data, sections_critiques, score_global, profil_global)
    print(f"✓ Rapport critique sauvegardé.")
    print(f"  JSON : {output_path.resolve()}")
    print(f"  MD   : {md_path.resolve()}")
    print(f"\n  Étape suivante : python 08_visualise.py {output_path.name}")


if __name__ == "__main__":
    main()
