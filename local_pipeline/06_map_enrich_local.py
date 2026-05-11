"""
06_map_enrich_local.py
======================
Étape 6 du pipeline RAG documentaire — VERSION 100 % LOCALE.

Rôle : Cartographie enrichie du manuscrit. Pour chaque section ou argument
clé du manuscrit fourni, recherche dans le corpus les passages qui
l'éclairent, le corroborent, le nuancent ou le contredisent, et produit
une carte structurée des relations entre le manuscrit et les sources.
Cette carte est une entrée pour le script 07, le rapport produit en format md
pour en faciliter la lisibilité n'a pas d'autre fonction que de se prononcer sur 
le fonctionnement du script, il n'est pas une sortie actionnable par l'historien.
La sortie destinée à celui ci est le fichier au format md, produit par le script 7.

Pipeline :
    …
    05_rag_write_local.py    → passage rédigé
    06_map_enrich_local.py   → carte enrichie du manuscrit (ce script)
    07_map_critique_local.py → rapport critique structuré (JSON)
    08_visualise.py          → visualisation

Entrée :
    Un fichier texte contenant le manuscrit (ou des extraits) à cartographier.
    Par défaut : manuscript.txt à la racine du projet.
    Le manuscrit peut être en FR, EN, DE ou IT — le modèle multilingue
    gère tous ces cas sans configuration supplémentaire.

Sortie :
    Un fichier JSON structuré : map_{timestamp}.json
    et un rapport texte lisible : map_{timestamp}_rapport.txt

Format de sortie JSON :
    {
      "manuscrit_source": "chemin/vers/manuscrit.txt",
      "date": "2024-...",
      "sections": [
        {
          "id": 1,
          "titre": "Section ou argument identifié",
          "texte_extrait": "...",
          "passages_corpus": [
            {
              "source": "fichier.pdf",
              "page": 3,
              "score_L2": 42.3,
              "extrait": "...",
              "relation_pressentie": "à déterminer par 07_map_critique_local.py"
            },
            …
          ]
        },
        …
      ]
    }

Prérequis :
    pip install sentence-transformers faiss-cpu numpy requests
    ollama serve && ollama pull qwen2.5:14b
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

from rag_config_local import (
    VECTOR_DIR,
    CORPUS_FILE,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

# Fichier manuscrit à cartographier
MANUSCRIPT_FILE = "manuscript.txt"

# Dossier de sortie des cartes
OUTPUT_DIR = "outputs"

# Nombre de passages corpus à associer à chaque section du manuscrit
TOP_K_MAP = 6

# Longueur maximale d'un extrait dans la carte JSON (caractères)
# Tronquer les passages trop longs pour garder un JSON lisible
MAX_EXCERPT_CHARS = 600

# Demander au LLM d'identifier automatiquement les sections du manuscrit ?
# True  : le LLM segmente le manuscrit (utile pour les textes non structurés)
# False : segmentation par paragraphes (plus rapide, moins fin)
AUTO_SEGMENT = False

# Nombre maximum de sections à cartographier (None = toutes)
# Limiter pour tester sur un extrait avant de traiter le manuscrit entier
MAX_SECTIONS = 5   # ex : 5 pour un test

# Seuil de distance L2 au-delà duquel un passage est jugé hors-sujet
DISTANCE_THRESHOLD = None

# =============================================================================
# IMPORTS
# =============================================================================

import json
import sys
import argparse
import numpy as np
import faiss
import requests
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =============================================================================
# INITIALISATION
# =============================================================================

vector_path   = Path(VECTOR_DIR)
index_file    = vector_path / "faiss.index"
metadata_file = vector_path / "metadata.json"
corpus_path   = Path(CORPUS_FILE)
output_dir    = Path(OUTPUT_DIR)

# =============================================================================
# FONCTIONS — CHARGEMENT
# =============================================================================

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """Charge l'index FAISS et les métadonnées."""
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(f"Introuvable : {path.resolve()}")
    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def load_passage(source: str, page: int) -> str:
    """Récupère le passage complet depuis corpus.txt."""
    if not corpus_path.exists():
        return ""
    marker = f"[{source} — page {page}]"
    text   = corpus_path.read_text(encoding="utf-8")
    start  = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end    = text.find("[", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()


def load_manuscript(path: str) -> str:
    """
    Charge le manuscrit depuis un fichier texte.

    Le manuscrit peut être en n'importe quelle langue (FR, EN, DE, IT).
    Le modèle d'embeddings multilingue gère la recherche cross-lingue :
    une section en allemand peut trouver des passages pertinents en français.

    Args:
        path : Chemin vers le fichier texte du manuscrit.

    Returns:
        Contenu textuel du manuscrit.

    Raises:
        FileNotFoundError : si le fichier est absent.
    """
    manuscript_path = Path(path)
    if not manuscript_path.exists():
        raise FileNotFoundError(
            f"Manuscrit introuvable : {manuscript_path.resolve()}\n"
            f"Créez le fichier {path} ou modifiez MANUSCRIPT_FILE."
        )
    return manuscript_path.read_text(encoding="utf-8")


# =============================================================================
# FONCTIONS — SEGMENTATION
# =============================================================================

def segment_by_paragraphs(text: str) -> list[dict]:
    """
    Segmente le manuscrit en sections par paragraphes.

    Segmentation par regex flexible : accepte les doubles sauts de ligne
    même si des espaces ou tabulations s'y intercalent (fréquent dans les
    exports Word/PDF). Filtre les paragraphes trop courts (< 150 caractères)
    pour éliminer les en-têtes courants, numéros de page et artefacts de
    mise en forme.

    Args:
        text : Texte brut du manuscrit.

    Returns:
        Liste de dicts {"id": N, "titre": "…", "texte": "…"}.
    """
    import re
    # Regex flexible : 1 saut de ligne + éventuels espaces + 1 saut de ligne
    separateur = r"\n\s*\n"
    paragraphs = [p.strip() for p in re.split(separateur, text) if p.strip()]
    sections   = []

    for i, para in enumerate(paragraphs, 1):
        if len(para) < 150:
            continue   # filtre en-têtes, numéros de page, titres isolés
        sections.append({
            "id"    : i,
            "titre" : para[:80].replace("\n", " ") + ("…" if len(para) > 80 else ""),
            "texte" : para,
        })

    return sections


def segment_by_llm(text: str) -> list[dict]:
    """
    Demande au LLM d'identifier les sections et arguments du manuscrit.

    Cette approche est plus fine que la segmentation par paragraphes :
    le LLM identifie les unités sémantiques (arguments, thèses, transitions)
    plutôt que les unités typographiques.

    Le LLM retourne une liste JSON de sections avec titre et extrait.

    Args:
        text : Texte brut du manuscrit.

    Returns:
        Liste de dicts {"id": N, "titre": "…", "texte": "…"}.
        En cas d'erreur de parsing JSON, repli sur la segmentation par paragraphes.
    """
    prompt = f"""Tu es un historien expert en analyse de manuscrits.
Identifie les sections, arguments ou thèses principales du texte ci-dessous.
Pour chaque section, extrais un titre court (< 10 mots) et le passage correspondant.

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ni après, de la forme :
[
  {{"titre": "Titre de la section 1", "texte": "Extrait de la section 1..."}},
  {{"titre": "Titre de la section 2", "texte": "Extrait de la section 2..."}}
]

TEXTE DU MANUSCRIT :
{text[:6000]}

JSON :"""

    payload = {
        "model"       : LLM_MODEL,
        "prompt"      : prompt,
        "temperature" : 0.1,
        "num_predict" : 2000,
        "stream"      : False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw  = resp.json().get("response", "").strip()

        # Nettoyage des backticks Markdown si présents
        raw = raw.strip("```json").strip("```").strip()

        sections_raw = json.loads(raw)
        sections = [
            {"id": i, "titre": s.get("titre", f"Section {i}"), "texte": s.get("texte", "")}
            for i, s in enumerate(sections_raw, 1)
            if s.get("texte", "").strip()
        ]
        return sections

    except (requests.exceptions.ConnectionError, json.JSONDecodeError) as e:
        print(f"  ⚠ Segmentation LLM échouée ({e}), repli sur paragraphes.")
        return []


# =============================================================================
# FONCTIONS — RECHERCHE ET CARTOGRAPHIE
# =============================================================================

def search_for_section(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    section_text: str,
) -> list[dict]:
    """
    Recherche les passages du corpus les plus proches d'une section du manuscrit.

    La recherche est cross-lingue : une section en français peut retrouver
    des passages pertinents en allemand ou en italien si le corpus est multilingue.

    Args:
        model        : Modèle SentenceTransformer.
        index        : Index FAISS.
        metadata     : Métadonnées des chunks.
        section_text : Texte de la section du manuscrit.

    Returns:
        Liste de dicts {source, page, score_L2, extrait, relation_pressentie}.
    """
    q_vector = model.encode(
        [section_text[:512]],   # tronqué à la fenêtre du modèle
        convert_to_numpy=True,
    ).astype("float32")

    distances, indices = index.search(q_vector, TOP_K_MAP)

    results = []
    seen    = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        if DISTANCE_THRESHOLD and dist > DISTANCE_THRESHOLD:
            continue

        meta = metadata[idx]
        key  = (meta["source"], meta["page"])
        if key in seen:
            continue
        seen.add(key)

        passage = load_passage(meta["source"], meta["page"])
        if not passage:
            continue

        results.append({
            "source"             : meta["source"],
            "page"               : meta["page"],
            "score_L2"           : round(float(dist), 2),
            "extrait"            : passage[:MAX_EXCERPT_CHARS],
            "relation_pressentie": "à analyser (07_map_critique_local.py)",
        })

    return results


# =============================================================================
# FONCTIONS — SORTIE
# =============================================================================

# =============================================================================
# FONCTIONS — RAPPORT MARKDOWN
# =============================================================================

def generer_rapport_md(
    manuscript_file: str,
    sections_map: list[dict],
    timestamp: str,
) -> Path:
    """
    Génère map_{timestamp}.md — version lisible de la carte enrichie.

    Produit en parallèle du JSON par save_map(). Remplace l'ancien .txt brut.

    Structure par section :
      - Titre et extrait du manuscrit (200 premiers caractères)
      - Liste des passages corpus associés :
          score L2 | source | page | extrait (300 premiers caractères)

    Le score L2 est la distance vectorielle brute : plus il est bas, plus le
    passage est proche sémantiquement de la section du manuscrit. Il n'est pas
    normalisé — les valeurs typiques sont entre 20 et 120 selon le corpus.

    Le rapport ne contient pas d'analyse des relations (c'est le rôle du 07) ;
    il donne la matière brute organisée pour lecture humaine directe.

    Args:
        manuscript_file : Chemin du fichier manuscrit source.
        sections_map    : Liste des sections avec leurs passages corpus.
        timestamp       : Horodatage de la session.

    Returns:
        Chemin du fichier MD produit.
    """
    lignes = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lignes += [
        "# Carte enrichie du manuscrit",
        "",
        f"**Source** : `{manuscript_file}`  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle embeddings** : {EMBEDDING_MODEL}  ",
        f"**Passages par section (top-k)** : {TOP_K_MAP}  ",
        f"**Sections cartographiées** : {len(sections_map)}  ",
        "",
        "> *Ce rapport présente la matière brute de la cartographie vectorielle.*  ",
        "> *L'analyse des relations (conforte / contredit / nuance…) est produite*  ",
        "> *par `07_map_critique_local.py` à partir du JSON associé.*",
        "",
        "---",
        "",
    ]

    # ── Tableau de synthèse ───────────────────────────────────────────────────
    lignes += [
        "## Synthèse",
        "",
        "| § | Titre | Passages associés | Score L2 min |",
        "|---|---|---|---|",
    ]
    for sec in sections_map:
        passages = sec.get("passages_corpus", [])
        n = len(passages)
        score_min = min((p["score_L2"] for p in passages), default=0)
        lignes.append(
            f"| {sec['id']} "
            f"| {sec['titre'][:50]} "
            f"| {n} "
            f"| {score_min:.1f} |"
        )
    lignes += ["", "---", ""]

    # ── Sections détaillées ───────────────────────────────────────────────────
    lignes.append("## Détail par section\n")

    for sec in sections_map:
        lignes += [
            f"### § {sec['id']} — {sec['titre']}",
            "",
            "**Extrait manuscrit :**",
            "",
            f"> {sec.get('texte', '')[:300].replace(chr(10), ' ')}{'…' if len(sec.get('texte','')) > 300 else ''}",
            "",
        ]

        passages = sec.get("passages_corpus", [])
        if not passages:
            lignes += ["*Aucun passage corpus associé.*", "", "---", ""]
            continue

        lignes += [
            f"**{len(passages)} passage(s) corpus associé(s)** *(classés par proximité vectorielle)* :",
            "",
        ]

        for i, p in enumerate(passages, 1):
            source  = p.get("source", "?")
            page    = p.get("page", "?")
            score   = p.get("score_L2", 0)
            extrait = p.get("extrait", "")[:300]
            lignes += [
                f"**{i}.** `{source}` — p. {page} &nbsp;|&nbsp; score L2 : `{score:.1f}`",
                "",
                f"> {extrait.replace(chr(10), ' ')}{'…' if len(p.get('extrait','')) > 300 else ''}",
                "",
            ]

        lignes += ["---", ""]

    # ── Pied de page ──────────────────────────────────────────────────────────
    lignes += [
        "",
        "*Rapport généré automatiquement par `06_map_enrich_local.py`.*  ",
        f"*Score L2 : distance vectorielle brute (L2). Valeurs typiques : 20–120.*  ",
        "*Étape suivante : `python 07_map_critique_local.py`*",
    ]

    contenu   = "\n".join(lignes)
    chemin_md = output_dir / f"map_{timestamp}.md"
    chemin_md.write_text(contenu, encoding="utf-8")
    return chemin_md



def save_map(manuscript_file: str, sections_map: list[dict]) -> tuple[Path, Path]:
    """
    Sauvegarde la carte enrichie en JSON et en Markdown lisible.

    Le JSON est consommé par 07_map_critique_local.py pour l'analyse critique.
    Le fichier Markdown est destiné à la lecture humaine directe ; il remplace
    l'ancien rapport .txt brut (supprimé dans cette version).

    Args:
        manuscript_file : Chemin du fichier manuscrit source.
        sections_map    : Liste des sections avec leurs passages corpus associés.

    Returns:
        Tuple (chemin_json, chemin_md).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON structuré (pour 07_map_critique_local.py)
    json_data = {
        "manuscrit_source" : manuscript_file,
        "date"             : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modele_embeddings": EMBEDDING_MODEL,
        "top_k"            : TOP_K_MAP,
        "sections"         : sections_map,
    }
    json_path = output_dir / f"map_{timestamp}.json"
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Rapport Markdown lisible (remplace l'ancien .txt)
    md_path = generer_rapport_md(manuscript_file, sections_map, timestamp)

    return json_path, md_path


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cartographie enrichie d'un manuscrit via le corpus RAG local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 06_map_enrich_local.py
  python 06_map_enrich_local.py --manuscrit mon_texte.txt
  python 06_map_enrich_local.py --manuscrit brouillon.txt --max-sections 10
  python 06_map_enrich_local.py --manuscrit article.txt --auto-segment
        """,
    )
    parser.add_argument(
        "--manuscrit", "-m",
        default=None,
        help=f"Chemin vers le fichier texte du manuscrit (défaut : {MANUSCRIPT_FILE}).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Nombre maximum de sections à cartographier (écrase MAX_SECTIONS).",
    )
    parser.add_argument(
        "--auto-segment",
        action="store_true",
        default=None,
        help="Force la segmentation par LLM (écrase AUTO_SEGMENT=False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Résolution des paramètres : argument CLI > constante du script
    manuscript_file = args.manuscrit if args.manuscrit else MANUSCRIPT_FILE
    max_sections    = args.max_sections if args.max_sections is not None else MAX_SECTIONS
    auto_segment    = args.auto_segment if args.auto_segment else AUTO_SEGMENT

    # Chargement
    index, metadata = load_vector_store()
    print(f"Index chargé : {index.ntotal} vecteurs")

    print("\nChargement du modèle d'embeddings…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Modèle prêt.\n")

    print(f"Chargement du manuscrit : {manuscript_file}")
    text = load_manuscript(manuscript_file)
    print(f"  {len(text):,} caractères chargés.\n")

    # Segmentation
    if auto_segment:
        print("Segmentation automatique du manuscrit via LLM…")
        sections = segment_by_llm(text)
        if not sections:
            print("  Repli sur segmentation par paragraphes.")
            sections = segment_by_paragraphs(text)
    else:
        print("Segmentation par paragraphes…")
        sections = segment_by_paragraphs(text)

    if max_sections:
        sections = sections[:max_sections]
        print(f"  Limité aux {max_sections} premières sections.")

    print(f"  {len(sections)} sections identifiées.\n")

    # Cartographie section par section
    sections_map = []

    for sec in sections:
        print(f"§ {sec['id']} — {sec['titre']}")
        passages = search_for_section(model, index, metadata, sec["texte"])
        print(f"  {len(passages)} passages corpus associés.")

        sections_map.append({
            "id"             : sec["id"],
            "titre"          : sec["titre"],
            "texte"          : sec["texte"][:MAX_EXCERPT_CHARS],
            "passages_corpus": passages,
        })

    # Sauvegarde
    json_path, md_path = save_map(manuscript_file, sections_map)

    print(f"\n{'═'*60}")
    print(f"✓ Carte enrichie générée.")
    print(f"  JSON (pour 07) : {json_path.resolve()}")
    print(f"  Rapport MD     : {md_path.resolve()}")
    print(f"\n  Étape suivante : python 07_map_critique_local.py {json_path.name}")


if __name__ == "__main__":
    main()
