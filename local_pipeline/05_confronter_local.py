"""
05_confronter_local.py
======================
Étape 5 du pipeline RAG — Confrontation paragraphe / corpus (VERSION LOCALE).

Rôle : Lit un paragraphe du manuscrit (fichier .txt ou saisie interactive),
l'utilise comme requête RAG pour trouver les passages du corpus les plus
pertinents, puis demande à Ollama d'analyser chaque passage et de proposer
comment l'intégrer dans le texte.

Pipeline local :
    03_build_embeddings_local.py → vector_store_local/{…}
    04_rag_query_local.py        → exploration thématique libre
    05_confronter_local.py       → confrontation paragraphe ↔ corpus  (ce script)
    05bis_rediger_local.py       → rédaction assistée par consigne
    06_map_enrich_local.py       → cartographie enrichissement

Différence avec 05bis_rediger_local.py :
    - Ce script : part d'un PARAGRAPHE existant → l'ancre dans le corpus
    - 05bis     : part d'une CONSIGNE           → rédige un nouveau passage

Embeddings : sentence-transformers (local, paraphrase-multilingual-mpnet-base-v2)
Génération : Ollama (qwen2.5:14b)
Aucune API externe, aucune clé requise.

Usage :
    python 05_confronter_local.py --paragraphe mon_paragraphe.txt
    python 05_confronter_local.py --paragraphe mon_paragraphe.txt --sortie analyse.txt
    python 05_confronter_local.py --paragraphe mon_paragraphe.txt --top_k 5
    python 05_confronter_local.py   # saisie interactive du paragraphe

Arguments :
    --paragraphe FICHIER   Fichier .txt contenant le paragraphe (optionnel)
                           Si absent : saisie interactive dans le terminal
    --sortie FICHIER       Fichier de sortie pour l'analyse (optionnel)
    --top_k N              Nombre de passages récupérés

Prérequis :
    ollama serve
    ollama pull qwen2.5:14b
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
    TOP_K,
)

# TOP_K local (None = utilise TOP_K de rag_config_local.py)
TOP_K_LOCAL = None

# Longueur minimale d'un passage pour être inclus (en caractères)
LONGUEUR_PASSAGE_MIN = 100

# Encodage de sortie
ENCODAGE_SORTIE = "utf-8"

# Sauvegarder automatiquement dans outputs/ ?
SAVE_OUTPUT = True
OUTPUT_DIR  = "outputs"

# =============================================================================
# PROMPT SYSTÈME
# =============================================================================

SYSTEM_PROMPT = """Tu es un assistant spécialisé en histoire, expert en analyse documentaire.
Tu aides un historien à enrichir son manuscrit en identifiant des passages
du corpus scientifique qui peuvent renforcer son argumentation.

Règles absolues :
- Ne cite QUE des passages explicitement présents dans les extraits fournis.
- N'invente aucune citation, aucun auteur, aucune date.
- Si un extrait n'est pas pertinent pour le paragraphe, dis-le clairement.
- Reste ancré dans ce que les textes disent réellement."""

USER_TEMPLATE = """Voici un paragraphe d'un manuscrit historique :

--- PARAGRAPHE ---
{paragraphe}
--- FIN DU PARAGRAPHE ---

Voici des extraits du corpus scientifique récupérés par similarité sémantique :

{extraits}

Pour chaque extrait pertinent, fournis une analyse structurée en quatre points :

1. CITATION PERTINENTE
   Cite le passage exact (entre guillemets) qui peut être utile au paragraphe.

2. FONCTION ARGUMENTATIVE
   Explique précisément comment cette citation renforce, nuance ou complète
   l'argument du paragraphe (2–3 phrases).

3. PROPOSITION D'INSERTION
   Rédige une phrase ou deux montrant comment intégrer cette citation dans
   le paragraphe, avec une formule d'introduction appropriée au style historique.

4. RÉFÉRENCE DE RENVOI
   Indique : Fichier source : [nom du fichier] | Page : [numéro de page]

Si un extrait n'est pas pertinent pour ce paragraphe, indique simplement :
[Extrait non pertinent — raison brève]

Commence ton analyse."""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import sys
import json
import requests
import numpy as np
import faiss
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

def charger_paragraphe_fichier(chemin: str) -> str:
    """
    Lit le fichier .txt contenant le paragraphe à analyser.

    Args:
        chemin : Chemin vers le fichier .txt.

    Returns:
        Contenu textuel nettoyé.

    Raises:
        FileNotFoundError : Si le fichier est introuvable.
        ValueError        : Si le fichier est vide.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}\n"
            "Créez un fichier .txt contenant le paragraphe à analyser."
        )
    texte = path.read_text(encoding="utf-8").strip()
    if not texte:
        raise ValueError(f"Le fichier {chemin} est vide.")
    return texte


def charger_paragraphe_interactif() -> str:
    """
    Saisie interactive du paragraphe dans le terminal.

    Permet de coller directement un paragraphe sans créer de fichier.
    La saisie se termine par une ligne vide + Entrée.

    Returns:
        Paragraphe saisi, nettoyé.
    """
    print("Collez votre paragraphe ci-dessous.")
    print("Terminez par une ligne vide (Entrée deux fois) :")
    print()

    lignes = []
    while True:
        ligne = input()
        if ligne == "" and lignes:
            break
        lignes.append(ligne)

    return "\n".join(lignes).strip()


def charger_index() -> tuple[faiss.Index, list[dict]]:
    """Charge l'index FAISS et les métadonnées."""
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord 03_build_embeddings_local.py."
            )
    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def charger_corpus() -> str:
    """Charge corpus.txt en mémoire."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {corpus_path.resolve()}\n"
            "Lancez d'abord 01_extract_text.py."
        )
    return corpus_path.read_text(encoding="utf-8")


def recuperer_passage(corpus_text: str, source: str, page: int) -> str:
    """
    Extrait le passage complet d'une page depuis corpus.txt.

    Le marqueur de section a la forme : [source — page N]

    Args:
        corpus_text : Contenu de corpus.txt.
        source      : Nom du fichier source.
        page        : Numéro de page.

    Returns:
        Texte de la section, ou chaîne vide si introuvable.
    """
    marker = f"[{source} — page {page}]"
    start  = corpus_text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end    = corpus_text.find("[", start)
    return corpus_text[start:end].strip() if end != -1 else corpus_text[start:].strip()


# =============================================================================
# FONCTIONS — RECHERCHE
# =============================================================================

def rechercher_passages(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    paragraphe: str,
    corpus_text: str,
    top_k: int,
) -> list[dict]:
    """
    Encode le paragraphe et recherche les passages les plus proches.

    Args:
        model       : Modèle SentenceTransformer chargé.
        index       : Index FAISS.
        metadata    : Métadonnées des vecteurs.
        paragraphe  : Texte du paragraphe à analyser.
        corpus_text : Contenu de corpus.txt.
        top_k       : Nombre de chunks à récupérer.

    Returns:
        Liste de dicts {source, page, passage}, dédoublonnés, filtrés.
    """
    vecteur            = model.encode([paragraphe], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vecteur, top_k)

    passages = []
    vus      = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        cle  = (meta["source"], meta["page"])
        if cle in vus:
            continue
        vus.add(cle)

        passage = recuperer_passage(corpus_text, meta["source"], meta["page"])
        if len(passage) < LONGUEUR_PASSAGE_MIN:
            continue

        passages.append({
            "source":   meta["source"],
            "page":     meta["page"],
            "passage":  passage,
            "distance": float(dist),
        })

    return passages


# =============================================================================
# FONCTIONS — ANALYSE LLM (Ollama)
# =============================================================================

def formater_extraits(passages: list[dict]) -> str:
    """Formate les passages pour le prompt avec numérotation et référence."""
    blocs = []
    for i, p in enumerate(passages, 1):
        ref = f"[{p['source']}, p. {p['page']}]"
        blocs.append(f"Extrait {i} {ref} :\n{p['passage']}")
    return "\n\n---\n\n".join(blocs)


def call_ollama(prompt: str) -> str:
    """
    Soumet le prompt à Ollama avec streaming token par token.

    Args:
        prompt : Prompt complet (système + paragraphe + extraits).

    Returns:
        Réponse complète du LLM.
    """
    payload = {
        "model":       LLM_MODEL,
        "prompt":      prompt,
        "temperature": LLM_TEMPERATURE,
        "num_predict": LLM_MAX_TOKENS,
        "stream":      True,
    }

    try:
        response = requests.post(
            OLLAMA_URL, json=payload, stream=True, timeout=180
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            f"\n❌ Ollama inaccessible. Lancez : ollama serve\n"
            f"   URL : {OLLAMA_URL}"
        )
        sys.exit(1)

    tokens = []
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            tokens.append(token)
            if chunk.get("done", False):
                break
    print()

    return "".join(tokens)


def analyser_paragraphe(paragraphe: str, passages: list[dict]) -> str:
    """
    Construit le prompt et soumet le paragraphe + extraits à Ollama.

    Args:
        paragraphe : Texte du paragraphe du manuscrit.
        passages   : Liste de dicts {source, page, passage, distance}.

    Returns:
        Analyse textuelle produite par Ollama.
    """
    if not passages:
        return (
            "Aucun passage pertinent trouvé dans le corpus pour ce paragraphe.\n"
            "Suggestions :\n"
            "  - Augmentez TOP_K (--top_k)\n"
            "  - Le sujet est peut-être peu représenté dans le corpus"
        )

    extraits = formater_extraits(passages)
    prompt   = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(paragraphe=paragraphe, extraits=extraits)}"

    return call_ollama(prompt)


# =============================================================================
# FONCTIONS — SORTIE
# =============================================================================

def formater_rapport(
    paragraphe: str,
    passages: list[dict],
    analyse: str,
    source_label: str,
) -> str:
    """Formate le rapport final."""
    separateur = "═" * 60
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M")

    sources = "\n".join(
        f"  [{p['distance']:6.2f}] {p['source']} (p. {p['page']})"
        for p in passages
    )

    return f"""{separateur}
  CONFRONTATION RAG — ÉCRITURE ASSISTÉE (Local/Ollama)
  Source   : {source_label}
  Modèle   : {LLM_MODEL}
  Date     : {horodatage}
  Passages : {len(passages)} extrait(s) analysé(s)
{separateur}

PARAGRAPHE SOUMIS :
{'-' * 40}
{paragraphe}
{'-' * 40}

SOURCES IDENTIFIÉES (score L2) :
{sources}

ANALYSE :
{analyse}

{separateur}
  FIN DE L'ANALYSE
{separateur}
"""


def sauvegarder_rapport(rapport: str, chemin: str):
    """Sauvegarde le rapport dans un fichier texte."""
    path = Path(chemin)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rapport, encoding=ENCODAGE_SORTIE)
    print(f"\n✓ Rapport sauvegardé : {path.resolve()}")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="05_confronter_local.py — confrontation paragraphe ↔ corpus (Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 05_confronter_local.py --paragraphe mon_paragraphe.txt
  python 05_confronter_local.py --paragraphe mon_paragraphe.txt --sortie analyse.txt
  python 05_confronter_local.py --top_k 5
  python 05_confronter_local.py   # saisie interactive
        """
    )
    parser.add_argument("--paragraphe", default=None, metavar="FICHIER",
                        help="Fichier .txt contenant le paragraphe (optionnel, défaut : saisie interactive)")
    parser.add_argument("--sortie", default=None, metavar="FICHIER",
                        help="Fichier de sortie pour le rapport (optionnel)")
    parser.add_argument("--top_k", type=int, default=None, metavar="N",
                        help="Nombre de passages à récupérer")
    args = parser.parse_args()

    top_k = args.top_k or TOP_K_LOCAL or TOP_K

    print("=" * 60)
    print("  05 — CONFRONTATION PARAGRAPHE ↔ CORPUS (Local/Ollama)")
    print("=" * 60)
    print(f"  LLM        : {LLM_MODEL}")
    print(f"  Embeddings : {EMBEDDING_MODEL}")
    print(f"  TOP_K      : {top_k}")
    print()

    # Chargement
    try:
        print("Chargement de l'index FAISS…")
        index, metadata = charger_index()
        print(f"  {index.ntotal} vecteurs dans l'index.")

        print("Chargement du corpus…")
        corpus_text = charger_corpus()

    except FileNotFoundError as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    print("Chargement du modèle d'embeddings…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Modèle prêt.\n")

    # Boucle interactive
    while True:
        print("─" * 60)

        # Chargement du paragraphe
        if args.paragraphe:
            try:
                paragraphe   = charger_paragraphe_fichier(args.paragraphe)
                source_label = args.paragraphe
            except (FileNotFoundError, ValueError) as e:
                print(f"❌ {e}")
                sys.exit(1)
            # En mode fichier : une seule passe, pas de boucle
            boucle = False
        else:
            print("Entrez votre paragraphe (ou 'q' pour quitter) :")
            premier_input = input().strip()
            if premier_input.lower() in ("q", "quit", "exit"):
                print("Au revoir.")
                break
            # Récupère les lignes suivantes jusqu'à ligne vide
            lignes = [premier_input]
            while True:
                ligne = input()
                if ligne == "":
                    break
                lignes.append(ligne)
            paragraphe   = "\n".join(lignes).strip()
            source_label = "saisie interactive"
            boucle       = True

        if not paragraphe:
            print("Paragraphe vide, ignoré.")
            continue

        print(f"\n  {len(paragraphe)} caractères chargés.")

        # Recherche
        print(f"Recherche des {top_k} passages les plus pertinents…")
        passages = rechercher_passages(model, index, metadata, paragraphe, corpus_text, top_k)
        print(f"  {len(passages)} passage(s) trouvé(s) :")
        for p in passages:
            print(f"    [{p['distance']:6.2f}] {p['source']} (p. {p['page']})")

        # Analyse
        print(f"\n--- Analyse ({LLM_MODEL}) ---\n")
        analyse = analyser_paragraphe(paragraphe, passages)

        # Rapport
        rapport = formater_rapport(paragraphe, passages, analyse, source_label)

        if args.sortie:
            sauvegarder_rapport(rapport, args.sortie)
        elif SAVE_OUTPUT:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chemin    = output_dir / f"confrontation_local_{timestamp}.txt"
            sauvegarder_rapport(rapport, str(chemin))
        else:
            print("\n" + rapport)

        print("\n✓ Analyse terminée.")

        if not boucle:
            break


if __name__ == "__main__":
    main()
