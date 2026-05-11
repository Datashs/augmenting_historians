"""
05_rag_write_local.py
=====================
Étape 5 du pipeline RAG documentaire — VERSION 100 % LOCALE.

Rôle : Rédaction assistée. À partir d'une consigne de rédaction (plan,
angle, question historiographique), recherche les passages pertinents
dans le corpus, puis demande à Ollama de rédiger un passage académique
structuré, ancré dans les sources, avec citations explicites.

Pipeline :
    …
    03_build_embeddings_local.py → vector_store_local/{…}
    04_rag_query_local.py        → réponse synthétique
    05_rag_write_local.py        → passage rédigé avec citations (ce script)
    06_map_enrich_local.py       → carte enrichie du manuscrit
    …

Différences avec 04_rag_query_local.py :
    - TOP_K plus élevé pour un contexte plus riche.
    - Température légèrement plus haute (0.3) pour fluidité stylistique.
    - Prompt orienté rédaction académique avec citations intégrées.
    - Sortie optionnelle vers un fichier texte horodaté.

Usage :
    python 05_rag_write_local.py
    # Entrez la consigne de rédaction au prompt interactif.
    # Exemple : "Rédigez un paragraphe sur le rôle des marchands
    #            lombards dans la circulation monétaire au XIVe siècle."

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
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT_WRITE,
)

# TOP_K plus élevé que pour la requête simple : la rédaction bénéficie
# d'un contexte plus large pour tisser des arguments nuancés.
TOP_K_WRITE = 18

# Température plus haute que pour la requête pure :
# encourage la fluidité stylistique tout en restant factuel.
TEMPERATURE_WRITE = 0.3

# Sauvegarder la réponse dans un fichier texte ?
SAVE_OUTPUT = True
OUTPUT_DIR  = "outputs"    # dossier de sortie (créé automatiquement)

# Seuil de distance L2 pour filtrer les passages hors-sujet (None = pas de filtre)
DISTANCE_THRESHOLD = None

# =============================================================================
# IMPORTS
# =============================================================================

import json
import sys
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
# FONCTIONS (reprises et adaptées de 04_rag_query_local.py)
# =============================================================================

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
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


def search_chunks(
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    query: str,
) -> list[tuple[str, str, int, float]]:
    """
    Recherche les TOP_K_WRITE passages les plus pertinents.

    Returns:
        Liste de tuples (texte_passage, source, page, score_L2).
        Retourner source et page séparément facilite la construction
        des citations dans le prompt.
    """
    q_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vector, TOP_K_WRITE)

    results = []
    seen    = set()   # évite les doublons (même source+page)

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        if DISTANCE_THRESHOLD and dist > DISTANCE_THRESHOLD:
            continue

        meta    = metadata[idx]
        key     = (meta["source"], meta["page"])
        if key in seen:
            continue
        seen.add(key)

        passage = load_passage(meta["source"], meta["page"])
        if passage:
            results.append((passage, meta["source"], meta["page"], float(dist)))

    return results


def build_writing_prompt(consigne: str, passages: list[tuple]) -> str:
    """
    Construit le prompt de rédaction avec les extraits et les instructions.

    Structure :
    - Instructions système (rôle, style, format des citations)
    - Consigne de rédaction de l'utilisateur
    - Extraits numérotés du corpus (avec référence source+page)
    - Demande de rédaction

    Les extraits sont présentés avec leur référence explicite [source, p. N]
    pour que le LLM puisse les intégrer naturellement dans le texte rédigé.

    Args:
        consigne : Consigne de rédaction formulée par l'utilisateur.
        passages : Liste de tuples (texte, source, page, score).

    Returns:
        Prompt complet prêt à être soumis à Ollama.
    """
    # Formatage des extraits avec référence pour les citations
    extraits_formattés = []
    for i, (texte, source, page, _score) in enumerate(passages, 1):
        ref = f"[{source}, p. {page}]"
        extraits_formattés.append(f"Extrait {i} {ref} :\n{texte}")

    context = "\n\n---\n\n".join(extraits_formattés)

    prompt = f"""{SYSTEM_PROMPT_WRITE}

RÈGLES DE CITATION — OBLIGATOIRES :
- Chaque affirmation appuyée sur un extrait doit être suivie de sa référence entre crochets : [source, p. N].
- Si un extrait exprime l'idée de façon particulièrement nette, cite-le brièvement entre guillemets avant la référence.
- Ne cite ou ne réfère QUE des passages explicitement présents dans les extraits fournis ci-dessous.
- N'invente aucune citation, aucun auteur, aucune date.

CONSIGNE DE RÉDACTION :
{consigne}

EXTRAITS DU CORPUS (à citer entre crochets sous la forme [source, p. N]) :

{context}

PASSAGE RÉDIGÉ :"""

    return prompt


def call_ollama_write(prompt: str) -> str:
    """
    Soumet le prompt de rédaction à Ollama avec streaming.

    Paramètres spécifiques à la rédaction :
    - Température légèrement plus haute (TEMPERATURE_WRITE) pour la fluidité.
    - num_predict plus élevé pour des passages longs.

    Args:
        prompt : Prompt de rédaction complet.

    Returns:
        Texte rédigé par le LLM.
    """
    payload = {
        "model"       : LLM_MODEL,
        "prompt"      : prompt,
        "temperature" : TEMPERATURE_WRITE,
        "num_predict" : 4096,
        "stream"      : True,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            "\n❌ Ollama inaccessible. Lancez : ollama serve\n"
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


def save_output(consigne: str, passage: str, passages_source: list[tuple]) -> Path:
    """
    Sauvegarde le passage rédigé dans un fichier texte horodaté.

    Format du fichier :
        CONSIGNE
        --------
        [consigne]

        SOURCES CONSULTÉES
        ------------------
        [liste des sources avec scores]

        PASSAGE RÉDIGÉ
        --------------
        [texte]

    Args:
        consigne        : Consigne de rédaction originale.
        passage         : Texte rédigé par le LLM.
        passages_source : Tuples (texte, source, page, score) utilisés.

    Returns:
        Chemin du fichier sauvegardé.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"redaction_{timestamp}.txt"

    sources_list = "\n".join(
        f"  [{source}, p. {page}] — score L2 : {score:.1f}"
        for _, source, page, score in passages_source
    )

    content = (
        f"CONSIGNE\n{'─'*60}\n{consigne}\n\n"
        f"SOURCES CONSULTÉES ({len(passages_source)} passages)\n{'─'*60}\n"
        f"{sources_list}\n\n"
        f"PASSAGE RÉDIGÉ\n{'─'*60}\n{passage}\n"
    )

    filepath.write_text(content, encoding="utf-8")
    return filepath


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    index, metadata = load_vector_store()
    print(f"Index chargé : {index.ntotal} vecteurs")
    print(f"LLM          : {LLM_MODEL} | TOP_K : {TOP_K_WRITE} | T° : {TEMPERATURE_WRITE}")

    print("\nChargement du modèle d'embeddings…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Modèle prêt.\n")

    while True:
        print("─" * 60)
        print("Entrez votre consigne de rédaction (ou 'q' pour quitter).")
        print("Exemple : 'Rédigez un paragraphe sur les pratiques notariales")
        print("          dans les villes lombardes au XIVe siècle.'")
        consigne = input("\nConsigne : ").strip()

        if consigne.lower() in ("q", "quit", "exit"):
            print("Au revoir.")
            break

        if not consigne:
            print("Consigne vide, ignorée.")
            continue

        # Recherche des passages pertinents
        print(f"\nRecherche de {TOP_K_WRITE} passages pertinents…")
        passages = search_chunks(model, index, metadata, consigne)
        print(f"{len(passages)} passages trouvés :")
        for _, source, page, score in passages:
            print(f"  [{score:7.1f}] {source} (p. {page})")

        if not passages:
            print("Aucun passage pertinent. Reformulez la consigne.")
            continue

        # Rédaction
        prompt = build_writing_prompt(consigne, passages)
        print(f"\n--- Passage rédigé ({LLM_MODEL}) ---\n")
        texte_rédigé = call_ollama_write(prompt)

        # Sauvegarde optionnelle
        if SAVE_OUTPUT:
            filepath = save_output(consigne, texte_rédigé, passages)
            print(f"\n💾 Sauvegardé : {filepath.resolve()}")


if __name__ == "__main__":
    main()
