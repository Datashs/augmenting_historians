"""
04_rag_query_local.py
=====================
Étape 4 du pipeline RAG documentaire — VERSION 100 % LOCALE.

Rôle : Reçoit une question en entrée, recherche les passages les plus
pertinents dans l'index FAISS, puis soumet question + n passages à Ollama
(LLM local) pour produire une réponse synthétique ancrée dans le corpus.

Pipeline :
    01_extract_text.py              → extracted_text/corpus.txt
    02_chunk_corpus.py              → extracted_text/chunks.json
    03_build_embeddings_local.py    → vector_store_local/{…}
    04_rag_query_local.py           → réponse synthétique via Ollama (ce script)

Différences avec la version OpenAI (04_rag_query.py) :
    - Embeddings via sentence-transformers (local, modèle 768 dim).
    - Génération via Ollama (API REST locale, http://localhost:11434).
    - Pas de clé API, pas de coût par token.
    - Streaming de la réponse : les tokens s'affichent au fur et à mesure.

Prérequis :
    pip install sentence-transformers faiss-cpu numpy requests
    ollama serve              # démarrer Ollama en arrière-plan
    ollama pull qwen2.5:14b   # télécharger le modèle (une seule fois)

Structure de fichiers attendue :
    projet/
    ├── 00_config_local.py
    ├── 04_rag_query_local.py
    ├── extracted_text/
    │   └── corpus.txt
    └── vector_store_local/
        ├── faiss.index
        └── metadata.json
"""

# =============================================================================
# PARAMÈTRES — surchargent les valeurs de 00_config_local.py si besoin
# =============================================================================

from rag_config_local import (
    VECTOR_DIR,
    CORPUS_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    OLLAMA_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K,
    SYSTEM_PROMPT_QUERY,
)

# Afficher les scores de distance FAISS pour chaque passage retourné ?
# Utile pour diagnostiquer la qualité de la recherche.
# Score L2 : plus il est bas, plus le passage est proche de la question.
#   < 50    : très bonne correspondance
#   50–150  : correspondance acceptable
#   > 150   : passage probablement peu pertinent
SHOW_SCORES = True

# Seuil de distance L2 au-delà duquel un passage est considéré hors-sujet.
# Les passages dont le score dépasse ce seuil sont exclus du contexte LLM.
# Mettre à None pour désactiver le filtre.
# Ce seuil est à calibrer sur votre corpus spécifique.
DISTANCE_THRESHOLD = 170  # ex : 200.0

# =============================================================================
# IMPORTS
# =============================================================================

import json
import sys
import numpy as np
import faiss
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =============================================================================
# INITIALISATION
# =============================================================================

vector_path   = Path(VECTOR_DIR)
index_file    = vector_path / "faiss.index"
metadata_file = vector_path / "metadata.json"
corpus_path   = Path(CORPUS_FILE)

# =============================================================================
# FONCTIONS
# =============================================================================

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis vector_store_local/.

    Returns:
        index    : Index FAISS prêt pour la recherche.
        metadata : Liste de dicts {source, page, tokens} indexée comme l'index.

    Raises:
        FileNotFoundError : si faiss.index ou metadata.json sont absents.
    """
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord 03_build_embeddings_local.py."
            )

    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """
    Encode la question en vecteur via sentence-transformers.

    Le modèle utilisé DOIT être identique à celui de 03_build_embeddings_local.py.
    Un modèle différent produirait des vecteurs dans un espace différent,
    rendant les distances FAISS sans signification.

    Args:
        model : Modèle SentenceTransformer déjà chargé.
        query : Question posée par l'utilisateur.

    Returns:
        Vecteur float32 de forme (1, EMBEDDING_DIM) prêt pour index.search().
    """
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    return vector.astype("float32")


def load_passage(source: str, page: int) -> str:
    """
    Récupère le passage complet d'une page depuis corpus.txt.

    Plutôt que de retourner uniquement le chunk (potentiellement tronqué par
    la fenêtre du modèle d'embeddings), cette fonction restitue le texte
    intégral de la section correspondante dans le corpus brut,
    offrant davantage de contexte au LLM.

    Le marqueur de page a la forme : [nom_source — page N]

    Args:
        source : Nom du fichier source (ex : "manuscrit.pdf").
        page   : Numéro de page.

    Returns:
        Texte de la section, ou chaîne vide si le marqueur est introuvable.
    """
    if not corpus_path.exists():
        return ""

    marker = f"[{source} — page {page}]"
    text   = corpus_path.read_text(encoding="utf-8")
    start  = text.find(marker)

    if start == -1:
        return ""

    start += len(marker)
    end    = text.find("[", start)   # prochain marqueur = fin de la section
    return text[start:end].strip() if end != -1 else text[start:].strip()


def search_chunks(
    index: faiss.Index,
    metadata: list[dict],
    q_vector: np.ndarray,
) -> list[tuple[str, float]]:
    """
    Recherche les TOP_K chunks les plus proches dans l'index FAISS.

    Returns:
        Liste de tuples (passage_formaté, score_distance_L2).
        Les passages vides (source introuvable dans corpus.txt) sont ignorés.
        Si DISTANCE_THRESHOLD est défini, les passages trop éloignés sont exclus.

    Note sur les scores L2 :
        FAISS retourne des distances L2 (euclidienne au carré).
        Contrairement à la similarité cosinus, une distance faible = bonne correspondance.
        La valeur absolue dépend de la norme des vecteurs et du modèle utilisé.
        Calibrez DISTANCE_THRESHOLD empiriquement sur votre corpus.
    """
    distances, indices = index.search(q_vector, TOP_K)

    excerpts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:   # FAISS retourne -1 pour les résultats manquants
            continue
        if DISTANCE_THRESHOLD is not None and dist > DISTANCE_THRESHOLD:
            continue

        meta    = metadata[idx]
        passage = load_passage(meta["source"], meta["page"])
        if passage:
            label = f"{meta['source']} (p. {meta['page']})"
            excerpts.append((f"{label}:\n{passage}", float(dist)))

    return excerpts


def call_ollama(prompt: str, stream: bool = True) -> str:
    """
    Soumet un prompt à Ollama et retourne la réponse générée.

    Ollama expose une API REST locale sur http://localhost:11434.
    Le streaming (stream=True) affiche les tokens au fur et à mesure,
    ce qui améliore l'expérience utilisateur pour les réponses longues.

    Args:
        prompt : Prompt complet (système + question + extraits).
        stream : Si True, affiche la réponse token par token en temps réel.

    Returns:
        Réponse complète du LLM sous forme de chaîne.

    Raises:
        SystemExit : Si Ollama n'est pas joignable (daemon non démarré).
    """
    payload = {
        "model"       : LLM_MODEL,
        "prompt"      : prompt,
        "temperature" : LLM_TEMPERATURE,
        "num_predict" : LLM_MAX_TOKENS,
        "stream"      : stream,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=stream,
            timeout=120,  # secondes ; augmenter si le modèle est lent à charger
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            "\n❌ Impossible de joindre Ollama.\n"
            "   Vérifiez que le daemon est démarré : ollama serve\n"
            f"   URL configurée : {OLLAMA_URL}"
        )
        sys.exit(1)

    full_response = []

    if stream:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if chunk.get("done", False):
                    break
        print()   # saut de ligne final
    else:
        data = response.json()
        full_response = [data.get("response", "")]

    return "".join(full_response)


def generate_answer(query: str, excerpts: list[tuple[str, float]]) -> str:
    """
    Construit le prompt et génère la réponse via Ollama.

    Le prompt est structuré en trois parties :
    1. Instructions système (rôle, contraintes).
    2. Extraits du corpus (contexte).
    3. Question de l'utilisateur.

    Args:
        query   : Question originale de l'utilisateur.
        excerpts: Passages récupérés par search_chunks() avec leurs scores.

    Returns:
        Réponse textuelle générée par le LLM.
    """
    if not excerpts:
        return "Aucun passage pertinent trouvé dans le corpus pour répondre à cette question."

    context = "\n\n---\n\n".join(text for text, _ in excerpts)

    prompt = f"""{SYSTEM_PROMPT_QUERY}

EXTRAITS DU CORPUS :
{context}

QUESTION :
{query}

RÉPONSE :"""

    return call_ollama(prompt, stream=True)


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # Chargement de l'index vectoriel
    index, metadata = load_vector_store()
    print(f"Index chargé : {index.ntotal} vecteurs | modèle : {EMBEDDING_MODEL}")
    print(f"LLM          : {LLM_MODEL} (via Ollama)")

    # Chargement du modèle d'embeddings
    print(f"\nChargement du modèle d'embeddings…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Modèle prêt.")

    # Boucle de questions interactive
    while True:
        print()
        query = input("Votre question (ou 'q' pour quitter) : ").strip()

        if query.lower() in ("q", "quit", "exit"):
            print("Au revoir.")
            break

        if not query:
            print("Question vide, ignorée.")
            continue

        # Recherche vectorielle
        print(f"\nRecherche des {TOP_K} passages les plus pertinents…")
        q_vector = embed_query(model, query)
        excerpts = search_chunks(index, metadata, q_vector)

        if SHOW_SCORES and excerpts:
            print(f"{len(excerpts)} passages trouvés :")
            for i, (text, score) in enumerate(excerpts, 1):
                source_line = text.split("\n")[0]   # première ligne = "source (p. N):"
                print(f"  {i:2}. [{score:7.1f}] {source_line}")

        # Génération de la réponse
        print(f"\n--- Réponse ({LLM_MODEL}) ---\n")
        generate_answer(query, excerpts)


if __name__ == "__main__":
    main()
