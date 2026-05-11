"""
04_rag_query.py
===============
Étape 4 (finale) du pipeline RAG documentaire.

Rôle : Reçoit une question en entrée, recherche les passages les plus
pertinents dans l'index FAISS, puis soumet question + passages à un LLM
pour produire une réponse synthétique ancrée dans le corpus.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → réponse synthétique via LLM                 (ce script)
    ──
    tool_semantic_search.py  Renvoie les références des segments les plus pertinents
    tool_add_document.py     ajout incrémental d'un PDF au corpus

Structure de fichiers attendue :
    projet/
    ├── .env                         ← OPENAI_API_KEY=sk-...
    ├── 04_rag_query.py
    ├── extracted_text/
    │   └── corpus.txt               ← requis pour récupérer les passages complets
    └── vector_store/
        ├── faiss.index              ← requis
        └── metadata.json            ← requis

Fonctionnement :
    1. La question est encodée en vecteur via le même modèle d'embeddings
       que celui utilisé dans 03_build_embeddings.py.
    2. FAISS recherche les TOP_K chunks les plus proches dans l'index.
    3. Pour chaque chunk trouvé, le passage complet de la page source est
       récupéré depuis corpus.txt (pour donner plus de contexte au LLM).
    4. Le LLM génère une réponse en se basant uniquement sur ces passages.
"""

# =============================================================================
# PARAMÈTRES — ajustez selon votre contexte
# =============================================================================

# --- Chemins ---
VECTOR_DIR   = "vector_store"
CORPUS_FILE  = "extracted_text/corpus.txt"

# --- Modèles ---
# Doit être identique au modèle utilisé dans build_embeddings.py.
# Un modèle différent produirait des vecteurs incompatibles avec l'index.
EMBEDDING_MODEL = "text-embedding-3-large"

# Modèle LLM pour la synthèse finale.
# "gpt-4.1-mini" : bon équilibre coût / qualité pour la majorité des usages.
# "gpt-4.1"      : meilleure qualité, plus coûteux.
LLM_MODEL = "gpt-4.1-mini"

# --- Recherche ---
# Nombre de chunks retournés par FAISS pour construire le contexte.
# Plus TOP_K est élevé → plus le contexte fourni au LLM est large, mais aussi plus
# de bruit potentiel et un prompt plus long (coût et latence accrus).
# Valeurs typiques : 5–20.
TOP_K = 15

# --- Génération ---
# Température du LLM : contrôle le caractère aléatoire de la réponse.
# 0.0 → réponses déterministes et factuelles (recommandé pour RAG).
# 1.0 → réponses plus créatives et variées.
LLM_TEMPERATURE = 0.1

# Langue et rôle du LLM dans le prompt système.
# Modifiez ce prompt pour adapter le comportement à votre corpus.
SYSTEM_PROMPT = """You are a historian's assistant.
Answer the question using ONLY the excerpts provided.
Do not introduce any information that is not explicitly present in the excerpts.
If the excerpts are insufficient to answer the question, say so clearly."""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# INITIALISATION
# =============================================================================

load_dotenv()
client = OpenAI()  # Lit automatiquement OPENAI_API_KEY depuis .env

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"
corpus_path   = Path(CORPUS_FILE)

# =============================================================================
# FONCTIONS
# =============================================================================

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis le dossier vector_store.

    Returns:
        index    : Index FAISS prêt pour la recherche.
        metadata : Liste de dicts {source, page, tokens} indexée comme l'index FAISS.

    Raises:
        FileNotFoundError : Si faiss.index ou metadata.json sont absents.
    """
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord build_embeddings.py pour générer le vector store."
            )

    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def embed_query(query: str) -> np.ndarray:
    """
    Encode la question en vecteur via l'API OpenAI.

    Le modèle utilisé doit être identique à celui de build_embeddings.py,
    sans quoi les distances calculées par FAISS n'auraient aucun sens.

    Args:
        query: Question posée par l'utilisateur.

    Returns:
        Vecteur float32 de forme (1, EMBEDDING_DIM) prêt pour index.search().
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    vector = np.array(response.data[0].embedding, dtype="float32")
    return vector.reshape(1, -1)


def load_passage(source: str, page: int) -> str:
    """
    Récupère le passage complet d'une page depuis corpus.txt.

    Plutôt que de retourner uniquement le chunk (potentiellement tronqué),
    cette fonction restitue le texte intégral de la section correspondante,
    ce qui donne davantage de contexte au LLM.

    Args:
        source : Nom de la source (ex : "rapport_2024.pdf").
        page   : Numéro de page.

    Returns:
        Texte de la section, ou chaîne vide si l'en-tête est introuvable.
    """
    if not corpus_path.exists():
        return ""

    marker = f"[{source} — page {page}]"
    text   = corpus_path.read_text(encoding="utf-8")
    start  = text.find(marker)

    if start == -1:
        return ""

    start += len(marker)
    end    = text.find("[", start)  # Prochain en-tête = fin de la section
    return text[start:end].strip()


def search_chunks(index: faiss.Index, metadata: list[dict], q_vector: np.ndarray) -> list[str]:
    """
    Recherche les TOP_K chunks les plus proches dans l'index FAISS
    et reconstruit les passages complets correspondants.

    Args:
        index    : Index FAISS chargé.
        metadata : Métadonnées associées à chaque vecteur.
        q_vector : Vecteur de la question, forme (1, dim).

    Returns:
        Liste de chaînes formatées "source (p. N):\\npassage…"
        (les passages vides sont ignorés).
    """
    _, indices = index.search(q_vector, TOP_K)

    excerpts = []
    for idx in indices[0]:
        meta    = metadata[idx]
        passage = load_passage(meta["source"], meta["page"])
        if passage:
            excerpts.append(f"{meta['source']} (p. {meta['page']}):\n{passage}")

    return excerpts


def generate_answer(query: str, excerpts: list[str]) -> str:
    """
    Soumet la question et les passages au LLM pour produire une réponse synthétique.

    Le prompt est structuré en deux parties :
    - Un message système (SYSTEM_PROMPT) qui définit le rôle et les contraintes.
    - Un message utilisateur qui contient la question et les extraits.

    Args:
        query    : Question originale de l'utilisateur.
        excerpts : Passages récupérés par search_chunks().

    Returns:
        Réponse textuelle générée par le LLM.
    """
    if not excerpts:
        return "Aucun passage pertinent trouvé dans le corpus pour répondre à cette question."

    context = "\n\n---\n\n".join(excerpts)
    user_message = f"Question:\n{query}\n\nExcerpts:\n{context}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=LLM_TEMPERATURE,
    )

    return response.choices[0].message.content


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    index, metadata = load_vector_store()
    print(f"Index chargé : {index.ntotal} vecteurs | modèle : {EMBEDDING_MODEL}")

    query = input("\nVotre question : ").strip()
    if not query:
        print("Question vide, abandon.")
        return

    print(f"\nRecherche des {TOP_K} passages les plus pertinents…")
    q_vector = embed_query(query)
    excerpts = search_chunks(index, metadata, q_vector)
    print(f"{len(excerpts)} passages trouvés.")

    print("\n--- Réponse synthétique ---\n")
    answer = generate_answer(query, excerpts)
    print(answer)


if __name__ == "__main__":
    main()
