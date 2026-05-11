"""
rag_config_local.py
==================
Configuration centrale du pipeline RAG 100 % local.

Ce fichier est importé par tous les scripts _local du pipeline.
Modifier les valeurs ici suffit pour reconfigurer l'ensemble du pipeline.

VERSION LOCALE — aucune transmission à des API externes.
    Embeddings : sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
    Génération : Ollama (qwen2.5:14b recommandé sur Mac M2 32 Go)

Pipeline complet :
    01_extract_text.py          → extracted_text/corpus.txt         (inchangé)
    02_chunk_corpus.py          → extracted_text/chunks.json        (inchangé)
    03_build_embeddings_local.py → vector_store_local/{…}
    04_rag_query_local.py       → réponse synthétique via Ollama
    05_rag_write_local.py       → passage rédigé ancré dans le corpus
    06_map_enrich_local.py      → carte enrichie du manuscrit
    07_map_critique_local.py    → rapport critique structuré (JSON)
    08_visualise.py             → visualisation (inchangé)

Prérequis système :
    pip install sentence-transformers faiss-cpu numpy
    # Ollama doit être lancé en arrière-plan avant d'exécuter les scripts :
    ollama serve
    ollama pull qwen2.5:14b
"""

# =============================================================================
# CHEMINS
# =============================================================================

# Dossier des chunks produits par 02_chunk_corpus.py
CHUNKS_FILE  = "extracted_text/chunks.json"

# Corpus texte brut produit par 01_extract_text.py
CORPUS_FILE  = "extracted_text/corpus.txt"

# Dossier de l'index vectoriel local (distinct du dossier OpenAI pour éviter
# tout mélange : les dimensions des vecteurs sont différentes)
VECTOR_DIR   = "vector_store_local"

# =============================================================================
# MODÈLE D'EMBEDDINGS (sentence-transformers)
# =============================================================================

# paraphrase-multilingual-mpnet-base-v2 :
#   - Multilingue natif (50+ langues : FR, EN, DE, IT, etc.)
#   - Dimension des vecteurs : 768
#   - Taille du modèle : ~1,1 Go (téléchargé une seule fois dans ~/.cache)
#   - Fenêtre maximale : 128 tokens (les chunks > 128 tokens sont tronqués
#     silencieusement ; voir NOTE ci-dessous)
#
# NOTE sur la fenêtre de 128 tokens :
#   Ce modèle est optimisé pour des phrases et paragraphes courts.
#   Si vos chunks font 800 tokens (paramètre de 02_chunk_corpus.py),
#   seuls les 128 premiers tokens seront encodés — le reste est ignoré.
#   Options pour les corpus avec longs passages :
#     - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" : 128 tokens, plus léger
#     - "intfloat/multilingual-e5-large" : 512 tokens, meilleure qualité, plus lourd
#     - Réduire MAX_TOKENS dans 02_chunk_corpus.py à ~100 tokens
#   Pour un manuscrit historique avec passages courts, 128 tokens est souvent suffisant.
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Dimension des vecteurs produits par le modèle ci-dessus.
# À mettre à jour si vous changez de modèle :
#   paraphrase-multilingual-mpnet-base-v2  → 768
#   paraphrase-multilingual-MiniLM-L12-v2 → 384
#   multilingual-e5-large                 → 1024
EMBEDDING_DIM = 768

# Taille des lots pour la vectorisation : nombre de chunks traités en parallèle.
# Augmenter si votre machine a suffisamment de RAM (valeurs typiques : 32–128).
# Réduire si vous observez des erreurs mémoire.
EMBEDDING_BATCH_SIZE = 32

# =============================================================================
# MODÈLE LLM (Ollama)
# =============================================================================

# URL de l'API Ollama (locale, aucun trafic réseau externe)
OLLAMA_URL = "http://localhost:11434/api/generate"

# ─────────────────────────────────────────────────────────────────────────────
# CHOIX DU MODÈLE LLM LOCAL
# ─────────────────────────────────────────────────────────────────────────────
# Modifiez LLM_MODEL ci-dessous pour changer de modèle sur tout le pipeline.
# Téléchargez le modèle choisi une seule fois avec : ollama pull <nom>
#
# ┌─────────────────────────┬──────────┬──────────────────────────────────────┐
# │ Modèle                  │ RAM min. │ Notes                                │
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ deepseek-r1:14b         │ 16 Go    │ Meilleur sur raisonnement structuré  │
# │                         │          │ (Toulmin, RST). Recommandé pour 09/10│
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ gemma3:12b              │ 16 Go    │ Très bon en français académique.     │
# │                         │          │ Bon suivi de format contraint.       │
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ qwen2.5:14b             │ 16 Go    │ Défaut actuel. Bon équilibre         │
# │                         │          │ qualité/vitesse, multilingue natif.  │
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ mistral:7b-instruct     │  8 Go    │ Rapide, très bon en français.        │
# │                         │          │ Légèrement moins précis sur RST.     │
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ qwen2.5:7b              │  8 Go    │ Si RAM < 16 Go. Qualité correcte     │
# │                         │          │ pour 04/05/06/07.                    │
# ├─────────────────────────┼──────────┼──────────────────────────────────────┤
# │ llama3.1:8b             │  8 Go    │ Bon en anglais, moins performant     │
# │                         │          │ en français et sur Perelman/Toulmin. │
# └─────────────────────────┴──────────┴──────────────────────────────────────┘
#
# NOTE SUR LES TÂCHES SPÉCIALISÉES
# ─────────────────────────────────
# Scripts 04/05 (requête et rédaction) : tous les modèles 7B+ conviennent.
# Scripts 06/07 (cartographie critique) : 14B recommandé pour la taxonomie.
# Scripts 09/10 (Toulmin/Perelman) : deepseek-r1:14b ou gemma3:12b préférés
#   car ces cadres théoriques sont mieux représentés dans leurs données
#   d'entraînement. Voir la discussion épistémique dans le working paper.
#
# NOTE SUR LA SOUVERAINETÉ
# ─────────────────────────
# DeepSeek-R1 est open weights (modèle public) et tourne entièrement en local
# via Ollama — aucune donnée ne transite vers des serveurs externes, quelle
# que soit son origine. La question de la juridiction ne se pose qu'en mode
# API cloud, pas en mode local.
#
# CHANGER DE MODÈLE
# ──────────────────
# 1. ollama pull deepseek-r1:14b   (ou le modèle choisi)
# 2. Modifier LLM_MODEL ci-dessous
# 3. Relancer le script souhaité — aucune autre modification nécessaire.

LLM_MODEL = "qwen2.5:14b"

# Température du LLM : contrôle le caractère aléatoire de la réponse.
# 0.0 → réponses déterministes et factuelles (recommandé pour RAG et critique)
# 0.3–0.5 → légère créativité (utile pour la rédaction assistée, script 05)
# 1.0 → réponses créatives et variées
LLM_TEMPERATURE = 0.1

# Nombre maximum de tokens générés par le LLM dans sa réponse.
# Augmenter pour les scripts de rédaction longue (05, 06) si les réponses
# sont tronquées. Réduire pour accélérer les scripts de requête courte (04).
LLM_MAX_TOKENS = 2548

# =============================================================================
# RECHERCHE VECTORIELLE (FAISS)
# =============================================================================

# Type d'index FAISS.
# "flat"  : IndexFlatL2 — recherche exacte, recommandé jusqu'à ~50 000 chunks.
#           Aucune perte de précision, aucun entraînement requis.
# "hnsw"  : IndexHNSWFlat — très rapide sur gros volumes, léger surcoût RAM.
# "ivf"   : IndexIVFFlat — nécessite un entraînement, utile > 50 000 chunks.
FAISS_INDEX_TYPE = "flat"

# Nombre de chunks retournés par FAISS pour construire le contexte LLM.
# Un TOP_K élevé enrichit le contexte mais allonge le prompt (latence accrue).
# Valeurs typiques selon l'usage :
#   Requête ponctuelle (04)   : 10–15
#   Rédaction assistée (05)   : 15–25
#   Cartographie (06, 07)     : 20–30
TOP_K = 15

# =============================================================================
# PROMPTS SYSTÈME PAR DÉFAUT
# =============================================================================
# Ces prompts peuvent être surchargés dans chaque script _local si nécessaire.

SYSTEM_PROMPT_QUERY = """Tu es un assistant historien spécialisé dans l'analyse de corpus documentaires.
Réponds à la question en te basant UNIQUEMENT sur les extraits fournis.
N'introduis aucune information absente des extraits.
Si les extraits sont insuffisants, dis-le clairement.
Réponds dans la langue de la question."""

SYSTEM_PROMPT_WRITE = """Tu es un assistant historien expert en rédaction académique.
Rédige un passage structuré, ancré dans les sources fournies.
Cite les sources entre crochets [source, p. N] à chaque affirmation.
Ne dépasse pas les faits attestés par les extraits.
Respecte le registre académique."""

SYSTEM_PROMPT_CRITIQUE = """Tu es un historien critique expert en analyse de manuscrits.
Analyse les relations entre le texte soumis et les sources du corpus.
Structure ta réponse selon la taxonomie imposée.
Sois précis, nuancé, et cite toujours les passages concernés."""
