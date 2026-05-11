"""
00_config.py
============
Module de configuration partagé pour l'ensemble du pipeline RAG.

Rôle : Centralise tous les paramètres communs (chemins, modèles, LLM)
et expose un client LLM unique, switchable entre OpenAI et Ollama,
sans modifier les scripts appelants.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → exploration thématique
    05_rag_write.py        → écriture assistée (paragraphe → citations)
    06_map_enrich.py       → cartographie enrichissement (manuscrit entier)
    07_map_critique.py     → cartographie critique (manuscrit entier)
    ──
    00_config.py           ← CE MODULE : configuration et client LLM partagés

Pourquoi ce module ?
    Sans fichier de configuration central, chaque script définit ses propres
    chemins et paramètres — et toute modification (changer de modèle, déplacer
    un dossier) impose d'éditer chaque fichier séparément, avec des risques
    d'incohérence. Ce module règle ce problème : une seule source de vérité.

    Il expose également un objet LLMClient qui masque la différence entre
    OpenAI et Ollama. Les scripts 05, 06, 07 appellent toujours la même
    fonction generate(), quel que soit le backend configuré.

╔══════════════════════════════════════════════════════════════╗
║  CHOIX DU BACKEND LLM — À LIRE AVANT TOUT AUTRE PARAMÈTRE  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Ce pipeline utilise deux composants LLM distincts :         ║
║                                                              ║
║  1. EMBEDDINGS (toujours OpenAI, non modifiable)             ║
║     L'index FAISS a été construit avec text-embedding-3-     ║
║     large. La recherche doit utiliser le même modèle.        ║
║     → Nécessite OPENAI_API_KEY dans le fichier .env          ║
║     → Coût : très faible (fraction de centime par requête)   ║
║                                                              ║
║  2. GÉNÉRATION DE TEXTE (switchable ici)                     ║
║     Cherchez LLM_BACKEND dans la section PARAMÈTRES          ║
║     et choisissez l'une des deux valeurs :                   ║
║                                                              ║
║     LLM_BACKEND = "openai"                                   ║
║       → Génération via API OpenAI                            ║
║       → Meilleure qualité, payant à l'usage                  ║
║       → Prérequis : OPENAI_API_KEY dans .env                 ║
║                                                              ║
║     LLM_BACKEND = "ollama"                                   ║
║       → Génération via modèle local (Ollama)                 ║
║       → Gratuit, privé, fonctionne hors ligne                ║
║       → Prérequis : ollama serve + ollama pull qwen2.5:14b   ║
║       → Recommandé pour les scripts 06 et 07 (boucles        ║
║         longues sur manuscrit entier)                        ║
║                                                              ║
║  EN RÉSUMÉ : le fichier .env est toujours nécessaire.        ║
║  Seule la génération peut basculer sur Ollama.               ║
╚══════════════════════════════════════════════════════════════╝

Usage dans les autres scripts :
    from config_00 import VECTOR_DIR, CORPUS_FILE, TOP_K, LLMClient
    client = LLMClient()
    reponse = client.generate(system_prompt, user_message)

Prérequis selon le backend :
    OpenAI : fichier .env à la racine avec OPENAI_API_KEY=sk-...
    Ollama : Ollama installé et lancé (ollama serve), modèle téléchargé
             Installation : https://ollama.com
             Télécharger un modèle : ollama pull qwen2.5:14b

Structure de fichiers attendue :
    projet/
    ├── .env                  ← OPENAI_API_KEY=sk-... (si backend OpenAI)
    ├── 00_config.py          ← ce fichier
    ├── 04_rag_query.py
    ├── 05_rag_write.py
    ├── 06_map_enrich.py
    ├── 07_map_critique.py
    ├── extracted_text/
    │   ├── corpus.txt
    │   └── chunks.json
    └── vector_store/
        ├── faiss.index
        ├── embeddings.npy
        └── metadata.json
"""

# =============================================================================
# PARAMÈTRES — MODIFIEZ UNIQUEMENT CETTE SECTION
# =============================================================================

# -----------------------------------------------------------------------------
# BACKEND LLM
# -----------------------------------------------------------------------------
# Choix du moteur de génération de texte.
#
# "openai"  → API OpenAI distante. Meilleure qualité, payant à l'usage.
#             Nécessite un fichier .env avec OPENAI_API_KEY=sk-...
#
# "ollama"  → Modèle local via Ollama. Gratuit, privé, fonctionne hors ligne.
#             Nécessite : ollama serve + ollama pull <OLLAMA_MODEL>
#             Recommandé sur Apple Silicon : qwen2.5:14b ou mistral:7b
#
LLM_BACKEND = "openai"

# -----------------------------------------------------------------------------
# MODÈLES
# -----------------------------------------------------------------------------

# Modèle OpenAI pour la génération (ignoré si LLM_BACKEND = "ollama")
# "gpt-4.1-mini" : bon équilibre coût/qualité pour l'analyse documentaire
# "gpt-4.1"      : meilleure qualité, plus coûteux
OPENAI_LLM_MODEL = "gpt-4.1-mini"

# Modèle Ollama pour la génération (ignoré si LLM_BACKEND = "openai")
# Modèles recommandés pour l'analyse en français :
#   qwen2.5:14b   → excellent en français, 14B paramètres, ~9 Go RAM
#   mistral:7b    → rapide, bon en français, 7B paramètres, ~5 Go RAM
#   llama3.2:3b   → très léger, qualité moindre mais réactif
# Télécharger le modèle choisi : ollama pull qwen2.5:14b
OLLAMA_MODEL = "qwen2.5:14b"

# URL du serveur Ollama (ne pas modifier sauf configuration réseau atypique)
OLLAMA_BASE_URL = "http://localhost:11434"

# Modèle d'embeddings OpenAI — doit être identique à celui utilisé dans
# 03_build_embeddings.py. Changer ce modèle invalide l'index FAISS existant
# et impose de relancer 03_build_embeddings.py depuis zéro.
EMBEDDING_MODEL = "text-embedding-3-large"

# -----------------------------------------------------------------------------
# CHEMINS
# -----------------------------------------------------------------------------

# Dossier contenant l'index FAISS et les métadonnées
VECTOR_DIR = "vector_store"

# Fichier texte brut du corpus (produit par 01_extract_text.py)
# Utilisé pour récupérer les passages complets autour des chunks trouvés
CORPUS_FILE = "extracted_text/corpus.txt"

# -----------------------------------------------------------------------------
# RECHERCHE FAISS
# -----------------------------------------------------------------------------

# Nombre de chunks récupérés par FAISS pour construire le contexte du LLM.
# Plus TOP_K est élevé → contexte plus riche, mais aussi plus de bruit
# et des prompts plus longs (coût et latence accrus).
# Valeurs typiques pour l'analyse de manuscrit : 5–8
# Valeurs typiques pour l'exploration large : 10–20
TOP_K = 7

# -----------------------------------------------------------------------------
# GÉNÉRATION LLM
# -----------------------------------------------------------------------------

# Température : contrôle l'aléatoire de la réponse.
# 0.0 → réponse déterministe et factuelle (recommandé pour RAG et critique)
# 0.3 → légère créativité, acceptable pour l'enrichissement
# 1.0 → très créatif, déconseillé pour l'analyse documentaire
LLM_TEMPERATURE = 0.1

# Nombre maximum de tokens dans la réponse du LLM.
# 1024  : suffisant pour une analyse par paragraphe
# 2048  : utile pour les rapports globaux (06, 07)
LLM_MAX_TOKENS = 1024

# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# INITIALISATION
# =============================================================================

# Chargement de la clé API depuis le fichier .env (nécessaire pour OpenAI)
# python-dotenv permet de ne pas écrire la clé en dur dans le code,
# ce qui évite de l'exposer accidentellement (ex : publication sur GitHub).
load_dotenv()

# =============================================================================
# CLIENT LLM UNIFIÉ
# =============================================================================

class LLMClient:
    """
    Client LLM unique, compatible OpenAI et Ollama.

    Masque la différence entre les deux backends : les scripts appelants
    (05, 06, 07) n'ont pas à connaître le backend utilisé. Ils appellent
    toujours la même méthode generate().

    Attributs :
        backend  : "openai" ou "ollama" (lu depuis LLM_BACKEND)
        model    : nom du modèle actif (selon le backend)
        _client  : client interne OpenAI (None si backend Ollama)

    Exemple d'utilisation :
        from 00_config import LLMClient, TOP_K
        llm = LLMClient()
        reponse = llm.generate(
            system_prompt="Tu es un assistant historien...",
            user_message="Paragraphe : ...\n\nExtraits : ..."
        )
        print(reponse)
    """

    def __init__(self):
        self.backend = LLM_BACKEND

        if self.backend == "openai":
            # Import conditionnel : si openai n'est pas installé mais que
            # l'utilisateur veut Ollama, l'import ne plante pas au démarrage.
            from openai import OpenAI
            self._client = OpenAI()  # Lit OPENAI_API_KEY depuis l'environnement
            self.model = OPENAI_LLM_MODEL
            print(f"[config] Backend LLM : OpenAI ({self.model})")

        elif self.backend == "ollama":
            # Ollama expose une API compatible OpenAI sur localhost.
            # On utilise donc le client openai avec une base_url différente,
            # ce qui évite d'ajouter une dépendance supplémentaire.
            from openai import OpenAI
            self._client = OpenAI(
                base_url=f"{OLLAMA_BASE_URL}/v1",
                api_key="ollama",  # Valeur fictive requise par le client openai
            )
            self.model = OLLAMA_MODEL
            print(f"[config] Backend LLM : Ollama local ({self.model})")
            print(f"         URL         : {OLLAMA_BASE_URL}")
            self._verifier_ollama()

        else:
            raise ValueError(
                f"LLM_BACKEND inconnu : '{self.backend}'.\n"
                "Valeurs acceptées : 'openai' ou 'ollama'."
            )

    def _verifier_ollama(self):
        """
        Vérifie qu'Ollama est accessible et que le modèle est disponible.

        Appelé automatiquement à l'initialisation si backend = "ollama".
        Affiche un avertissement clair si Ollama n'est pas lancé, plutôt
        que de laisser planter le script plus tard lors du premier appel.

        Ne lève pas d'exception : un avertissement suffit à ce stade,
        l'erreur réelle surviendra lors du premier appel à generate().
        """
        try:
            import urllib.request
            urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            print(f"         Ollama      : ✓ accessible")
        except Exception:
            print(
                f"\n⚠️  Ollama ne semble pas lancé sur {OLLAMA_BASE_URL}.\n"
                "   Lancez Ollama avec : ollama serve\n"
                f"   Puis vérifiez que le modèle est disponible : ollama list\n"
                f"   Pour télécharger le modèle : ollama pull {self.model}\n"
            )

    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Envoie un prompt au LLM et retourne la réponse textuelle.

        Compatible avec OpenAI et Ollama sans modification du code appelant.
        La structure du prompt (system + user) est standard pour les deux.

        Args:
            system_prompt : Instructions de rôle et de contraintes pour le LLM.
                            Exemple : "Tu es un historien critique. Ne cite
                            que des passages présents dans les extraits."
            user_message  : Contenu de la requête, typiquement le paragraphe
                            du manuscrit suivi des extraits du corpus.

        Returns:
            Réponse textuelle du LLM (chaîne de caractères).

        Raises:
            Exception : Toute erreur réseau ou API est propagée telle quelle
                        pour être gérée par le script appelant.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content


# =============================================================================
# FONCTIONS UTILITAIRES PARTAGÉES
# =============================================================================

def charger_corpus() -> str:
    """
    Charge le fichier corpus.txt en mémoire.

    Utilisé par les scripts 04, 05, 06, 07 pour récupérer les passages
    complets autour des chunks trouvés par FAISS (fonction load_passage).

    Returns:
        Contenu textuel complet de corpus.txt.

    Raises:
        FileNotFoundError : Si corpus.txt est absent (01_extract_text.py
                            n'a pas encore été lancé).
    """
    corpus_path = Path(CORPUS_FILE)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {corpus_path.resolve()}\n"
            "Lancez d'abord 01_extract_text.py pour générer ce fichier."
        )
    return corpus_path.read_text(encoding="utf-8")


def recuperer_passage(corpus_text: str, source: str, page: int) -> str:
    """
    Extrait le passage complet d'une page depuis le texte du corpus.

    Plutôt que de retourner uniquement le chunk (fragment potentiellement
    tronqué), cette fonction restitue le texte intégral de la section
    balisée [source — page N], ce qui donne davantage de contexte au LLM.

    Cette fonction est une version centralisée de load_passage() présente
    dans 04_rag_query.py. En la plaçant ici, tous les scripts partagent
    exactement la même logique de récupération.

    Args:
        corpus_text : Contenu de corpus.txt (chargé via charger_corpus()).
        source      : Nom du fichier source (ex : "article_dupont_2018.pdf").
        page        : Numéro de page.

    Returns:
        Texte de la section, ou chaîne vide si l'en-tête est introuvable.
    """
    marker = f"[{source} — page {page}]"
    start  = corpus_text.find(marker)

    if start == -1:
        return ""

    start += len(marker)
    end    = corpus_text.find("[", start)  # Prochain en-tête = fin de section
    return corpus_text[start:end].strip() if end != -1 else corpus_text[start:].strip()


def formater_extraits(extraits: list[dict]) -> str:
    """
    Formate une liste d'extraits en bloc texte lisible pour le LLM.

    Chaque extrait est séparé par une ligne de tirets pour aider le LLM
    à distinguer les sources. Ce formatage est identique dans tous les
    scripts pour garantir la cohérence des prompts.

    Args:
        extraits : Liste de dicts avec les clés :
                   - "source"  : nom du fichier PDF
                   - "page"    : numéro de page
                   - "passage" : texte extrait

    Returns:
        Bloc texte formaté, prêt à insérer dans un prompt LLM.

    Exemple de sortie :
        SOURCE : dupont_2018.pdf (p. 12)
        Le concept de frontière chez les historiens du XIXe siècle...

        ---

        SOURCE : martin_2021.pdf (p. 4)
        La question de la périodisation reste ouverte...
    """
    blocs = []
    for e in extraits:
        entete = f"SOURCE : {e['source']} (p. {e['page']})"
        blocs.append(f"{entete}\n{e['passage']}")
    return "\n\n---\n\n".join(blocs)


# =============================================================================
# VÉRIFICATION AU DÉMARRAGE (mode script direct)
# =============================================================================

def _afficher_configuration():
    """
    Affiche un résumé de la configuration active.
    Utile pour vérifier rapidement l'état du système avant de lancer
    un script de traitement.
    """
    print("=" * 55)
    print("  CONFIGURATION RAG — résumé")
    print("=" * 55)
    print(f"  Backend LLM      : {LLM_BACKEND}")

    if LLM_BACKEND == "openai":
        cle = os.getenv("OPENAI_API_KEY", "")
        statut = "✓ clé présente" if cle else "✗ clé manquante (vérifiez .env)"
        print(f"  Modèle LLM       : {OPENAI_LLM_MODEL}")
        print(f"  Clé API          : {statut}")
    else:
        print(f"  Modèle LLM       : {OLLAMA_MODEL}")
        print(f"  URL Ollama       : {OLLAMA_BASE_URL}")

    print(f"  Modèle embedding : {EMBEDDING_MODEL}")
    print(f"  TOP_K            : {TOP_K}")
    print(f"  Température      : {LLM_TEMPERATURE}")
    print(f"  Max tokens       : {LLM_MAX_TOKENS}")
    print(f"  Corpus           : {CORPUS_FILE}")
    print(f"  Vector store     : {VECTOR_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    # Lancé directement (python 00_config.py) : affiche la configuration
    # et tente d'initialiser le client LLM pour vérifier que tout fonctionne.
    _afficher_configuration()
    print("\nTest d'initialisation du client LLM…")
    try:
        llm = LLMClient()
        print("✓ Client LLM initialisé avec succès.\n")
        print("Ce fichier est prêt à être importé par les scripts 04 à 07.")
    except Exception as e:
        print(f"✗ Erreur : {e}")
