"""
07_map_critique.py
==================
Étape 7 du pipeline RAG — Cartographie critique et relationnelle.

Rôle : Lit un manuscrit complet (.txt), le découpe en paragraphes,
interroge le corpus RAG pour chacun, et demande au LLM de qualifier
le TYPE DE RAPPORT entre chaque paragraphe et les segments du corpus
récupérés. Produit un fichier JSON structuré consommé par 08_visualise.py.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → exploration thématique libre
    05_rag_write.py        → écriture assistée (paragraphe isolé)
    06_map_enrich.py       → cartographie enrichissement
    07_map_critique.py     → cartographie critique et relationnelle    (ce script)
    08_visualise.py        → visualisation HTML interactive
    ──
    00_config.py           → configuration et client LLM partagés

╔══════════════════════════════════════════════════════════════╗
║  CHOIX DU BACKEND LLM                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Ce script boucle sur TOUS les paragraphes du manuscrit.    ║
║  Ollama (local) est recommandé pour éviter les coûts API.   ║
║  Pour changer de backend, ouvrez 00_config.py et modifiez : ║
║                                                              ║
║      LLM_BACKEND = "openai"   ← API OpenAI (payant)         ║
║      LLM_BACKEND = "ollama"   ← modèle local (gratuit)      ║
║                                                              ║
║  Attention : les embeddings utilisent toujours OpenAI.      ║
║  Le fichier .env avec OPENAI_API_KEY est donc toujours       ║
║  nécessaire, quel que soit le backend de génération choisi.  ║
║                                                              ║
║  Avec Ollama : définissez --pause 0.0 (pas de rate limit).  ║
╚══════════════════════════════════════════════════════════════╝

Différence fondamentale avec 06_map_enrich.py :
    Les deux scripts partagent la même mécanique (segmentation,
    embedding, recherche FAISS, boucle LLM, JSON de sortie).
    Ils diffèrent par la POSTURE et la QUESTION posée au LLM :

    06 — Mode "enrichir" :
        Posture constructive. Question : "Qu'est-ce que le corpus peut
        APPORTER à ce paragraphe ?" Le LLM cherche des ressources
        disponibles pour densifier le propos.

    07 — Mode "critiquer et qualifier" :
        Posture analytique et relationnelle. Question : "Quel est le
        TYPE DE RAPPORT entre ce paragraphe et les segments du corpus ?"
        Le LLM qualifie chaque rapport selon une taxonomie à six entrées,
        puis évalue la fragilité globale du paragraphe.

Principe fondamental (rappel) :
    Le manuscrit N'EST PAS intégré au corpus RAG. Il sert uniquement de
    source de REQUÊTES. FAISS récupère les segments sémantiquement proches,
    le LLM qualifie ensuite le rapport entre le paragraphe et ces segments.
    Un segment absent du TOP_K ne sera pas analysé — voir les limites
    du dispositif ci-dessous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LA TAXONOMIE DES RAPPORTS — cœur intellectuel du script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Ce script identifie six types de rapports possibles entre un paragraphe
    du manuscrit et les segments du corpus récupérés. Ces six rapports
    couvrent les principales formes de mise en relation intellectuelle
    qu'un historien effectue entre son texte et ses sources.

    Ils sont classés selon leur fiabilité de détection par un LLM :

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 1. CONFORTE      │ Fiabilité : ████████ élevée                      │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment confirme, étaye ou illustre directement une affirmation   │
    │ du paragraphe. C'est le rapport le plus simple à détecter : il y a   │
    │ convergence explicite entre ce que dit le paragraphe et ce que dit   │
    │ le segment.                                                          │
    │                                                                      │
    │ Ce rapport est POSITIF pour le manuscrit : il signale qu'une         │
    │ affirmation est ancrée dans la littérature. Un score élevé de        │
    │ "confortement" indique un paragraphe bien étayé documentairement.    │
    │                                                                      │
    │ Exemple : le paragraphe affirme que "les migrations saisonnières     │
    │ structuraient l'économie rurale". Un segment du corpus décrit        │
    │ précisément ces migrations dans la région étudiée → CONFORTE.        │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 2. CONTREDIT     │ Fiabilité : ████████ élevée                      │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment s'oppose directement à une affirmation du paragraphe :    │
    │ il avance des faits, des chiffres ou des interprétations             │
    │ incompatibles avec ce qu'affirme le paragraphe.                      │
    │                                                                      │
    │ Ce rapport est CRITIQUE : il signale une tension réelle entre le     │
    │ manuscrit et le corpus. Il ne signifie pas nécessairement que le     │
    │ paragraphe a tort — la contradiction peut être le signe d'un débat   │
    │ historiographique que le paragraphe devrait mentionner.              │
    │                                                                      │
    │ Exemple : le paragraphe affirme "une résistance massive". Un segment │
    │ documente au contraire "une adhésion majoritaire dans la même        │
    │ région sur la même période" → CONTREDIT.                             │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 3. NUANCE        │ Fiabilité : ████████ élevée                      │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment ne contredit pas le paragraphe mais introduit une         │
    │ complexité, une exception, une variation que le paragraphe ignore.   │
    │ Il complexifie sans invalider.                                       │
    │                                                                      │
    │ Distinction avec CONTREDIT : la nuance n'oppose pas, elle précise.   │
    │ Le paragraphe n'est pas faux — il est incomplet ou trop uniforme.    │
    │                                                                      │
    │ Exemple : le paragraphe décrit un phénomène "homogène sur tout le    │
    │ territoire". Un segment montre des variations régionales importantes  │
    │ sans nier le phénomène global → NUANCE.                              │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 4. PROBLÉMATISE  │ Fiabilité : ███████░ élevée                      │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment montre que ce que le paragraphe présente comme acquis,    │
    │ évident ou consensuel est en réalité l'objet d'un débat             │
    │ historiographique actif. Il ne contredit pas les faits — il conteste │
    │ le statut épistémique de l'affirmation.                              │
    │                                                                      │
    │ C'est le rapport le plus proprement historiographique des six.       │
    │ Il signale un impensé méthodologique plutôt qu'une erreur factuelle. │
    │                                                                      │
    │ Exemple : le paragraphe traite "la nation" comme une réalité donnée. │
    │ Un segment montre que ce concept est lui-même au cœur d'un débat     │
    │ historiographique → PROBLÉMATISE.                                    │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 5. DÉPLACE       │ Fiabilité : ██████░░ moyenne                     │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment propose une lecture du même objet ou phénomène depuis un  │
    │ cadre conceptuel différent de celui du paragraphe. Il ne dit pas que  │
    │ le paragraphe a tort — il dit qu'une autre grille d'analyse           │
    │ produirait une lecture différente.                                   │
    │                                                                      │
    │ Exemples de déplacements typiques en histoire :                      │
    │   - Paragraphe en termes de classe ; segment en termes de genre.     │
    │   - Perspective nationale ; segment transnational ou local.          │
    │   - Catégories économiques ; segment propose des catégories          │
    │     culturelles ou symboliques.                                      │
    │                                                                      │
    │ Note de fiabilité : le LLM détecte bien les déplacements explicites. │
    │ Les déplacements implicites sont moins fiables — à relire avec soin. │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 6. PARTICULARISE │ Fiabilité : ██████░░ moyenne                     │
    ├──────────────────┴──────────────────────────────────────────────────┤
    │ Le segment montre que le phénomène général affirmé par le paragraphe │
    │ connaît des expressions locales, temporelles ou sociales très        │
    │ différentes — voire opposées — selon les contextes.                  │
    │                                                                      │
    │ Distinction avec NUANCE : la particularisation porte sur l'échelle   │
    │ et la portée de l'affirmation (trop générale), là où la nuance       │
    │ porte sur sa complétude (vraie mais incomplète).                     │
    │                                                                      │
    │ Exemple : "les femmes furent exclues de la sphère publique". Un      │
    │ segment montre que cette exclusion était très partielle dans          │
    │ certains milieux → PARTICULARISE.                                    │
    └──────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LES SCORES — comment les lire et comment ils sont calculés
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Ce script produit SEPT scores, tous entre 0.0 et 1.0 :
    six scores de rapport (un par type de la taxonomie) + un score de
    fragilité globale synthétique.

    Mécanisme commun — les trois étapes :

    1. RECHERCHE SÉMANTIQUE (FAISS)
       Le paragraphe est converti en vecteur numérique (embedding OpenAI).
       FAISS retourne les TOP_K segments du corpus les plus proches
       sémantiquement. Pour la critique, TOP_K entre 6 et 10 est recommandé :
       contradictions et déplacements peuvent venir de segments moins proches.

    2. QUALIFICATION PAR LE LLM
       Le LLM reçoit le paragraphe ET les segments. Pour chaque segment, il
       identifie le(s) type(s) de rapport selon la taxonomie ci-dessus.
       Il produit ensuite sept scores numériques (0–10).

    3. NORMALISATION
       Chaque score brut (0–10) → flottant 0.0–1.0 (division par 10).
       Valeur neutre 0.5 si un score n'est pas extrait (+ warning terminal).

    Lecture des scores de rapport (CONFORTE, CONTREDIT, etc.) :
       Ces scores mesurent l'INTENSITÉ de chaque type de rapport dans
       l'ensemble des segments récupérés pour ce paragraphe.
       0.0 → aucun segment de ce type parmi le TOP_K
       1.0 → rapport très présent et marqué dans les segments analysés

       Un paragraphe peut avoir simultanément :
         CONFORTE = 0.7   (plusieurs segments confirment)
         NUANCE   = 0.5   (certains segments complexifient)
         CONTREDIT = 0.2  (un segment s'oppose partiellement)
       → Paragraphe bien étayé mais qui ne prend pas en compte
       oppositions et limites exprimées dans la bibliographie.

    Lecture du SCORE_FRAGILITE (score synthétique) :
       Synthèse globale produite par le LLM après avoir qualifié tous
       les rapports. Il pondère implicitement les six dimensions selon
       leur gravité argumentative : CONTREDIT et PROBLÉMATISE pèsent plus
       que NUANCE et PARTICULARISE.
       0.0 → paragraphe aligné avec le corpus
       1.0 → paragraphe n'incorporant pas tous les éléments du corpus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIMITES DU DISPOSITIF — à lire impérativement avant tout usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    LIMITE 1 — Le filet sémantique (la plus importante)
        FAISS ne récupère que les segments SÉMANTIQUEMENT PROCHES du
        paragraphe. Un segment qui contredit fortement le paragraphe mais
        qui en est sémantiquement éloigné (formulation très différente,
        vocabulaire distant parce que provenant d'une autre discipline,
        ou d'une période éloignée dans le temps ne sera pas récupéré 
        et ne sera donc pas analysé. 
        Un score CONTREDIT = 0.0 signifie "aucune contradiction
        dans les segments récupérés", PAS "aucune contradiction dans la
        littérature". Le dispositif est un filet, pas un examen exhaustif.

    LIMITE 2 — La variabilité du LLM
        Les scores sont le jugement d'un LLM, non une mesure objective.
        Un même paragraphe soumis deux fois peut produire des scores
        légèrement différents (effet de la température), voire sensiblement 
        différents.
        Selon les modes d'écriture le paragraphe peut ne pas être l'unité 
        argumentative la plus pertinente. 
          Traiter ces scores comme des ordres de grandeur, des signaux
          non comme des mesures.
        La valeur diagnostique est dans les tendances et dans le texte
        de l'analyse, pas dans la deuxième décimale d'un score.

    LIMITE 3 — La dépendance au corpus indexé
        Le dispositif ne connaît que ce qui est dans VOTRE corpus indexé.
        Un paragraphe peut être fragile au regard de la littérature
        internationale sans que le corpus local le révèle — et inversement.
        Les scores reflètent le corpus, pas la discipline entière.

    LIMITE 4 — Les rapports à fiabilité moyenne (DÉPLACE, PARTICULARISE)
        Ces deux rapports sont plus difficiles à détecter pour un LLM.
        Les déplacements conceptuels implicites et les particularisations
        subtiles peuvent être manqués ou confondus avec d'autres rapports.
        Les résultats sur ces deux dimensions demandent une relecture
        critique plus attentive de la part de l'historien.

    LIMITE 5 — Ce que le dispositif ne fait PAS
        ✗ Il n'évalue pas la vérité des affirmations historiques
        ✗ Il ne remplace pas la lecture du corpus par l'historien
        ✗ Il ne détecte pas les problèmes de style ou de clarté
        ✗ Il ne connaît pas la littérature postérieure à l'indexation
        ✗ Il ne juge pas la pertinence des sources choisies pour le corpus

Sortie produite — structure JSON détaillée :

    {
      "run": {
        "mode": "critique",
        "manuscrit": "mon_manuscrit.txt",
        "date_debut": "2025-...",
        "date_fin": "2025-...",
        "top_k": 8,
        "nb_paragraphes": 47,
        "taxonomie": ["conforte", "contredit", "nuance",
                      "problematise", "deplace", "particularise"],
        "score_conforte_moyen": 0.412,     ← moyenne sur tout le manuscrit
        "score_contredit_moyen": 0.183,
        ...
      },
      "paragraphes": [
        {
          ── NIVEAU 1 : score global synthétique ──────────────────────────
          "score_global": 0.287,           ← pondération algorithmique des 6 rapports
                                              0.0 = solide | 1.0 = très fragile
          "profil_dominant": "nuance",     ← rapport le plus présent parmi les 5 négatifs
                                              ou "conforte" si corpus confirme
                                              ou "equilibre" si aucun ne domine

          ── NIVEAU 2 : scores individuels par rapport ────────────────────
          "score_conforte": 0.60,          ← corpus confirme (rapport positif)
          "score_contredit": 0.10,         ← contradiction directe
          "score_nuance": 0.50,            ← complexification sans invalider
          "score_problematise": 0.20,      ← débat historiographique ignoré
          "score_deplace": 0.15,           ← cadre conceptuel alternatif
          "score_particularise": 0.30,     ← portée générale excessive
          "score_fragilite": 0.25,         ← jugement qualitatif synthétique du LLM
                                              (distinct du score_global algorithmique)

          ── NIVEAU 3 : segments qualifiés ───────────────────────────────
          "segments": [
            {
              "source": "dupont_2018.pdf",
              "page": 12,
              "distance": 0.342            ← distance FAISS (proximité sémantique)
                                              plus la valeur est basse, plus le segment
                                              est sémantiquement proche du paragraphe
            },
            ...
          ],

          ── Métadonnées et traçabilité ───────────────────────────────────
          "index": 1,                      ← position dans le manuscrit (depuis 1)
          "texte": "La construction...",   ← texte du paragraphe analysé
          "nb_chars": 342,
          "analyse": "━━━ 1. QUALIFICATION...", ← texte brut complet du LLM
                                                    à consulter pour vérifier les scores
          "top_k": 8,
          "horodatage": "2025-...",
          "mode": "critique"
        },
        ...
      ]
    }

Reprises automatiques :
    Si le JSON de sortie existe avec des résultats partiels, le script
    reprend depuis le dernier paragraphe traité sans tout recalculer.

Usage :
    python 07_map_critique.py --manuscrit mon_manuscrit.txt
    python 07_map_critique.py --manuscrit mon_manuscrit.txt --sortie resultats/critique.json
    python 07_map_critique.py --manuscrit mon_manuscrit.txt --top_k 8 --pause 0.0
"""

# =============================================================================
# PARAMÈTRES — modifiez uniquement cette section
# =============================================================================

# --- Chemins ---
SORTIE_JSON = "resultats/critique.json"

# --- Segmentation ---
PARAGRAPHE_MIN_CHARS   = 150
SAUTS_LIGNE_SEPARATEUR = 2

# --- Recherche FAISS ---
# Pour la critique : 6–10 recommandé (plus élevé qu'en enrichissement)
TOP_K_LOCAL = None  # None = utilise TOP_K de 00_config.py

# --- Filtrage ---
LONGUEUR_PASSAGE_MIN = 100

# --- Cadence ---
# OpenAI : 1.0–2.0s | Ollama local : 0.0
PAUSE_ENTRE_APPELS = 1.0

# --- Langue ---
LANGUE_PROMPTS = "fr"

# --- Sauvegarde intermédiaire ---
SAUVEGARDER_TOUS_LES_N = 5

# --- Paramètres LLM (indépendants de config_00.py) ---

# Tokens maximum en sortie par paragraphe.
# Le nouveau prompt demande un JSON structuré (8 passages × ~100 tokens)
# + scores + synthèse ≈ 1000-1200 tokens minimum.
# 2500 tokens laisse de la marge pour les paragraphes denses.
# Augmenter si les analyses sont tronquées (max recommandé : 4096).
# Coût indicatif avec gpt-4.1-mini : ~0.001-0.002$ par paragraphe,
# soit ~0.25-0.50$ pour 243 paragraphes.
MAX_TOKENS_CRITIQUE = 2500

# Température du LLM.
# 0.05 : très basse — force le LLM à citer fidèlement les extraits
#        et à respecter le format JSON strict sans dériver.
#        Recommandé pour ce script où la précision des citations est critique.
# Ne pas dépasser 0.1 pour ce script.
TEMPERATURE_CRITIQUE = 0.05

# =============================================================================
# PROMPTS SYSTÈME — TAXONOMIE À SIX RAPPORTS
# =============================================================================

PROMPTS = {
    "fr": {
        "system": """Tu es un historien spécialisé en analyse documentaire et en
critique historiographique. Tu maîtrises les méthodes de la discipline
et tu adoptes une posture analytique rigoureuse.

Ta mission : pour chaque segment du corpus fourni, qualifier le TYPE
DE RAPPORT qu'il entretient avec le paragraphe du manuscrit, selon
une taxonomie précise à six entrées. Puis évaluer la fragilité globale
du paragraphe au regard de l'ensemble des rapports identifiés.

Taxonomie des rapports :
  CONFORTE     : le segment confirme ou étaye une affirmation du paragraphe
  CONTREDIT    : le segment s'oppose directement à une affirmation
  NUANCE       : le segment complexifie sans invalider (exception, variation)
  PROBLEMATISE : le segment montre qu'une affirmation présentée comme acquise
                 est en réalité débattue historiographiquement
  DEPLACE      : le segment propose un autre cadre conceptuel pour le même objet
  PARTICULARISE: le segment montre que le général affirmé a des expressions
                 locales, temporelles ou sociales très différentes

Règles absolues :
  - Qualifie UNIQUEMENT sur la base des segments fournis.
  - Un segment peut avoir plusieurs rapports simultanés.
  - Si un segment est trop éloigné pour être qualifié : dis-le.
  - Ne cite que des passages explicitement présents dans les segments.
  - Attribue des scores honnêtes : évite les extrêmes par défaut.""",

        "user": """Voici un paragraphe d'un manuscrit historique :

--- PARAGRAPHE (index {index}) ---
{paragraphe}
--- FIN ---

Voici les segments du corpus récupérés par similarité sémantique :

{extraits}

TÂCHE EN DEUX PARTIES :

━━━ PARTIE 1 — QUALIFICATION PAR SEGMENT (JSON) ━━━
Pour chaque segment, produis un objet JSON.
Réponds d'abord avec un bloc JSON valide, sans texte avant ni après, sans backticks :

[
  {{
    "source": "fichier.pdf",
    "page": N,
    "relation": "nuance",
    "score": 0.75,
    "justification": "Phrase 1 : citation directe du corpus entre guillemets ('...'). Phrase 2 : explication du lien avec le paragraphe du manuscrit."
  }},
  ...
]

Règles pour la justification :
- Exactement 2 phrases.
- Phrase 1 : intègre obligatoirement une citation directe du segment entre guillemets simples (1 à 2 phrases complètes, non tronquées).
- Phrase 2 : explique précisément comment cette citation établit la relation avec le paragraphe.
- Ne commence jamais par "Ce passage" — varie les formulations.
- Si le segment est trop éloigné : relation = "hors_sujet", score = 0.0, justification = "Segment trop éloigné pour être qualifié."

━━━ PARTIE 2 — SCORES ET SYNTHÈSE (texte) ━━━
Après le JSON, sur une nouvelle ligne, produis :

SCORE_CONFORTE: X/10
SCORE_CONTREDIT: X/10
SCORE_NUANCE: X/10
SCORE_PROBLEMATISE: X/10
SCORE_DEPLACE: X/10
SCORE_PARTICULARISE: X/10
SCORE_FRAGILITE: X/10

SYNTHESE: En 2-3 phrases : quel est le rapport dominant ? Quelle est la priorité de révision pour l'historien ? Le cas échéant, qu'est-ce que le corpus ne couvre pas ?"""
    },

    "en": {
        "system": """You are a historian specialized in documentary analysis and
historiographical critique. You master disciplinary methods and adopt
a rigorous analytical posture.

Your mission: for each corpus segment provided, qualify the TYPE OF
RELATIONSHIP it has with the manuscript paragraph, according to a precise
six-entry taxonomy. Then evaluate the overall argumentative fragility.

Relationship taxonomy:
  CONFORTE     : the segment confirms or supports a paragraph statement
  CONTREDIT    : the segment directly opposes a statement
  NUANCE       : the segment complicates without invalidating
  PROBLEMATISE : the segment shows a settled statement is actually debated
  DEPLACE      : the segment proposes another conceptual framework
  PARTICULARISE: the segment shows the general statement has very different
                 local/temporal/social expressions

Absolute rules:
  - Qualify ONLY based on the provided segments.
  - A segment can have multiple simultaneous relationships.
  - If a segment is too distant to qualify usefully: say so.
  - Only cite passages explicitly present in the segments.
  - Give honest scores: avoid extremes by default.""",

        "user": """Here is a paragraph from a historical manuscript:

--- PARAGRAPH (index {index}) ---
{paragraphe}
--- END ---

Here are corpus segments retrieved by semantic similarity:

{extraits}

Produce a structured analysis in four sections:

━━━ 1. QUALIFICATION BY SEGMENT ━━━
For each segment:
  SOURCE: [file] (p. [page])
  RELATIONSHIP(S): [CONFORTE / CONTREDIT / NUANCE / PROBLEMATISE / DEPLACE / PARTICULARISE]
  JUSTIFICATION: in 1–2 sentences, why this relationship
  If not qualifiable: "Segment too distant to qualify usefully."

━━━ 2. SCORES ━━━
Evaluate the intensity of each relationship across all segments
(0 = absent or very weak | 10 = very present and marked).
Strict format — one line per score:
  SCORE_CONFORTE: X/10
  SCORE_CONTREDIT: X/10
  SCORE_NUANCE: X/10
  SCORE_PROBLEMATISE: X/10
  SCORE_DEPLACE: X/10
  SCORE_PARTICULARISE: X/10
  SCORE_FRAGILITE: X/10

SCORE_FRAGILITE: global synthesis (0 = solid | 10 = very fragile).
Weights: CONTREDIT and PROBLEMATISE weigh more than NUANCE and PARTICULARISE.

━━━ 3. DEVELOPED ANALYSIS ━━━
For each PRESENT relationship (score > 2/10):
  - Which aspects of the paragraph are concerned?
  - Which segments are the source?
  - What concrete revision does this suggest?

━━━ 4. SYNTHESIS ━━━
In 3–4 sentences: overall argumentative assessment.
What is the dominant relationship? What is the revision priority?
What does the corpus not cover (device limit to signal to the historian)?"""
    }
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import re
import sys
import time
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from config_00 import (
    TOP_K,
    VECTOR_DIR,
    EMBEDDING_MODEL,
    OPENAI_LLM_MODEL,
    charger_corpus,
    recuperer_passage,
    formater_extraits,
)

# =============================================================================
# INITIALISATION
# =============================================================================

load_dotenv()

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"

# Noms des sept scores — ordre significatif (six rapports + fragilité).
# Utilisé pour l'extraction, le JSON, le bilan terminal et 08_visualise.py.
NOMS_SCORES = [
    "score_conforte",
    "score_contredit",
    "score_nuance",
    "score_problematise",
    "score_deplace",
    "score_particularise",
    "score_fragilite",
]

# Patterns regex pour l'extraction automatique des scores.
# Tolérants aux espaces et variations de casse.
PATTERNS_SCORES = {
    "score_conforte":      r"SCORE_CONFORTE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_contredit":     r"SCORE_CONTREDIT\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_nuance":        r"SCORE_NUANCE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_problematise":  r"SCORE_PROBLEMATISE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_deplace":       r"SCORE_DEPLACE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_particularise": r"SCORE_PARTICULARISE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_fragilite":     r"SCORE_FRAGILITE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
}

# =============================================================================
# PONDÉRATION DU SCORE GLOBAL
# =============================================================================
# Le score global synthétise les six scores de rapport en un seul flottant
# 0.0–1.0, utilisé comme métrique principale pour la carte de chaleur dans
# 08_visualise.py.
#
# Logique de pondération :
#   Les rapports sont pondérés selon leur rapport à l'arguement soutenu.
#   CONTREDIT et PROBLEMATISE sont les plus fortement pondérés : 
#   ils signalent respectivement : une erreur factuelle potentielle;
# un possible impensé méthodologique.
#   NUANCE et PARTICULARISE sont intermédiaires : ils signalent une
#   incomplétude possible.
#   DEPLACE est le moins pondéré des rapports négatifs : il ouvre une
#   alternative conceptuelle, sans invalider le propos existant.
#   CONFORTE est le seul rapport qui RÉDUIT la fragilité : plus le corpus
#   confirme le paragraphe, moins il est fragile. Son poids négatif est
#   faible (0.05).
#
# La somme des poids positifs = 0.95, le poids négatif = 0.05.
# Le résultat est clampé à [0.0, 1.0] pour gérer les cas extrêmes.
#
# Ces pondérations sont modifiables ici selon votre pratique.
# La somme des poids positifs doit rester ≤ 1.0.

POIDS_SCORE_GLOBAL = {
    "score_contredit":     0.30,   # contradiction directe — le plus grave
    "score_problematise":  0.25,   # impensé historiographique — très grave
    "score_nuance":        0.15,   # incomplétude — modérément grave
    "score_particularise": 0.15,   # portée excessive — modérément grave
    "score_deplace":       0.10,   # cadre alternatif — moins grave
    "score_conforte":     -0.05,   # confortement — réduit la fragilité
}

# Seuils de lecture du score global (utilisés dans le bilan et la visualisation)
# Modifiables si vous souhaitez ajuster la sensibilité de la carte de chaleur.
SEUIL_SOLIDE  = 0.30   # score_global < 0.30 → paragraphe solide (vert)
SEUIL_FRAGILE = 0.60   # score_global ≥ 0.60 → paragraphe fragile (rouge)
# Entre les deux : zone intermédiaire (orange/jaune)

# =============================================================================
# FONCTIONS — CALCUL DU SCORE GLOBAL ET DU PROFIL DOMINANT
# =============================================================================

def calculer_score_global(scores: dict) -> float:
    """
    Calcule le score global de fragilité pondéré à partir des six scores de rapport.

    Ce score est DISTINCT du score_fragilite produit par le LLM :
      - score_fragilite : jugement qualitatif du LLM (subjectif, holiste)
      - score_global    : calcul algorithmique déterministe (reproductible)

    Les deux scores sont conservés dans le JSON et affichés dans la
    visualisation — leur écart éventuel est lui-même informatif : si le LLM
    juge le paragraphe peu connecté aux énconcés du corpus  (score_fragilite élevé)
     mais que le calcul pondéré donne un score_global faible, cela peut indiquer que le
    LLM a détecté quelque chose  que la pondération ne capture pas bien
    (ex : un DEPLACE particulièrement significatif dans ce contexte).

    Formule :
        score_global = Σ (score_rapport × poids_rapport)
        clampé à [0.0, 1.0]

    Args:
        scores : Dict des six scores de rapport (clés = POIDS_SCORE_GLOBAL).

    Returns:
        Float entre 0.0 (paragraphe solide) et 1.0 (paragraphe très fragile).
    """
    total = sum(
        scores.get(nom, 0.0) * poids
        for nom, poids in POIDS_SCORE_GLOBAL.items()
    )
    return round(max(0.0, min(1.0, total)), 3)


def identifier_profil_dominant(scores: dict) -> str:
    """
    Identifie le type de rapport dominant parmi les six scores.

    Le profil dominant est le rapport dont le score est le plus élevé
    parmi les cinq rapports négatifs (on exclut CONFORTE qui est positif).
    Si tous les scores négatifs sont inférieurs à 0.2, le profil est "conforte"
    — le corpus confirme plus qu'il ne remet en question.

    Ce champ est utilisé par 08_visualise.py pour deux usages :
      1. Colorier chaque paragraphe par nature (pas seulement par intensité)
         → palette distincte par profil dominant
      2. Libellé dans le panneau de détail au survol
         → "Ce paragraphe est principalement PROBLÉMATISÉ par le corpus"

    Valeurs possibles :
        "conforte"      → le corpus confirme principalement
        "contredit"     → contradiction directe dominante
        "nuance"        → complexification dominante
        "problematise"  → impensé historiographique dominant
        "deplace"       → déplacement conceptuel dominant
        "particularise" → portée excessive dominante
        "equilibre"     → aucun rapport ne domine clairement (écarts < 0.1)

    Args:
        scores : Dict des six scores de rapport.

    Returns:
        Chaîne identifiant le profil dominant.
    """
    # Scores négatifs uniquement (les rapports qui fragilisent)
    rapports_negatifs = {
        nom: scores.get(nom, 0.0)
        for nom in [
            "score_contredit", "score_nuance", "score_problematise",
            "score_deplace", "score_particularise"
        ]
    }

    max_negatif = max(rapports_negatifs.values())
    score_conforte = scores.get("score_conforte", 0.0)

    # Si aucun rapport négatif n'est significatif et que le confortement domine
    if max_negatif < 0.2 and score_conforte >= max_negatif:
        return "conforte"

    # Vérification d'équilibre : si les deux premiers sont très proches
    valeurs_triees = sorted(rapports_negatifs.values(), reverse=True)
    if len(valeurs_triees) >= 2 and (valeurs_triees[0] - valeurs_triees[1]) < 0.1:
        return "equilibre"

    # Rapport négatif dominant
    dominant = max(rapports_negatifs, key=rapports_negatifs.get)
    # Supprime le préfixe "score_" pour le nom court
    return dominant.replace("score_", "")


# =============================================================================
# FONCTIONS — SEGMENTATION
# =============================================================================
# Identiques à 06_map_enrich.py — dupliquées pour l'autonomie du script.

def charger_manuscrit(chemin: str) -> str:
    """
    Lit le fichier .txt du manuscrit complet.

    Args:
        chemin : Chemin vers le fichier .txt.

    Returns:
        Contenu textuel brut.

    Raises:
        FileNotFoundError : Fichier introuvable.
        ValueError        : Fichier vide.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Manuscrit introuvable : {path.resolve()}\n"
            "Vérifiez le chemin passé via --manuscrit."
        )
    texte = path.read_text(encoding="utf-8").strip()
    if not texte:
        raise ValueError(f"Le fichier {chemin} est vide.")
    return texte


def segmenter_manuscrit(texte: str, min_chars: int) -> list[dict]:
    """
    Découpe le manuscrit en paragraphes exploitables.

    Les blocs trop courts (titres, intertitres) sont ignorés — ils n'ont
    pas assez de substance argumentative pour une qualification de rapport
    pertinente.

    Args:
        texte     : Contenu brut du manuscrit.
        min_chars : Longueur minimale en caractères.

    Returns:
        Liste de dicts {index, texte, nb_chars}, indexés depuis 1.
    """
    separateur = r"\n" + r"\s*\n" * (SAUTS_LIGNE_SEPARATEUR - 1)
    blocs      = re.split(separateur, texte)

    paragraphes = []
    index = 1

    for bloc in blocs:
        bloc_nettoye = bloc.strip()
        if len(bloc_nettoye) < min_chars:
            continue
        paragraphes.append({
            "index":    index,
            "texte":    bloc_nettoye,
            "nb_chars": len(bloc_nettoye),
        })
        index += 1

    return paragraphes


# =============================================================================
# FONCTIONS — RECHERCHE FAISS
# =============================================================================

def encoder_texte(texte: str, client_openai: OpenAI) -> np.ndarray:
    """
    Encode un texte en vecteur via l'API OpenAI embeddings.

    L'embedding utilise toujours OpenAI, même si le backend LLM est Ollama.
    L'index FAISS a été construit avec des vecteurs OpenAI — la requête
    doit utiliser le même espace vectoriel pour que les distances aient un sens.

    Args:
        texte         : Texte à encoder.
        client_openai : Client OpenAI dédié aux embeddings.

    Returns:
        Vecteur float32 de forme (1, dim).
    """
    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texte,
    )
    vecteur = np.array(response.data[0].embedding, dtype="float32")
    return vecteur.reshape(1, -1)


def rechercher_segments(
    index: faiss.Index,
    metadata: list[dict],
    vecteur: np.ndarray,
    corpus_text: str,
    top_k: int,
) -> list[dict]:
    """
    Recherche les segments les plus proches dans FAISS.

    "Segment" plutôt que "passage" : rappel que ce sont des chunks du corpus,
    potentiellement tronqués. La qualification de rapport porte sur ces
    fragments, pas sur les articles entiers.

    Les distances FAISS sont conservées : un rapport CONTREDIT sur un segment
    très proche (distance faible) est plus significatif que sur un segment
    lointain. 08_visualise.py peut utiliser cette information.

    Args:
        index       : Index FAISS chargé.
        metadata    : Métadonnées des vecteurs.
        vecteur     : Vecteur du paragraphe, forme (1, dim).
        corpus_text : Contenu complet de corpus.txt.
        top_k       : Nombre de chunks à récupérer.

    Returns:
        Liste de dicts {source, page, passage, distance}, dédoublonnée.
    """
    distances, indices = index.search(vecteur, top_k)

    segments = []
    vus      = set()

    for rang, idx in enumerate(indices[0]):
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

        segments.append({
            "source":   meta["source"],
            "page":     meta["page"],
            "passage":  passage,
            "distance": float(distances[0][rang]),
        })

    return segments


# =============================================================================
# FONCTIONS — ANALYSE LLM
# =============================================================================

def extraire_scores(analyse: str) -> dict:
    """
    Extrait les sept scores numériques depuis la réponse textuelle du LLM.

    Pourquoi sept scores ?
        Six scores de rapport (un par type de la taxonomie) + un score de
        fragilité globale. Cette granularité permet à 08_visualise.py
        d'afficher non seulement "fragile" mais "fragile parce que
        CONTREDIT" vs "fragile parce que PROBLEMATISE" — deux situations
        qui peuvent demadner des révisions de nature très différente.

    Mécanique :
        Chaque score est extrait indépendamment par regex (PATTERNS_SCORES).
        Les valeurs hors bornes sont clampées à [0, 10] avant normalisation.
        Fallback à 0.5 (neutre) si un score est absent, avec warning terminal.

    Args:
        analyse : Texte brut de la réponse LLM.

    Returns:
        Dict avec les sept clés de NOMS_SCORES, valeurs float 0.0–1.0.
    """
    scores = {}
    for nom, pattern in PATTERNS_SCORES.items():
        match = re.search(pattern, analyse, re.IGNORECASE)
        if match:
            brut = float(match.group(1))
            scores[nom] = round(max(0.0, min(10.0, brut)) / 10.0, 2)
        else:
            print(f"    ⚠️  {nom} non extrait — valeur neutre 0.5 utilisée.")
            scores[nom] = 0.5
    return scores


def analyser_paragraphe(
    para: dict,
    segments: list[dict],
    client_openai: OpenAI,
    top_k_effectif: int,
) -> dict:
    """
    Soumet un paragraphe et ses segments au LLM, calcule les scores
    et retourne un dict structuré en trois niveaux pour le JSON de sortie.

    Les trois niveaux de sortie :

    NIVEAU 1 — Score global synthétique (pour la carte de chaleur)
        score_global    : float 0.0–1.0, calculé algorithmiquement par
                          pondération des six scores (POIDS_SCORE_GLOBAL).
                          Reproductible et déterministe — à la différence
                          du score_fragilite LLM qui est qualitatif.
        profil_dominant : type de rapport le plus présent ("contredit",
                          "problematise", "nuance", "particularise",
                          "deplace", "conforte", "equilibre").

    NIVEAU 2 — Scores individuels par rapport (pour le détail au survol)
        score_conforte, score_contredit, score_nuance, score_problematise,
        score_deplace, score_particularise : produits par le LLM (0.0–1.0).
        score_fragilite : synthèse qualitative du LLM (distinct du
                          score_global algorithmique — les deux coexistent).

    NIVEAU 3 — Segments qualifiés (pour les références citables)
        Liste des segments avec source, page et distance FAISS.
        Le texte complet reste dans corpus.txt (allègement du JSON).
        La distance FAISS est conservée : un CONTREDIT proche est plus
        significatif qu'un CONTREDIT lointain.

    Args:
        para           : Dict {index, texte, nb_chars} du paragraphe.
        segments       : Segments trouvés par FAISS.
        llm            : Instance de LLMClient (OpenAI ou Ollama).
        top_k_effectif : Valeur de TOP_K utilisée (traçabilité JSON).

    Returns:
        Dict structuré à trois niveaux, prêt pour le JSON de sortie.
    """
    # ── Cas sans segments : pas d'appel LLM, tous les scores à 0.0 ──────────
    if not segments:
        return {
            # NIVEAU 1 — Score global synthétique
            "score_global":    0.0,
            "profil_dominant": "conforte",   # pas de rapport négatif détecté

            # NIVEAU 2 — Scores individuels par type de rapport
            **{nom: 0.0 for nom in NOMS_SCORES},

            # NIVEAU 3 — Segments qualifiés (vide ici)
            "segments": [],

            # Métadonnées du paragraphe
            "index":    para["index"],
            "texte":    para["texte"],
            "nb_chars": para["nb_chars"],

            # Analyse brute (message d'absence)
            "analyse":    "Aucun segment trouvé dans le corpus pour ce paragraphe.",

            # Traçabilité
            "top_k":      top_k_effectif,
            "horodatage": datetime.now().isoformat(),
            "mode":       "critique",
        }

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    langue   = LANGUE_PROMPTS if LANGUE_PROMPTS in PROMPTS else "fr"
    prompt   = PROMPTS[langue]
    extraits = formater_extraits(segments)

    user_message = prompt["user"].format(
        index=para["index"],
        paragraphe=para["texte"],
        extraits=extraits,
    )

    analyse = client_openai.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS_CRITIQUE,
        temperature=TEMPERATURE_CRITIQUE,
    ).choices[0].message.content

    # ── Extraction et calcul des scores ───────────────────────────────────────
    scores          = extraire_scores(analyse)
    score_global    = calculer_score_global(scores)
    profil_dominant = identifier_profil_dominant(scores)

    # ── Parsing du JSON des passages ──────────────────────────────────────────
    passages_structures = []
    synthese = ""
    try:
        # Le LLM produit d'abord un bloc JSON puis les scores et la synthèse
        # On isole le JSON en cherchant le premier [ et le ] correspondant
        debut = analyse.find("[")
        fin   = analyse.rfind("]")
        if debut != -1 and fin != -1 and fin > debut:
            json_brut = analyse[debut:fin+1]
            items = json.loads(json_brut)
            for item in items:
                relation = item.get("relation", "hors_sujet")
                if relation not in ["conforte", "contredit", "nuance",
                                    "problematise", "deplace", "particularise",
                                    "hors_sujet"]:
                    relation = "hors_sujet"
                passages_structures.append({
                    "source":        item.get("source", "?"),
                    "page":          item.get("page", 0),
                    "relation":      relation,
                    "score":         round(max(0.0, min(1.0, float(item.get("score", 0.0)))), 3),
                    "justification": item.get("justification", ""),
                })
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  Parsing JSON passages échoué ({e}) — passages non structurés.")

    # Extraction de la synthèse
    match_synthese = re.search(r"SYNTHESE\s*:\s*(.+?)(?:\n\n|\Z)", analyse, re.DOTALL | re.IGNORECASE)
    if match_synthese:
        synthese = match_synthese.group(1).strip()

    # ── Construction du dict de sortie ────────────────────────────────────────
    return {
        # NIVEAU 1
        "score_global":    score_global,
        "profil_dominant": profil_dominant,

        # NIVEAU 2
        **scores,

        # NIVEAU 3 — segments pour le 08 (compatibilité)
        "segments": [
            {
                "source":   s["source"],
                "page":     s["page"],
                "distance": s["distance"],
            }
            for s in segments
        ],

        # NOUVEAU — passages structurés avec justifications citées
        "passages": passages_structures,

        # NOUVEAU — synthèse lisible
        "synthese": synthese,

        # Métadonnées du paragraphe
        "index":    para["index"],
        "texte":    para["texte"],
        "nb_chars": para["nb_chars"],

        # Analyse brute conservée pour le 08
        "analyse": analyse,

        # Traçabilité
        "top_k":      top_k_effectif,
        "horodatage": datetime.now().isoformat(),
        "mode":       "critique",
    }


# =============================================================================
# FONCTIONS — JSON DE SORTIE
# =============================================================================

def charger_resultats_existants(chemin: str) -> list[dict]:
    """
    Charge les résultats partiels pour reprise automatique.

    Args:
        chemin : Chemin vers le fichier JSON de sortie.

    Returns:
        Liste de résultats existants (vide si fichier absent ou illisible).
    """
    path = Path(chemin)
    if not path.exists():
        return []
    try:
        data      = json.loads(path.read_text(encoding="utf-8"))
        existants = data.get("paragraphes", [])
        if existants:
            print(f"  Reprise détectée : {len(existants)} paragraphe(s) déjà traité(s).")
        return existants
    except (json.JSONDecodeError, KeyError):
        print("  ⚠️  JSON existant illisible — reprise depuis le début.")
        return []


def sauvegarder_resultats(resultats: list[dict], chemin: str, meta_run: dict):
    """
    Sauvegarde les résultats dans le fichier JSON de sortie.

    Le champ meta_run["mode"] = "critique" permet à 08_visualise.py
    d'adapter la palette (rouge = fragile), les libellés et d'afficher
    les six sous-scores dans le panneau de détail au survol.

    Args:
        resultats : Liste des dicts résultats par paragraphe.
        chemin    : Chemin du fichier JSON de sortie.
        meta_run  : Métadonnées du traitement.
    """
    path = Path(chemin)
    path.parent.mkdir(parents=True, exist_ok=True)
    sortie = {"run": meta_run, "paragraphes": resultats}
    path.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


EMOJIS_RELATION = {
    "conforte":      "✅",
    "contredit":     "❌",
    "nuance":        "🟠",
    "problematise":  "🟣",
    "deplace":       "🔵",
    "particularise": "🟡",
    "hors_sujet":    "⬜",
}

LABELS_RELATION = {
    "conforte":      "conforte",
    "contredit":     "contredit",
    "nuance":        "nuance",
    "problematise":  "problématise",
    "deplace":       "déplace",
    "particularise": "particularise",
    "hors_sujet":    "hors sujet",
}


def generer_rapport_markdown(resultats: list[dict], chemin_json: str) -> Path:
    """
    Génère un rapport Markdown lisible à partir des résultats critiques.

    Format par paragraphe :
        ## § N — [début du texte]
        > [extrait du paragraphe]
        **Profil dominant :** X | **Score :** 0.XX

        ### Sources mobilisées
        **📄 source.pdf, p. N** — `relation` (0.XX)
        [justification avec citation intégrée]

        ### Synthèse
        [2-3 phrases]

    Args:
        resultats   : Liste des dicts résultats par paragraphe.
        chemin_json : Chemin du JSON de sortie (sert à dériver le chemin .md).

    Returns:
        Chemin du fichier Markdown produit.
    """
    chemin_md = Path(chemin_json).with_suffix(".md")

    lignes = [
        "# Rapport critique — Cartographie du manuscrit",
        f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*{len(resultats)} paragraphes analysés*",
        "",
        "---",
        "",
    ]

    for r in resultats:
        index  = r.get("index", "?")
        texte  = r.get("texte", "")
        profil = r.get("profil_dominant", "—")
        score  = r.get("score_global", 0.0)

        # En-tête du paragraphe
        debut = texte[:80].replace("\n", " ")
        lignes.append(f"## § {index} — {debut}…")
        lignes.append("")

        # Extrait du paragraphe (150 premiers caractères)
        apercu = texte[:300].replace("\n", " ")
        lignes.append(f"> {apercu}…")
        lignes.append("")

        # Scores résumés
        emoji = EMOJIS_RELATION.get(profil, "◆")
        label = LABELS_RELATION.get(profil, profil)
        lignes.append(f"**Profil dominant :** {emoji} {label} &nbsp;|&nbsp; **Score global :** {score:.2f}")
        lignes.append("")

        # Scores détaillés sur une ligne
        scores_ligne = " · ".join([
            f"conforte {r.get('score_conforte', 0):.2f}",
            f"contredit {r.get('score_contredit', 0):.2f}",
            f"nuance {r.get('score_nuance', 0):.2f}",
            f"problématise {r.get('score_problematise', 0):.2f}",
            f"déplace {r.get('score_deplace', 0):.2f}",
            f"particularise {r.get('score_particularise', 0):.2f}",
        ])
        lignes.append(f"*Scores : {scores_ligne}*")
        lignes.append("")

        # Passages structurés
        passages = r.get("passages", [])
        passages_utiles = [p for p in passages if p.get("relation") != "hors_sujet"]

        if passages_utiles:
            lignes.append("### Sources mobilisées")
            lignes.append("")
            for p in passages_utiles:
                rel   = p.get("relation", "?")
                sc    = p.get("score", 0.0)
                emoji_rel = EMOJIS_RELATION.get(rel, "◆")
                label_rel = LABELS_RELATION.get(rel, rel)
                lignes.append(
                    f"**📄 {p['source']}, p. {p['page']}** — "
                    f"{emoji_rel} `{label_rel}` ({sc:.2f})"
                )
                justif = p.get("justification", "").strip()
                if justif:
                    lignes.append("")
                    lignes.append(justif)
                lignes.append("")
        else:
            lignes.append("*Aucun passage directement qualifiable trouvé.*")
            lignes.append("")

        # Synthèse
        synthese = r.get("synthese", "").strip()
        if synthese:
            lignes.append("### Synthèse")
            lignes.append("")
            lignes.append(synthese)
            lignes.append("")

        lignes.append("---")
        lignes.append("")

    chemin_md.write_text("\n".join(lignes), encoding="utf-8")
    return chemin_md


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parsing des arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="07_map_critique.py — cartographie critique et relationnelle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 07_map_critique.py --manuscrit mon_manuscrit.txt
  python 07_map_critique.py --manuscrit mon_manuscrit.txt --sortie resultats/critique.json
  python 07_map_critique.py --manuscrit mon_manuscrit.txt --top_k 8 --pause 0.0
        """
    )
    parser.add_argument("--manuscrit", required=True, metavar="FICHIER",
                        help="Fichier .txt du manuscrit complet (obligatoire)")
    parser.add_argument("--sortie", default=SORTIE_JSON, metavar="FICHIER",
                        help=f"Fichier JSON de sortie (défaut : {SORTIE_JSON})")
    parser.add_argument("--top_k", type=int, default=None, metavar="N",
                        help="Nombre de segments FAISS par paragraphe")
    parser.add_argument("--pause", type=float, default=PAUSE_ENTRE_APPELS,
                        metavar="SEC",
                        help=f"Pause LLM en secondes (défaut : {PAUSE_ENTRE_APPELS})")
    parser.add_argument("--min_chars", type=int, default=PARAGRAPHE_MIN_CHARS,
                        metavar="N",
                        help=f"Longueur minimale d'un paragraphe (défaut : {PARAGRAPHE_MIN_CHARS})")
    args = parser.parse_args()

    top_k = args.top_k or TOP_K_LOCAL or TOP_K

    # -------------------------------------------------------------------------
    # En-tête
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  07 — CARTOGRAPHIE CRITIQUE ET RELATIONNELLE")
    print("=" * 60)
    print(f"  Manuscrit   : {args.manuscrit}")
    print(f"  Sortie      : {args.sortie}")
    print(f"  TOP_K       : {top_k}")
    print(f"  Pause LLM   : {args.pause}s")
    print(f"  Langue      : {LANGUE_PROMPTS}")
    print(f"  Modèle LLM  : {OPENAI_LLM_MODEL}")
    print(f"  Max tokens  : {MAX_TOKENS_CRITIQUE}")
    print(f"  Température : {TEMPERATURE_CRITIQUE}")
    print(f"  Rapports    : conforte | contredit | nuance |")
    print(f"                problématise | déplace | particularise")
    print()

    # -------------------------------------------------------------------------
    # Chargement
    # -------------------------------------------------------------------------
    try:
        print("Chargement du manuscrit…")
        texte_manuscrit = charger_manuscrit(args.manuscrit)

        print("Segmentation en paragraphes…")
        paragraphes = segmenter_manuscrit(texte_manuscrit, args.min_chars)
        print(f"  {len(paragraphes)} paragraphe(s) exploitable(s).")

        print("Chargement de l'index FAISS…")
        for path in (index_file, metadata_file):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier introuvable : {path.resolve()}\n"
                    "Lancez d'abord 03_build_embeddings.py."
                )
        index    = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        print(f"  {index.ntotal} vecteurs dans l'index.")

        print("Chargement du corpus…")
        corpus_text = charger_corpus()

        print("Initialisation du client OpenAI…")
        client_openai = OpenAI()

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Reprise
    # -------------------------------------------------------------------------
    resultats       = charger_resultats_existants(args.sortie)
    indices_traites = {r["index"] for r in resultats}
    a_traiter       = [p for p in paragraphes if p["index"] not in indices_traites]

    print(f"\n{len(a_traiter)} paragraphe(s) à analyser "
          f"({len(indices_traites)} déjà traité(s)).\n")

    if not a_traiter:
        print("✓ Tous les paragraphes ont déjà été traités.")
        print(f"  Lancez : python 08_visualise.py --source {args.sortie}")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Métadonnées du run
    # -------------------------------------------------------------------------
    meta_run = {
        "mode":                 "critique",
        "manuscrit":            args.manuscrit,
        "date_debut":           datetime.now().isoformat(),
        "top_k":                top_k,
        "langue":               LANGUE_PROMPTS,
        "nb_paragraphes":       len(paragraphes),
        "paragraphe_min_chars": args.min_chars,
        "taxonomie":            [
            "conforte", "contredit", "nuance",
            "problematise", "deplace", "particularise"
        ],
    }

    # -------------------------------------------------------------------------
    # Boucle principale
    # -------------------------------------------------------------------------
    print("─" * 60)
    print("  ANALYSE EN COURS")
    print("─" * 60)

    for i, para in enumerate(a_traiter):
        print(f"\n[{para['index']}/{len(paragraphes)}] "
              f"{para['nb_chars']} caractères")
        print(f"  Début : {para['texte'][:80].replace(chr(10), ' ')}…")

        try:
            vecteur = encoder_texte(para["texte"], client_openai)
        except Exception as e:
            print(f"  ⚠️  Erreur embedding : {e} — ignoré.")
            continue

        segments = rechercher_segments(
            index, metadata, vecteur, corpus_text, top_k
        )
        print(f"  {len(segments)} segment(s) récupéré(s).")

        try:
            resultat = analyser_paragraphe(para, segments, client_openai, top_k)
        except Exception as e:
            print(f"  ⚠️  Erreur LLM : {e} — ignoré.")
            continue

        # Affichage des scores en temps réel
        print(f"  Conforte      : {resultat['score_conforte']:.2f}  │  "
              f"Contredit    : {resultat['score_contredit']:.2f}")
        print(f"  Nuance        : {resultat['score_nuance']:.2f}  │  "
              f"Problématise : {resultat['score_problematise']:.2f}")
        print(f"  Déplace       : {resultat['score_deplace']:.2f}  │  "
              f"Particularise: {resultat['score_particularise']:.2f}")
        print(f"  ── Score global (pondéré)  : {resultat['score_global']:.3f}")
        print(f"  ── Fragilité LLM (qualit.) : {resultat['score_fragilite']:.2f}")
        print(f"  ── Profil dominant         : {resultat['profil_dominant']}")

        resultats.append(resultat)
        resultats.sort(key=lambda r: r["index"])

        if (i + 1) % SAUVEGARDER_TOUS_LES_N == 0:
            sauvegarder_resultats(resultats, args.sortie, meta_run)
            print(f"  💾 Sauvegarde intermédiaire ({len(resultats)} paragraphes)")

        if args.pause > 0 and i < len(a_traiter) - 1:
            time.sleep(args.pause)

    # -------------------------------------------------------------------------
    # Sauvegarde finale
    # -------------------------------------------------------------------------
    meta_run["date_fin"]    = datetime.now().isoformat()
    meta_run["nb_analyses"] = len(resultats)

    if resultats:
        for nom in NOMS_SCORES:
            meta_run[f"{nom}_moyen"] = round(
                sum(r.get(nom, 0.0) for r in resultats) / len(resultats), 3
            )

    sauvegarder_resultats(resultats, args.sortie, meta_run)

    # Rapport Markdown lisible en parallèle du JSON
    chemin_md = generer_rapport_markdown(resultats, args.sortie)
    print(f"\n  📝 Rapport Markdown : {chemin_md.resolve()}")

    # -------------------------------------------------------------------------
    # Bilan terminal avec barres visuelles
    # -------------------------------------------------------------------------
    print(f"\n{'═' * 60}")
    print(f"  BILAN — CARTOGRAPHIE CRITIQUE")
    print(f"{'═' * 60}")
    print(f"  Paragraphes analysés : {len(resultats)}")
    print()

    # Score global moyen (pondéré algorithmique)
    score_global_moyen = round(
        sum(r.get("score_global", 0.0) for r in resultats) / len(resultats), 3
    ) if resultats else 0.0
    print(f"  Score global moyen (pondéré) : {score_global_moyen:.3f}")
    print()

    # Scores moyens par type de rapport avec barres visuelles
    print(f"  Scores moyens par type de rapport :")
    libelles = {
        "score_conforte":      "Conforte     ",
        "score_contredit":     "Contredit    ",
        "score_nuance":        "Nuance       ",
        "score_problematise":  "Problématise ",
        "score_deplace":       "Déplace      ",
        "score_particularise": "Particularise",
        "score_fragilite":     "Fragilité LLM",
    }
    for nom, libelle in libelles.items():
        val   = meta_run.get(f"{nom}_moyen", 0.0)
        barre = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"    {libelle} : {barre} {val:.2f}")

    # Distribution des profils dominants
    print()
    print(f"  Distribution des profils dominants :")
    from collections import Counter
    profils = Counter(r.get("profil_dominant", "?") for r in resultats)
    for profil, count in profils.most_common():
        barre = "█" * count
        print(f"    {profil:<15} : {barre} ({count})")

    # Zones solides / fragiles selon le score global pondéré
    zones_fragiles = sum(
        1 for r in resultats if r.get("score_global", 0) >= SEUIL_FRAGILE
    )
    zones_solides = sum(
        1 for r in resultats if r.get("score_global", 0) < SEUIL_SOLIDE
    )
    print()
    print(f"  Zones solides (score_global < {SEUIL_SOLIDE})  : {zones_solides}")
    print(f"  Zones fragiles (score_global ≥ {SEUIL_FRAGILE}) : {zones_fragiles}")
    print(f"  Résultats : {Path(args.sortie).resolve()}")
    print(f"{'═' * 60}")
    print(f"\n  Étape suivante : python 08_visualise.py --source {args.sortie}")


if __name__ == "__main__":
    main()
