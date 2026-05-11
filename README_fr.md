# Pipeline d'analyse argumentative et rhétorique pour l'historien

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20122308.svg)](https://doi.org/10.5281/zenodo.20122308)

## Ce que fait ce dispositif

Ce pipeline est un ensemble de scripts Python qui assiste l'historien dans
l'analyse critique de son propre manuscrit. Il ne génère pas de texte à la
place de l'auteur — il outille la **réflexivité** : il rend visible ce que
la lecture ordinaire laisse implicite, confronte le texte en cour d'écriture
au corpus bibliographique défini par l'utilisateur , il désigne les zones qui méritent
attention, et propose des cadres analytiques pour les explorer.

Le dispositif opère à trois niveaux distincts :

**Niveau 1 — Rapport au corpus historiographique.** Le manuscrit est
confronté aux sources secondaires que l'auteur a rassemblées. Chaque
paragraphe est mis en relation avec les passages corpus les plus proches
sémantiquement, et ces relations sont qualifiées selon une taxonomie de
six types : conforte, contredit, nuance, problématise, déplace, particularise.

**Niveau 2 — Structure argumentative.** La structure logique du texte est
analysée selon les cadres de Toulmin (1958), Adam (1992) et Walton (1996).
En mode batch (script 09), ce niveau produit des signaux de localisation, pas
des métriques fiables paragraphe par paragraphe. En mode segment défini
(script A), il produit une analyse rigoureuse sur une unité choisie par
l'auteur.

**Niveau 3 — Stratégies rhétoriques.** Les techniques argumentatives sont
analysées selon la Nouvelle Rhétorique de Perelman & Olbrechts-Tyteca (1958)
et l'analyse RST (Man et Thompson 1988).
En mode batch (script 10), même logique de signal. En mode segment
 (script B, script C), analyse approfondie avec identification du mouvement 
 rhétorique interne.

---

## Avertissement épistémique fondamental

Les scripts 09 et 10 opèrent paragraphe par paragraphe sur l'ensemble du
manuscrit. **Leurs scores individuels ne sont pas des métriques argumentatives
fiables**, pour deux raisons structurelles :

1. **Taille de contexte** : un paragraphe isolé est trop court pour que
   Toulmin soit applicable rigoureusement. L'argument d'un historien se
   construit sur plusieurs paragraphes, voire un chapitre entier.

2. **Polyphonie** : le texte historique rapporte, cite et reconstruit des
   argumentaires qui ne sont pas ceux de l'auteur. Le LLM peut confondre
   le discours de l'auteur et le discours rapporté.
   
3. **Imprévisibilité LLm** : la nature stochastique des llm fait que les sorties 
vont varier selon le Llm utilisé, la configuration d'utilisation.

Ce que les scripts 09 et 10 produisent de valide :
- Un **portrait rhétorique agrégé** du texte entier (distribution des schèmes,
  techniques dominantes, cohérence axiologique globale) dont la pertinence 
  doit être appréciée par l'utilisateur. .
- Des **signaux de localisation** : zones où quelque chose mérite relecture.

C'est le script 11 qui exploite ces signaux pour désigner des zones
prioritaires. Les scripts A, B, C produisent l'analyse rigoureuse sur ces
zones, à la granularité choisie par l'historien. **Ne jamais interpréter un
score 09 ou 10 individuel comme un jugement sur un paragraphe.**

---

## Souveraineté des données — trois niveaux

Le pipeline distingue trois étapes aux propriétés de souveraineté très
différentes, qui correspondent à des choix architecturaux délibérés.

**Étape 1 — Construction du corpus vectoriel (locale).** Les PDF du corpus
sont découpés en chunks, encodés en vecteurs par un modèle d'embeddings, et
indexés dans FAISS. Si l'on utilise un modèle SentenceTransformers local
(version `_local`), aucune donnée ne transite vers l'extérieur.

**Étape 2 — Confrontation manuscrit × corpus.** Pour chaque paragraphe, FAISS
sélectionne les passages les plus proches. Le LLM reçoit le paragraphe et ces
passages, et qualifie leurs relations. Si l'on utilise un LLM externe (OpenAI),
la totalité du manuscrit transite vers des serveurs tiers — pas d'un bloc, mais
paragraphe par paragraphe. Les scripts `_local` (Ollama) évitent cette
exposition.

**Étape 3 — Analyse approfondie sur segment.** L'historien délimite lui-même
les segments soumis aux scripts A, B, C. Seuls ces segments transitent vers le
LLM externe — exposition consentie et contrôlée.

**Architecture recommandée pour les manuscrits inédits :**
- Étape 1 : SentenceTransformers local (zéro exposition)
- Étape 2 batch (09/10) : Ollama local (zéro exposition)
- Étape 3 (A/B/C/D) : LLM externe possible au choix, 
sur les segments consciemment choisis

---

## Architecture du pipeline

```
VOIE A — Avec Zotero (références bibliographiques formées)
───────────────────────────────────────────────────────────
00a_html_to_pdf.py           Conversion snapshots HTML → PDF (WeasyPrint)
00_zotero_import.py          Import initial : pdfs/ + metadata_refs.json
00b_zotero_update.py         Mise à jour incrémentale (sans rebuild complet)

VOIE B — Sans Zotero (dossier pdfs/ manuel)
────────────────────────────────────────────
  Déposer les PDFs dans pdfs/ et démarrer directement à l'étape 01.
  Les références affichées seront les noms de fichiers (ex : Noiriel1988.pdf).

PRÉPARATION DU CORPUS (commun aux deux voies)
─────────────────────────────────────────────
01_extract_text.py           Extraction PDF → texte brut
02_chunk_corpus.py           Découpage en chunks avec chevauchement
03_build_embeddings.py       Index FAISS + metadata.json (embeddings OpenAI)
03_build_embeddings_local.py Idem, embeddings SentenceTransformers locaux

FONCTIONS RAG
─────────────────────────────────────────────

04_rag_query.py / 04_rag_query.py Réponse synthétique ancée dans le corpus 
05_rag_write.py / 04_rag_query.py Enrichissement d'un passage par le corpus

RAPPORT MANUSCRIT × CORPUS — niveau 1
──────────────────────────────────────
06_map_enrich.py / 06_map_enrich_local.py
  └→ map_{ts}.json + map_{ts}.md

07_map_critique.py / 07_map_critique_local.py
  └→ critique.json + critique.md

ANALYSE ARGUMENTATIVE BATCH — niveau 2 (signaux)
─────────────────────────────────────────────────
09_map_argumentation_openai.py / 09_map_argumentation_local.py
  └→ argumentation_{ts}.json + argumentation_{ts}.md

ANALYSE RHÉTORIQUE BATCH — niveau 3 (signaux)
──────────────────────────────────────────────
10_map_perelman_openai.py / 10_map_perelman_local.py
  └→ perelman_{ts}.json + perelman_{ts}.md

NAVIGATION — ZONES PRIORITAIRES (coût zéro)
────────────────────────────────────────────
11_zones_prioritaires.py     Croisement 09 × 10, aucun LLM
  └→ zones_{ts}.md  ← lire avant de lancer A/B/C

ANALYSE APPROFONDIE SUR SEGMENT
────────────────────────────────
A_toulmin_segment.py / A_toulmin_segment_local.py
  └→ toulmin_{ts}.md + toulmin_{ts}.json

B_perelman_segment.py / B_perelman_segment_local.py
  └→ perelman_seg_{ts}.md + perelman_seg_{ts}.json

C_rst_segment.py / C_rst_segment_local.py
  └→ rst_{ts}.md + rst_{ts}.json

D_synthese_croisee.py / D_synthese_croisee_local.py
  └→ synthese_croisee_{ts}.md
  
E_recommandations_local.py / E_recommandations_openai.py
      └→ conseils d'écriture


VISUALISATION (voie locale Ollama)
─────────────
09_viz_argumentation.py      HTML depuis argumentation_{ts}.json
10_viz_perelman.py           HTML  depuis perelman_{ts}.json
```

---

## Prérequis

### Environnement Python

```bash
conda create -n rag_historien python=3.11
conda activate rag_historien
pip install -r requirements.txt

```

On mac, if WeasyPrint échoue : `brew install pango cairo`

### Fichier `.env` (voie OpenAI uniquement)

```
OPENAI_API_KEY=sk-...
```

Non requis pour la voie locale. Le fichier doit être à la racine du projet.
Si usage d'un autre LLM ajouter la clé d'API''

### Ollama (voie locale uniquement)

```bash
# Installer depuis https://ollama.com
ollama serve                      # lancer en arrière-plan avant tout script _local
ollama pull qwen2.5:14b           # modèle par défaut
# Alternatives recommandées pour 09/10/A/B/C :
ollama pull deepseek-r1:14b       # meilleur sur raisonnement structuré
ollama pull gemma3:12b            # très bon en français académique
```

Pour changer de modèle sur tout le pipeline local, modifier `LLM_MODEL`
dans `rag_config_local.py` — une seule ligne, tout le pipeline suit.

### Structure des dossiers

```
MonProjet/
├── .env                         ← OPENAI_API_KEY (voie OpenAI uniquement)
├── rag_config_local.py          ← configuration voie locale
├── config_00.py                 ← configuration voie OpenAI
├── [tous les scripts .py]
├── pdfs/                        ← corpus PDF (voie B ou copié par 00_zotero_import)
├── extracted_text/
│   ├── corpus.txt               ← produit par 01
│   └── chunks.json              ← produit par 02
├── vector_store/                ← produit par 03 (voie OpenAI)
│   ├── faiss.index
│   ├── metadata.json            ← enrichi avec ref_courte/ref_longue si Zotero
│   └── embeddings.npy
├── vector_store_local/          ← produit par 03_local (voie Ollama)
│   └── [mêmes fichiers]
├── metadata_refs.json           ← produit par 00_zotero_import (voie A uniquement)
└── resultats/                   ← toutes les sorties 06-11, A-D voie locale
├ outputs/                       ← toutes les sorties 06-11, A-D voie api
```

---

## Workflows

### Voie A — avec Zotero (première fois)

```bash
# 1. Exporter depuis Zotero : Fichier → Exporter la bibliothèque → CSV (toutes colonnes)

# 2. Convertir les snapshots HTML en PDF
python 00a_html_to_pdf.py --csv IAHistoire.csv

# 3. Import initial
python 00_zotero_import.py --csv IAHistoire.csv

# 4. Corpus
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py    # local recommandé
# ou
python 03_build_embeddings.py          # OpenAI
```

### Voie B — sans Zotero

```bash
# Déposer les PDFs dans pdfs/ puis :
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py    # ou 03_build_embeddings.py
```

### Mise à jour Zotero (voie A — incrémentale)

```bash
# Quand de nouvelles références ont été ajoutées dans Zotero :
python 00a_html_to_pdf.py              # si nouveaux HTML
python 00b_zotero_update.py --csv IAHistoire.csv
# Relancer les analyses directement après — pas besoin de reconstruire
```

### Analyse complète (commune aux deux voies)

```bash
# Niveau 1
python 06_map_enrich.py --manuscrit mon_manuscrit.txt       # ou _local
python 07_map_critique.py --manuscrit mon_manuscrit.txt     # ou _local

# Niveau 2
python 09_map_argumentation_openai.py     # ou _local

# Niveau 3
python 10_map_perelman_openai.py          # ou _local

# Navigation — toujours lancer avant A/B/C
python 11_zones_prioritaires.py

# Visualisation optionnelle
python 09_viz_argumentation.py
python 10_viz_perelman.py

# Analyse approfondie sur segment (après lecture de zones_{ts}.md)
python A_toulmin_segment.py       # ou _local
python B_perelman_segment.py      # ou _local
python C_rst_segment.py           # ou _local
python D_synthese_croisee.py      # ou _local
python E_recommandations          # ou _local
```

---

## Description des scripts

Chaque script est documenté plus précisément dans le code.

### 00a — `00a_html_to_pdf.py`
Convertit les snapshots HTML sauvegardés par Zotero en PDF via WeasyPrint.
Produit un rapport MD listant succès et échecs. `--dry-run` pour prévisualiser.
`--force` pour reconvertir même si un PDF existe déjà.

### 00 — `00_zotero_import.py`
Lit le CSV Zotero, construit `metadata_refs.json` et copie les PDFs dans
`pdfs/`. Construit automatiquement les références courtes (Noiriel (1988))
et longues (Noiriel G., *Le Creuset français*, Seuil, 1988) selon le type
d'item (article, livre, chapitre). `--dry-run` disponible.

### 00b — `00b_zotero_update.py`
Détecte les nouveaux items Zotero par leur clé `Key`, extrait leur texte,
les chunke, et ajoute leurs vecteurs à l'index FAISS existant sans rebuild.
Vérifie la cohérence dimensionnelle (refuse si le modèle d'embeddings a changé).
Les suppressions et modifications sont ignorées — rebuild complet si nécessaire.

### 01 — `01_extract_text.py`
Extraction texte brut depuis les PDFs du corpus. Supporte natifs et scannés (OCR).
Sortie : `extracted_text/corpus.txt`

### 02 — `02_chunk_corpus.py`
Découpage de `corpus.txt` en chunks de taille contrôlée avec chevauchement.
Sortie : `extracted_text/chunks.json`

### 03 — `03_build_embeddings.py` / `03_build_embeddings_local.py`
Encode les chunks en vecteurs et construit l'index FAISS. Si
`metadata_refs.json` est présent (voie A), injecte `ref_courte` et `ref_longue`
dans chaque chunk — ces références se propagent dans toutes les sorties.
Si absent (voie B), comportement identique à avant (nom de fichier brut).
Reprend automatiquement en cas d'interruption.
Sorties : `vector_store/` (OpenAI) ou `vector_store_local/` (local).

### 04 - 04_rag_query.py  / 04_rag_query.py
Interroge le contenu documentaire stocké, sélectionne les segments pertinents
retourne une synthèse rédigée en langue naturelle.


### 05_rag_write.py  / 05_rag_write_local.py

Recherche les passages relatifs à une questions posée 
dans le corpus, puis rédige un passage académique
structuré, ancré dans les sources, avec citations explicites.

### 06 — `06_map_enrich.py` / `06_map_enrich_local.py`
Mode enrichissement. Posture constructive : "Qu'est-ce que le corpus peut
apporter à ce passage ?" Reprend automatiquement en cas d'interruption.
Sorties : `map_{ts}.json` + `map_{ts}.md`

### 07 — `07_map_critique.py` / `07_map_critique_local.py`
Mode critique et relationnel. Qualifie les rapports entre chaque paragraphe
et les passages corpus selon la taxonomie des six relations. Produit un score
de fragilité par paragraphe. Reprend automatiquement.

**Taxonomie des six relations :**

| Relation | Description |
|---|---|
| `conforte` | Le corpus confirme, étaye ou illustre |
| `contredit` | Le corpus s'oppose directement |
| `nuance` | Le corpus complexifie sans invalider |
| `problématise` | Une affirmation présentée comme acquise est débattue |
| `déplace` | Le corpus propose un autre cadre conceptuel |
| `particularise` | Le général affirmé a des expressions locales très différentes |

Sorties : `critique.json` + `critique.md`

### 09 — `09_map_argumentation_openai.py` / `09_map_argumentation_local.py`
Analyse argumentative batch (Toulmin / Adam / Walton) sur tout le manuscrit.
Voir l'avertissement épistémique — scores individuels = signaux uniquement.
`--reset` pour repartir de zéro.

**Scores (signaux, pas métriques) :**
- `score_completude_toulmin` : proportion des six composantes présentes
- `score_coherence_warrant` : solidité du lien grounds→claim
- `score_charge_probatoire` : présence de grounds explicites
- `score_risque_sophisme` : présence de schèmes fallacieux
- `score_robustesse_globale` : synthèse pondérée

Sorties : `argumentation_{ts}.json` + `argumentation_{ts}.md`

### 10 — `10_map_perelman_openai.py` / `10_map_perelman_local.py`
Analyse rhétorique batch (Perelman). Peut être chaîné avec le 09 via
`--avec_argumentation`.

**Scores (signaux, pas métriques) :**
- `score_force_persuasive` : sophistication rhétorique du passage
- `score_ancrage_auditoire` : adéquation aux conventions disciplinaires
- `score_coherence_valeurs` : cohérence interne du système de valeurs
- `score_risque_sophistique` : usages rhétoriques problématiques
- `score_profil_argumentatif` : synthèse pondérée

Sorties : `perelman_{ts}.json` + `perelman_{ts}.md`

### 11 — `11_zones_prioritaires.py`
Croise les JSON 09 et 10, calcule six signaux, désigne les zones prioritaires.
**Coût : zéro.** Toujours lancer avant A/B/C. Le MD produit s'ouvre sur une
section "Comment utiliser ce rapport" avec : pourquoi ne pas copier un
paragraphe seul, les deux façons de fournir un segment (console ou `--fichier`),
et un tableau indiquant quel script choisir selon les signaux actifs.

**Six signaux croisés :**

| Signal | Nom | Script recommandé |
|---|---|---|
| A | `rhéto_sans_preuve` | Script A prioritaire |
| B | `force_sans_grounds` | Scripts A + B |
| C | `double_sophisme` | Script A urgent |
| D | `09_muet_10_parle` | Script C (cohérence discursive) |
| E | `autorité_disciplinaire` | Script B |
| F | `cohérence_axiologique` | Signal positif — zone solide |

Sortie : `zones_{ts}.md`

### A — `A_toulmin_segment.py` / `A_toulmin_segment_local.py`
Analyse Toulmin/Adam/Walton sur segment défini par l'historien. Interface
interactive : titre obligatoire, texte collé dans la console ou
`--fichier mon_segment.txt`, question facultative. Identifie le discours
rapporté avant d'appliquer Toulmin — gestion partielle de la polyphonie.
Estimation du coût (OpenAI) ou du temps (local) avant appel.
Sorties : `toulmin_{ts}.md` + `toulmin_{ts}.json`

### B — `B_perelman_segment.py` / `B_perelman_segment_local.py`
Analyse Nouvelle Rhétorique sur segment défini. Même interface que A. Ajoute
une section mouvement rhétorique interne — invisible en mode batch.
Sorties : `perelman_seg_{ts}.md` + `perelman_seg_{ts}.json`

### C — `C_rst_segment.py` / `C_rst_segment_local.py`
Analyse RST (Rhetorical Structure Theory) sur segment défini. Granularité :
la phrase. Génère un arbre RST en syntaxe Mermaid lisible dans VS Code,
Obsidian, GitHub, Typora. Signale les relations à faible confiance.
Statut : approximation heuristique, non annotation RST rigoureuse.
Sorties : `rst_{ts}.md` + `rst_{ts}.json`

### D — `D_synthese_croisee.py` / `D_synthese_croisee_local.py`
Synthèse intégrée A + B + C. Détecte automatiquement les JSON les plus
récents dans `resultats/`. Vérifie que les trois analyses portent sur le même
segment. Valorise explicitement les divergences entre cadres.
Sortie : `synthese_croisee_{ts}.md`

### E_recommandations_local.py` / E_recommandations_openai.py

Lit le rapport Markdown produit par D_synthese_croisee.py
(synthèse croisée Toulmin + Perelman + RST) et demande au LLM de
transformer cette synthèse en recommandations concrètes, hiérarchisées
et directement actionnables pour l'historien.

### Visualisateurs — `09_viz_argumentation.py` / `10_viz_perelman.py`
HTML auto-contenu depuis le JSON 09 ou 10. Aucune dépendance réseau.
S'ouvre automatiquement dans le navigateur (`--no-open` pour désactiver).

---

## Coûts indicatifs (gpt-4.1-mini, 2026)

| Script | Coût OpenAI | Coût local |
|---|---|---|
| 06 / 07 | ~0.05–0.20 $ | 0.00 $ |
| 09 / 10 | ~0.10–0.50 $ | 0.00 $ |
| 11 | **0.00 $** | **0.00 $** |
| A / B / C | ~0.002–0.008 $ | 0.00 $ |
| D | ~0.004–0.007 $ | 0.00 $ |

Les scripts A, B, C, D affichent une estimation avant chaque appel et
demandent confirmation (`--no-confirm` pour désactiver).

---

## Choix du modèle LLM local

Modifier `LLM_MODEL` dans `rag_config_local.py` — une seule ligne suffit.

| Modèle | RAM | Usage recommandé |
|---|---|---|
| `deepseek-r1:14b` | 16 Go | Scripts 09, 10, A, B, C — raisonnement structuré (Toulmin/Perelman/RST) |
| `gemma3:12b` | 16 Go | Scripts 09, 10, A, B, C — excellent en français académique |
| `qwen2.5:14b` | 16 Go | Défaut — bon équilibre général sur tous les scripts |
| `mistral:7b-instruct` | 8 Go | Scripts 04, 05, 06, 07 — si RAM < 16 Go |
| `qwen2.5:7b` | 8 Go | Si RAM < 16 Go — qualité correcte pour 06/07 |

Note : DeepSeek-R1 est open weights et tourne entièrement en local via Ollama.
La question de la juridiction ne se pose que pour les API cloud, pas en local.
je suis en train de me dire qu'il me faut un avertissement,
les scripts en voie locale n'ont pas été testés avec tous les llms qu'il est possible d'actionner avec Ollama, 
il est possible que dans certains cas (en particulier dans le cas de modèles de petite taille) 
le parsing connaisse quelques difficultés avec les scripts 6 à 10. 
Cela n'affecte cependant généralement que les scores calculés, disponibles dans les sorties raw,
non les synthèses ou les analyses.  

---

## Propagation des références bibliographiques

Quand `metadata_refs.json` est présent (voie A), le script 03 injecte
`ref_courte` et `ref_longue` dans chaque chunk de `metadata.json`. Ces
références se propagent dans :

- Les rapports MD de 06 et 07 : `**📄 Noiriel (1988), p. 3**`
- Les prompts envoyés au LLM dans 07 : le LLM voit "Noiriel (1988)" et cite mieux
- Les rapports MD de 09 et 10 : `[Passage 1 — Noiriel (1988), p. 3]`

Sans `metadata_refs.json` (voie B), le nom de fichier brut est utilisé
partout en fallback. Aucune régression.

---

## Cadres théoriques mobilisés

**Toulmin, S. (1958).** *The Uses of Argument.* Cambridge University Press.
Six composantes : claim, grounds, warrant, backing, qualifier, rebuttal.

**Adam, J.-M. (1992).** *Les textes : types et prototypes.* Nathan.
Séquence argumentative : thèse antérieure, données, conclusion, restriction.

**Walton, D. (1996).** *Argumentation Schemes for Presumptive Reasoning.*
Lawrence Erlbaum. Schèmes légitimes et fallacieux.

**Perelman, C. & Olbrechts-Tyteca, L. (1958).** *Traité de l'argumentation —
La Nouvelle Rhétorique.* Éditions de l'Université de Bruxelles.
Techniques d'association (A1/A2/A3) et de dissociation (B).

**Mann, W. C. & Thompson, S. A. (1988).** "Rhetorical Structure Theory:
Toward a functional theory of text organization." *Text*, 8(3), 243–281.

---

## Limites assumées du dispositif

**Scores 09/10 non métriques.** Les cadres théoriques (Toulmin surtout) ont
été conçus pour des arguments en première personne, pas pour des textes qui
rapportent des argumentaires ou construisent leur thèse par accumulation.

**Polyphonie du texte historique.** L'historien cite et reconstruit des
positions adverses. Le script A gère partiellement ce problème, mais la limite
reste structurelle.

**Taille de contexte.** Un paragraphe isolé est trop court pour Toulmin.
Les scripts A, B, C répondent à cette limite en travaillant sur des segments
définis par l'historien.

**RST comme approximation.** Le script C produit une analyse heuristique, non
une annotation rigoureuse. Les relations à faible confiance sont signalées.

**Alignement 09/10 dans le 11.** Si les JSON ont été produits sur des corpus
différents, l'alignement par position peut être incorrect. Le script 11
affiche un avertissement si les longueurs diffèrent.

**Variabilité inter-modèles.** La qualité des analyses dépend du modèle LLM
et de sa représentation des cadres théoriques. Documenter précisément le
modèle utilisé (nom, version, température, date) pour la reproductibilité.

---

## Problèmes fréquents

**Scores tous à 0.50** — LLM n'a pas respecté le format `X/10`. La fonction
`extraire_scores()` robuste (virgule décimale, nom court, tableau Markdown) est
intégrée dans 09, 10, A, B. En voie API un utilitaire permet de recalculer 
les scores.

**Sections "Non disponible" dans D** — LLM a utilisé des séparateurs
différents de `━━━`. La fonction `extraire_section_llm()` tolérante est
intégrée dans D.

**Réponses tronquées dans A/B/C** — monter `MAX_TOKENS` à 4500 (A/B) ou
5000 (C) en tête du script.

**Ollama inaccessible** — lancer `ollama serve` avant tout script `_local`.

**`ref_courte` absent dans les sorties** — relancer `03_build_embeddings.py`
après `00_zotero_import.py`.

**PDF non trouvé pour un item Zotero** — vérifier `ZOTERO_STORAGE` dans
`00_zotero_import.py`.

**WeasyPrint échoue sur macOS** — `brew install pango cairo`.

---

*Pipeline développé dans le cadre d'un projet de recherche en humanités
numériques. Pour tout usage académique, mentionner les cadres théoriques
mobilisés et les limites du dispositif et les références du prototype*
