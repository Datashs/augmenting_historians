# Jeu de données de test — Pipeline d'analyse argumentative et rhétorique

## Contenu du dossier

Ce dossier fournit un **jeu de données prêt à l'emploi** pour valider l'installation et explorer les sorties du pipeline sans avoir à construire un corpus depuis zéro.

```
test_dataset/
├── texts/
│   └── manuscript_demo.txt        ← manuscrit de démonstration (5–10 paragraphes)
├── pdfs/
│   └── [fichiers PDF du corpus]   ← corpus bibliographique pré-sélectionné
├── embeddings_local/
│   ├── vector_store_local/        ← index FAISS (SentenceTransformers)
│   ├── extracted_text/
│   │   ├── corpus.txt             ← texte intégral extrait du corpus
│   │   ├── chunks.json            ← corpus découpé avec chevauchement
│   │   └── metadata.json         ← métadonnées des chunks (références, positions)
├── embeddings_voie_ai/
│   ├── vector_store/              ← index FAISS (embeddings OpenAI)
│   ├── extracted_text/
│   │   ├── corpus.txt
│   │   ├── chunks.json
│   │   └── metadata.json
```

---

## Ce qu'on peut faire avec ce jeu de données

Les embeddings pré-calculés permettent de **sauter les étapes 01, 02 et 03** et de passer directement aux scripts d'analyse. Cela est utile pour :

- **Valider l'installation** avant de lancer le pipeline sur son propre corpus
- **Explorer les sorties du pipeline** (signaux argumentatifs, profil rhétorique, analyse de segment)
- **Comprendre ce que produit chaque script** sans attendre la génération des embeddings

---

## Mode d'emploi — Voie locale (Ollama, zéro coût, zéro exposition)

**Prérequis :** Ollama installé et en cours d'exécution (`ollama serve`), au moins un modèle téléchargé (ex. `ollama pull qwen2.5:14b`).

**Étape 1 — Copier l'index pré-calculé dans le dossier projet**

```bash
cp -r test_dataset/embeddings_local/vector_store_local/  MonProjet/vector_store_local/
cp -r test_dataset/embeddings_local/extracted_text/      MonProjet/extracted_text/
cp    test_dataset/texts/manuscript_demo.txt             MonProjet/manuscript.txt
```

**Étape 2 — Lancer les scripts d'analyse directement**

```bash
cd MonProjet
conda activate rag_historien

# Cartographie manuscrit × corpus (niveau 1)
python 06_map_enrich_local.py --manuscript manuscript.txt

# Lecture critique (niveau 1)
python 07_map_critique_local.py --manuscript manuscript.txt

# Signaux argumentatifs en batch (niveau 2)
python 09_map_argumentation_local.py

# Signaux rhétoriques en batch (niveau 3)
python 10_map_perelman_local.py

# Zones prioritaires — zéro coût, aucun LLM
python 11_priority_zones.py

# Analyse de segment approfondie — coller un segment du manuscrit démo
python A_toulmin_segment_local.py
python B_perelman_segment_local.py
python C_rst_segment_local.py

# Synthèse croisée
python D_cross-synthesis_local.py
```

Toutes les sorties sont écrites dans le dossier `results/`.

---

## Mode d'emploi — Voie OpenAI

**Prérequis :** un fichier `.env` valide avec `OPENAI_API_KEY=sk-...` dans le dossier projet.

**Étape 1 — Copier l'index pré-calculé**

```bash
cp -r test_dataset/embeddings_voie_ai/vector_store/   MonProjet/vector_store/
cp -r test_dataset/embeddings_voie_ai/extracted_text/ MonProjet/extracted_text/
cp    test_dataset/texts/manuscript_demo.txt          MonProjet/manuscript.txt
```

**Étape 2 — Lancer les scripts d'analyse**

```bash
cd MonProjet
conda activate rag_historien

python 06_map_enrich.py --manuscript manuscript.txt
python 07_map_critique.py --manuscript manuscript.txt
python 09_map_argumentation_openai.py
python 10_map_perelman_openai.py
python 11_priority_zones.py
python A_toulmin_segment.py
python B_perelman_segment.py
python C_rst_segment.py
python D_cross-synthesis.py
```

---

## Premier lancement recommandé (5 minutes, local, zéro coût)

Pour vérifier simplement que tout fonctionne, lancer seulement ces trois scripts dans l'ordre :

```bash
python 07_map_critique_local.py --manuscript manuscript.txt
python 09_map_argumentation_local.py
python 11_priority_zones.py
```

Le script 11 ne fait aucun appel LLM et affiche immédiatement les zones que les signaux en batch ont identifiées comme méritant examen.

---

## Le manuscrit de démonstration

`manuscript_demo.txt` est un texte historique court (5–10 paragraphes) conçu pour solliciter les trois niveaux d'analyse : il contient des passages à structure argumentative affirmée, des passages à forte densité rhétorique mais à ancrage probatoire mince, et des passages de mise en dialogue explicite avec l'historiographie. Cette variation est intentionnelle — elle rend immédiatement visibles les différences de signaux entre les scripts 09, 10 et 11.

---

## Rappel important sur les scores

Les scripts 09 et 10 produisent des **signaux de localisation, non des métriques fiables au niveau du paragraphe**. Un score bas sur un paragraphe isolé n'est pas un jugement — c'est une invitation à relire cette zone. Toujours lancer le script 11 avant d'interpréter une sortie de 09 ou 10, et utiliser les scripts A, B, C pour l'analyse rigoureuse des zones qu'il signale.

Voir le `README.md` principal pour la discussion épistémologique complète.

---

## Pour utiliser son propre corpus

Une fois l'installation validée avec le jeu de test, reprendre le pipeline depuis l'étape 01 :

```bash
# Placer ses PDF dans pdfs/
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py   # ou 03_build_embeddings.py pour la voie OpenAI
# Puis enchaîner depuis l'étape 06
```

Voir le `README.md` principal et le `pipeline_guide.pdf` pour le workflow complet, notamment l'intégration Zotero (voie A) et les considérations sur la souveraineté des données.
