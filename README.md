# Argumentative and Rhetorical Analysis Pipeline for Historians

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20122308.svg)](https://doi.org/10.5281/zenodo.20122308)

## What this system does


This pipeline is a set of Python scripts that assists historians in the
critical analysis of their own manuscripts. It does not generate text in
place of the author — it supports **reflexivity**: it makes visible what
ordinary reading leaves implicit, confronts the text in progress with the
user-defined bibliographic corpus, identifies areas that deserve attention,
and proposes analytical frameworks to explore them.

The system operates at three distinct levels:

**Level 1 — Relation to the historiographical corpus.** The manuscript is
confronted with the secondary sources assembled by the author. Each paragraph
is matched with the most semantically similar corpus passages, and these
relations are qualified according to a taxonomy of six types: supports,
contradicts, nuances, problematizes, shifts, particularizes.

**Level 2 — Argumentative structure.** The logical structure of the text is
analyzed using the frameworks of Toulmin (1958), Adam (1992), and Walton (1996).
In batch mode (script 09), this level produces localization signals, not
reliable paragraph-level metrics. In defined-segment mode (script A), it
produces a rigorous analysis on a unit chosen by the author.

**Level 3 — Rhetorical strategies.** Argumentative techniques are analyzed
according to Perelman & Olbrechts-Tyteca’s New Rhetoric (1958) and RST
analysis (Mann and Thompson 1988).
In batch mode (script 10), the same signal logic applies. In segment mode
(script B, script C), a deeper analysis is produced, identifying the internal
rhetorical movement.

---

## Fundamental epistemological warning

Scripts 09 and 10 operate paragraph by paragraph across the entire manuscript.
**Their individual scores are not reliable argumentative metrics**, for two
structural reasons:

1. **Context size**: an isolated paragraph is too short for Toulmin to be
   applied rigorously. A historian’s argument is constructed across multiple
   paragraphs, or even an entire chapter.

2. **Polyphony**: historical writing reports, quotes, and reconstructs
   arguments that are not those of the author. The LLM may confuse the
   author’s voice with reported discourse.

3. **LLM unpredictability**: the stochastic nature of LLMs means outputs
   vary depending on the model and configuration.

What scripts 09 and 10 produce that is valid:
- An **aggregated rhetorical profile** of the entire text (distribution of
  schemes, dominant techniques, overall axiological coherence), whose relevance
  must be assessed by the user.
- **Localization signals**: areas where something deserves re-reading.

Script 11 uses these signals to designate priority zones. Scripts A, B, C
produce rigorous analysis on these zones, at the granularity chosen by the
historian. **Never interpret an individual 09 or 10 score as a judgment on a
paragraph.**

---

## Data sovereignty — three levels

The pipeline distinguishes three stages with very different sovereignty
properties, reflecting deliberate architectural choices.

**Stage 1 — Vector corpus construction (local).** PDFs are split into chunks,
encoded into vectors via an embedding model, and indexed in FAISS. If a local
SentenceTransformers model is used (`_local` version), no data leaves the machine.

**Stage 2 — Manuscript × corpus comparison.** For each paragraph, FAISS selects
the closest passages. The LLM receives the paragraph and these passages, and
qualifies their relations. If an external LLM is used (OpenAI), the entire
manuscript is sent to third-party servers — not as a whole, but paragraph by
paragraph. `_local` scripts (Ollama) avoid this exposure.

**Stage 3 — Deep segment analysis.** The historian defines the segments
submitted to scripts A, B, C. Only these segments are sent to the external LLM
— controlled and deliberate exposure.

**Recommended architecture for unpublished manuscripts:**
- Stage 1: local SentenceTransformers (zero exposure)
- Stage 2 batch (09/10): local Ollama (zero exposure)
- Stage 3 (A/B/C/D): external LLM optional, on selected segments

---

## Pipeline architecture

VOIE A — with Zotero 
───────────────────────────────────────────────────────────
00a_html_to_pdf.py           Conversion snapshots HTML → PDF (WeasyPrint)
00_zotero_import.py          Import initial : pdfs/ + metadata_refs.json
00b_zotero_update.py         Incremental update (without full rebuild)

VOIE B — Sans Zotero (dossier pdfs/ manuel)
────────────────────────────────────────────
Place the PDFs in the `pdfs/` directory and start directly from step 01.  
Displayed references will correspond to file names (e.g. `Noiriel1988.pdf`).

PRÉPARATION DU CORPUS (commun aux deux voies)
─────────────────────────────────────────────
01_extract_text.py           PDF extraction → raw text
02_chunk_corpus.py           Chunking with overlap
03_build_embeddings.py       FAISS index + metadata.json (OpenAI embeddings)
03_build_embeddings_local.py Idem, embeddings SentenceTransformers locally

FONCTIONS RAG
─────────────────────────────────────────────

04_rag_query.py / 04_rag_query.py Synthesized response grounded in the corpus 
05_rag_write.py / 04_rag_query.py Passage enrichment using the corpus

RAPPORT MANUSCRIT × CORPUS — niveau 1
──────────────────────────────────────
06_map_enrich.py / 06_map_enrich_local.py
  └→ map_{ts}.json + map_{ts}.md

07_map_critique.py / 07_map_critique_local.py
  └→ critique.json + critique.md

Batch argumentative analysis — Level 2 (signals)
─────────────────────────────────────────────────
09_map_argumentation_openai.py / 09_map_argumentation_local.py
  └→ argumentation_{ts}.json + argumentation_{ts}.md

Batch rhetorical analysis — Level 3 (signals)
──────────────────────────────────────────────
10_map_perelman_openai.py / 10_map_perelman_local.py
  └→ perelman_{ts}.json + perelman_{ts}.md

Navigation — Priority zones (zero cost)
────────────────────────────────────────────
11_zones_prioritaires.py     Croisement 09 × 10, aucun LLM
  └→ zones_{ts}.md  ← read before A/B/C

Deep segment analysis
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


---

## Prerequisites



### Python environment

```bash
conda create -n rag_historian python=3.11
conda activate rag_historian
pip install -r requirements.txt


On mac, if WeasyPrint échoue : `brew install pango cairo`

### Fichier `.env` (if open AI)

```
OPENAI_API_KEY=sk-...

### Ollama (local mode only)

```bash
# Install from https://ollama.com
ollama serve                      # run in the background before any _local script
ollama pull qwen2.5:14b           # default model

# Recommended alternatives for scripts 09/10/A/B/C:
ollama pull deepseek-r1:14b       # best for structured reasoning
ollama pull gemma3:12b            # excellent for academic French

#To change the model across the entire local pipeline, modify LLM_MODEL
# in rag_config_local.py — a single line update applies to all scripts.

### Directory structure

MonProjet/
├── .env                         ← OPENAI_API_KEY (Api only)
├── rag_config_local.py          ← config. local
├── config_00.py                 ← config. api
├── [tous les scripts .py]
├── pdfs/                        ← corpus PDF (B (without Zotero))
├── extracted_text/
│   ├── corpus.txt               ← produced by 01
│   └── chunks.json              ← produced by 02
├── vector_store/                ← produced by 03 (voie OpenAI)
│   ├── faiss.index
│   ├── metadata.json            ← enrichi avec ref_courte/ref_longue si Zotero
│   └── embeddings.npy
├── vector_store_local/          ← produced by 03_local (Ollama)
│   └── [mêmes fichiers]
├── metadata_refs.json           ← produced by 00_zotero_import 
└── resultats/                   ← outputs 06-11, A-D Ollama
├ outputs/                       ← outputs 06-11, A-D  api

## Workflows
### A — with Zotero 

```bash
# 1. Exporter from Zotero → CSV 

# 2. convert HTML to PDF
python 00a_html_to_pdf.py --csv IAHistoire.csv

# 3. Import 
python 00_zotero_import.py --csv IAHistoire.csv (your file)

# 4. Corpus
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py    # local recommandé
# or 
python 03_build_embeddings.py          # OpenAI
```

### Voie B — without Zotero

```bash
# PDFs in pdfs/ then :
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py    # or 03_build_embeddings.py
```

### ### Zotero update (Path A — incremental)

```bash
# When new references have been added in Zotero: :
python 00a_html_to_pdf.py              # if HTML
python 00b_zotero_update.py --csv IAHistoire.csv
# Rerun the analyses directly — no rebuild required
```

### Full analysis (common to both paths)

```bash
# Level 1
python 06_map_enrich.py --manuscrit mon_manuscrit.txt       # ou _local
python 07_map_critique.py --manuscrit mon_manuscrit.txt     # ou _local

# Level 2
python 09_map_argumentation_openai.py     # or _local

# Level 3
python 10_map_perelman_openai.py          # or _local

# Navigation — run before A/B/C
python 11_zones_prioritaires.py



# Deep segment analysis (after reading `zones_{ts}.md`)
python A_toulmin_segment.py       # ou _local
python B_perelman_segment.py      # ou _local
python C_rst_segment.py           # ou _local
python D_synthese_croisee.py      # ou _local
python E_recommandations          # ou _local
```
## Script descriptions

Each script is documented in more detail within the code.

### 00a — `00a_html_to_pdf.py`
Converts HTML snapshots saved by Zotero into PDFs using WeasyPrint.  
Generates a Markdown report listing successes and failures.  
Use `--dry-run` to preview the process.  
Use `--force` to reconvert files even if a PDF already exists.

### 00 — `00_zotero_import.py`
Reads the Zotero CSV export, builds `metadata_refs.json`, and copies PDFs into
the `pdfs/` directory. Automatically generates short references (e.g. Noiriel (1988))
and full references (e.g. Noiriel G., *Le Creuset français*, Seuil, 1988) based on
item type (article, book, chapter).  
Supports `--dry-run`.

---

### 00b — `00b_zotero_update.py`
Detects new Zotero items using their `Key`, extracts their text, chunks them,
and appends their vectors to the existing FAISS index without a full rebuild.  
Checks dimensional consistency (fails if the embedding model has changed).  
Deletions and modifications are ignored — perform a full rebuild if needed.

---

### 01 — `01_extract_text.py`
Extracts raw text from corpus PDFs. Supports both native and scanned documents (OCR).  
Output: `extracted_text/corpus.txt`

---

### 02 — `02_chunk_corpus.py`
Splits `corpus.txt` into size-controlled overlapping chunks.  
Output: `extracted_text/chunks.json`

---

### 03 — `03_build_embeddings.py` / `03_build_embeddings_local.py`
Encodes chunks into vectors and builds the FAISS index.  
If `metadata_refs.json` is present (Path A), injects `ref_courte` and `ref_longue`
into each chunk — these references propagate throughout all outputs.  
If absent (Path B), behavior remains unchanged (raw file names are used).  
Automatically resumes if interrupted.  

Outputs:
- `vector_store/` (OpenAI)
- `vector_store_local/` (local)

### 04 — `04_rag_query.py` / `04_rag_query_local.py`
Queries the indexed corpus, retrieves relevant segments, and returns a
synthesized response written in natural language.

---

### 05 — `05_rag_write.py` / `05_rag_write_local.py`
Searches the corpus for passages related to a given question, then generates
a structured academic paragraph grounded in the sources, with explicit citations.

---

### 06 — `06_map_enrich.py` / `06_map_enrich_local.py`
Enrichment mode. Constructive approach: *“What can the corpus contribute to this passage?”*  
Automatically resumes if interrupted.  

Outputs:
- `map_{ts}.json`
- `map_{ts}.md`

---

### 07 — `07_map_critique.py` / `07_map_critique_local.py`
Critical and relational mode. Qualifies the relationships between each paragraph
and corpus passages according to a six-relation taxonomy.  
Produces a fragility score per paragraph.  
Automatically resumes if interrupted.

**Six-relation taxonomy:**

| Relation | Description |
|----------|-------------|
| `supports` | The corpus confirms, supports, or illustrates the claim |
| `contradicts` | The corpus directly opposes the claim |
| `nuances` | The corpus adds complexity without invalidating the claim |
| `problematizes` | A claim presented as given is shown to be debated |
| `shifts` | The corpus introduces a different conceptual framework |
| `particularizes` | A general claim is shown to have highly variable local expressions |

Outputs:
- `critique.json`
- `critique.md`

### 09 — `09_map_argumentation_openai.py` / `09_map_argumentation_local.py`
Batch argumentation analysis (Toulmin / Adam / Walton) across the entire manuscript.  
See the epistemological warning — individual scores are **signals only**, not metrics.  
Use `--reset` to start from scratch.

**Scores (signals, not metrics):**
- `score_completude_toulmin` — proportion of the six Toulmin components present  
- `score_coherence_warrant` — strength of the link between grounds and claim  
- `score_charge_probatoire` — presence of explicit supporting grounds  
- `score_risque_sophisme` — presence of fallacious argumentation schemes  
- `score_robustesse_globale` — weighted synthesis  

Outputs:
- `argumentation_{ts}.json`
- `argumentation_{ts}.md`

---

### 10 — `10_map_perelman_openai.py` / `10_map_perelman_local.py`
Batch rhetorical analysis (Perelman). Can be chained with script 09 using
`--avec_argumentation`.

**Scores (signals, not metrics):**
- `score_force_persuasive` — rhetorical sophistication of the passage  
- `score_ancrage_auditoire` — alignment with disciplinary audience conventions  
- `score_coherence_valeurs` — internal coherence of the value system  
- `score_risque_sophistique` — problematic rhetorical usages  
- `score_profil_argumentatif` — weighted synthesis  

Outputs:
- `perelman_{ts}.json`
- `perelman_{ts}.md`

### 11 — `11_zones_prioritaires.py`
Cross-references the JSON outputs from scripts 09 and 10, computes six signals,
and identifies priority areas.  
**Cost: zero.** Always run before A/B/C.  

The generated Markdown report opens with a **“How to use this report”** section,
which explains:
- why not to analyze a single paragraph in isolation  
- the two ways to provide a segment (console input or `--file`)  
- a table indicating which script to run based on active signals  

---

**Six combined signals:**

| Signal | Name | Recommended script |
|--------|------|-------------------|
| A | `rhetoric_without_evidence` | Script A (priority) |
| B | `strong_form_weak_grounds` | Scripts A + B |
| C | `double_fallacy` | Script A (urgent) |
| D | `09_silent_10_speaks` | Script C (discursive coherence) |
| E | `disciplinary_authority` | Script B |
| F | `axiological_coherence` | Positive signal — stable area |

---

Output:
- `zones_{ts}.md`

### 11 — `11_zones_prioritaires.py`
Cross-references the JSON outputs from scripts 09 and 10, computes six signals,
and identifies priority areas.  
**Cost: zero.** Always run before A/B/C.  

The generated Markdown report opens with a **“How to use this report”** section,
which explains:
- why not to analyze a single paragraph in isolation  
- the two ways to provide a segment (console input or `--file`)  
- a table indicating which script to run based on active signals  

---

**Six combined signals:**

| Signal | Name | Recommended script |
|--------|------|-------------------|
| A | `rhetoric_without_evidence` | Script A (priority) |
| B | `strong_form_weak_grounds` | Scripts A + B |
| C | `double_fallacy` | Script A (urgent) |
| D | `09_silent_10_speaks` | Script C (discursive coherence) |
| E | `disciplinary_authority` | Script B |
| F | `axiological_coherence` | Positive signal — stable area |

---

Output:
- `zones_{ts}.md`

## Local LLM model selection

Modify `LLM_MODEL` in `rag_config_local.py` — a single line is sufficient.

| Model | RAM | Recommended use |
|-------|-----|------------------|
| `deepseek-r1:14b` | 16 GB | Scripts 09, 10, A, B, C — structured reasoning (Toulmin / Perelman / RST) |
| `gemma3:12b` | 16 GB | Scripts 09, 10, A, B, C — excellent for academic French |
| `qwen2.5:14b` | 16 GB | Default — well-balanced across all scripts |
| `mistral:7b-instruct` | 8 GB | Scripts 04, 05, 06, 07 — if RAM < 16 GB |
| `qwen2.5:7b` | 8 GB | If RAM < 16 GB — acceptable quality for 06/07 |

> **Note:** DeepSeek-R1 uses open weights and runs fully locally via Ollama.  
> Jurisdictional concerns apply only to cloud-based APIs, not to local execution.
## ⚠️ Local LLM compatibility warning

The local pipeline has not been tested with all LLMs available through Ollama.

In some cases—especially with smaller models, 
parsing issues may occur in scripts 06 to 10.  
These issues typically affect the computed scores, 
but not the generated syntheses or analytical outputs.
Scores can usually be retrieved throug raw outputs. 

As a result, scores may be incomplete or unreliable, while the qualitative analyses remain usable.

## Propagation of bibliographic references

When `metadata_refs.json` is present (Path A), script 03 injects
`ref_courte` and `ref_longue` into each chunk in `metadata.json`.  
These references are then propagated throughout the pipeline:

- In the Markdown reports from scripts 06 and 07: `**📄 Noiriel (1988), p. 3**`
- In prompts sent to the LLM in script 07: the model sees "Noiriel (1988)" and produces more accurate citations
- In the Markdown reports from scripts 09 and 10: `[Passage 1 — Noiriel (1988), p. 3]`

If `metadata_refs.json` is absent (Path B), raw file names are used as a fallback
throughout. No regression in functionality.

## Theoretical frameworks

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

## Known limitations of the system

**Non-metric scores (scripts 09/10).** The theoretical frameworks used (especially Toulmin)
were designed for first-person argumentation, not for texts that report, reconstruct,
or accumulate arguments over multiple layers.

**Polyphony in historical writing.** Historians quote and reconstruct opposing positions.
Script A partially addresses this issue, but the limitation remains structural.

**Context size.** An isolated paragraph is too short for rigorous Toulmin analysis.
Scripts A, B, and C address this limitation by working on user-defined segments.

**RST as approximation.** Script C produces a heuristic analysis, not a formal annotation.
Low-confidence relations are explicitly flagged.

**Alignment of 09/10 in script 11.** If the JSON files were generated from different corpora,
alignment by position may be incorrect. Script 11 displays a warning if lengths differ.

**Inter-model variability.** The quality of analysis depends on the LLM and its internal
representation of the theoretical frameworks. For reproducibility, document precisely
the model used (name, version, temperature, date).

---

## Common issues

**All scores at 0.50** — the LLM did not follow the expected `X/10` format.  
A robust `extract_scores()` function (handling decimal commas, short names,
and Markdown tables) is integrated into scripts 09, 10, A, and B.  
In API mode, a utility allows scores to be recalculated.

---

**"Not available" sections in script D** — the LLM used different separators
instead of `━━━`. A tolerant `extract_section_llm()` function is implemented in script D.

---

**Truncated outputs in A/B/C** — increase `MAX_TOKENS` to 4500 (A/B) or
5000 (C) at the top of the script.

---

**Ollama not running** — start `ollama serve` before any `_local` script.

---

**Missing `ref_courte` in outputs** — rerun `03_build_embeddings.py`
after `00_zotero_import.py`.

---

**Missing PDF for a Zotero item** — check `ZOTERO_STORAGE` in
`00_zotero_import.py`.

---

**WeasyPrint fails on macOS** — run `brew install pango cairo`.

---

*This pipeline was developed as part of a digital humanities research project.  
For academic use, cite the theoretical frameworks, the methodological limitations,
and the prototype references.*
