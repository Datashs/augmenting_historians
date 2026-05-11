# Test Dataset — RAG Argument Analysis Pipeline

## What this folder contains

This folder provides a **ready-to-use test dataset** so you can validate your installation and explore the pipeline's outputs without building a corpus from scratch.

```
test_dataset/
├── texts/
│   └── manuscript_demo.txt        ← demo manuscript (5–10 paragraphs)
├── pdfs/
│   └── [corpus PDF files]         ← bibliographic corpus (pre-selected)
├── embeddings_local/
│   ├── vector_store_local/        ← FAISS index (SentenceTransformers)
│   ├── extracted_text/
│   │   ├── corpus.txt             ← full extracted corpus text
│   │   ├── chunks.json            ← chunked corpus with overlap
│   │   └── metadata.json         ← chunk metadata (refs, positions)
├── embeddings_openai/
│   ├── vector_store/              ← FAISS index (OpenAI embeddings)
│   ├── extracted_text/
│   │   ├── corpus.txt
│   │   ├── chunks.json
│   │   └── metadata.json
```

---

## What you can do with this dataset

The pre-built embeddings let you **skip steps 01, 02, and 03** entirely and go straight to the analysis scripts. This is useful for:

- **Validating your installation** before running it on your own corpus
- **Exploring the pipeline's outputs** (argumentative signals, rhetorical profile, segment analysis)
- **Understanding what each script produces** without waiting for embedding generation

---

## How to use it — Local path (Ollama, zero cost, zero exposure)

**Prerequisites:** Ollama installed and running (`ollama serve`), at least one model pulled (e.g. `ollama pull qwen2.5:14b`).

**Step 1 — Copy the pre-built index to your project folder**

```bash
cp -r test_dataset/embeddings_local/vector_store_local/  MonProjet/vector_store_local/
cp -r test_dataset/embeddings_local/extracted_text/      MonProjet/extracted_text/
cp    test_dataset/texts/manuscript_demo.txt             MonProjet/manuscript.txt
```

**Step 2 — Run the analysis scripts directly**

```bash
cd MonProjet
conda activate rag_historian

# Corpus × manuscript mapping (Level 1)
python 06_map_enrich_local.py --manuscript manuscript.txt

# Critical reading (Level 1)
python 07_map_critique_local.py --manuscript manuscript.txt

# Batch argumentative signals (Level 2)
python 09_map_argumentation_local.py

# Batch rhetorical signals (Level 3)
python 10_map_perelman_local.py

# Priority zones — zero cost, no LLM
python 11_priority_zones.py

# Deep segment analysis — paste a segment from the demo manuscript
python A_toulmin_segment_local.py
python B_perelman_segment_local.py
python C_rst_segment_local.py

# Cross-synthesis
python D_cross-synthesis_local.py
```

All outputs are written to the `results/` folder.

---

## How to use it — OpenAI path

**Prerequisites:** A valid `.env` file with `OPENAI_API_KEY=sk-...` in your project folder.

**Step 1 — Copy the pre-built index**

```bash
cp -r test_dataset/embeddings_openai/vector_store/   MonProjet/vector_store/
cp -r test_dataset/embeddings_openai/extracted_text/ MonProjet/extracted_text/
cp    test_dataset/texts/manuscript_demo.txt         MonProjet/manuscript.txt
```

**Step 2 — Run the analysis scripts**

```bash
cd MonProjet
conda activate rag_historian

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

## Recommended first run (5 minutes, local, zero cost)

If you just want to check that everything works, run only these three scripts in order:

```bash
python 07_map_critique_local.py --manuscript manuscript.txt
python 09_map_argumentation_local.py
python 11_priority_zones.py
```

Script 11 costs nothing (no LLM call) and immediately shows you which zones the batch signals flagged as worth examining.

---

## What the demo manuscript is

`manuscript_demo.txt` is a short historical text  designed to exercise all three analysis levels: it contains passages with strong argumentative structure, passages with rhetorical density but thin evidentiary grounding, and passages with explicit citation of historiographical positions. This variation is intentional — it makes the signal differences between scripts 09, 10, and 11 visible immediately.

---

## Important reminder on scores

Scripts 09 and 10 produce **localization signals, not reliable paragraph-level metrics**. A low score on a single paragraph is not a judgment — it is an invitation to re-read that zone. Always run script 11 before interpreting any 09/10 output, and use scripts A, B, C for rigorous analysis of the zones it identifies.

See the main `README.md` for the full epistemological discussion.

---

## To use your own corpus instead

Once you have validated the installation with the test dataset, follow the full pipeline from step 01:

```bash
# Place your PDFs in pdfs/
python 01_extract_text.py
python 02_chunk_corpus.py
python 03_build_embeddings_local.py   # or 03_build_embeddings.py for OpenAI
# Then proceed from step 06 onwards
```

See the main `README.md` and the `pipeline_guide.pdf` for the complete workflow, including Zotero integration (Path A) and data sovereignty considerations.
