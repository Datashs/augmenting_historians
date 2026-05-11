# Examples — Local analytical outputs

This document presents a curated set of examples produced by the **local pipeline (Ollama-based execution)**.

Each example includes:

- a short excerpt (1–3 sentences)
- a structured analytical output
- an interpretation

These outputs are **heuristic signals**, not evaluations of historical truth.

---

# 1. Textual structure (RST)

## Excerpt

> “La France du dix-neuvième siècle et de la troisième République m’offrait un double avantage.  
> J’ai régulièrement arpenté ce terrain durant quelques décennies…”

## Detected structure

- central nucleus: justification of the research terrain  
- elaboration: familiarity and access to sources  
- concession: limits of individual work  
- causal chain: implications for historical practice  

## Interpretation

The segment combines:

- empirical justification  
- methodological reflection  

A weak transition between these two parts is detected :contentReference[oaicite:0]{index=0}  

## Insight

RST reveals a structural discontinuity embedded in an otherwise coherent paragraph.

---

# 2. Actionable recommendation

## Excerpt

> “J’ai choisi de travailler ce matériau dans le cadre d’un projet personnel…”

## Problem

Shift from empirical justification to methodological reflection without explicit transition.

## Recommendation

Add a linking sentence.

### Example

> “Cette contextualisation pragmatique ouvre ainsi la voie à une réflexion plus approfondie…”

## Interpretation

The system translates structural detection into a concrete editorial intervention :contentReference[oaicite:1]{index=1}  

---

# 3. Corpus misalignment (RAG)

## Excerpt

> “L'extradition est une procédure technique. Fréquente, réglée par des milliers de traités…”

## Observation

- no relevant sources retrieved  
- corpus unrelated to the topic  

## Interpretation

The available corpus focuses on digital humanities 
and does not support legal or diplomatic history.

## Insight

This reveals a mismatch between research question and corpus :contentReference[oaicite:2]{index=2}  

---

# 4. Balanced corpus interaction

## Excerpt

> “La naissance de vastes répertoires de textes numérisés… permet d’appréhender ces masses documentaires…”

## Output

- confirmation  
- nuance  
- problematization  

## Interpretation

The paragraph is well grounded but situated within methodological debates, 
that are not adressed by the manuscript.

## Insight

The system captures structured tensions rather than simple validation.

---

# 5. Argumentative structure (Toulmin)

## Excerpt

> “Ces préoccupations croisées expliquent pour une bonne part tant le choix du terrain…”

## Structure

- Claim ✔  
- Grounds ✔  
- Warrant ✔  
- Rebuttal ❌  

## Interpretation

The argument is coherent but insufficiently defended.

## Insight

A frequent pattern in academic writing: strong claims without explicit defense against counter-argument.

---

# 6. Rhetorical analysis (Perelman)

## Excerpt

> “Ce contexte apparaît comme idéal pour étudier…”

## Observation

- implicit universalization  
- disciplinary argumentation follows  

## Interpretation

The text shifts between universal and disciplinary audiences.

## Risk

- ambiguity in rhetorical positioning  

## Insight

Rhetorical analysis reveals audience instability not visible in logical structure.

---

# 7. Documentary enrichment — Low density

## Excerpt

> “Il permet quelques premiers constats…”

## Detection

- low documentary density  
- absence of empirical references  

## Interpretation

The paragraph is conceptually clear but empirically weak.

## Insight

The system detects lack of documentary support rather than logical flaws.

---

# 8. Documentary enrichment — Relevant case

## Excerpt

> “Ces histoires de l’extradition sont menées souvent en retraçant les étapes d’une affaire…”

## Detection

- medium documentary density  
- presence of actors (jurists, diplomats)  
- implicit historiographical framing  

## Suggested enrichment

- historiography of extradition  
- diplomatic practices  
- case-based studies  

## Interpretation

The suggestions are aligned with the paragraph’s logic.

## Example of integration

> “Ces approches rejoignent une historiographie récente qui insiste sur le rôle des diplomates et des juristes dans la formalisation progressive des normes d’extradition au XIXe siècle…”

## Insight

The system reinforces the argument rather than redirecting it.

---

# 9. Writing suggestion — Misaligned vs corrected

## Excerpt

> “L'extradition est une procédure technique…”

---

## Misaligned generation

> “La pratique des tribunaux arbitraires dans l’examen des attentes légitimes relatives aux investissements…”

## Problem

- domain drift  
- irrelevant but fluent text  

## Interpretation

This illustrates fluent but contextually incorrect generation.

---

## Corrected version

> “Longtemps considérée comme une procédure essentiellement technique, l’extradition s’inscrit pourtant dans des enjeux politiques et diplomatiques majeurs…”

## Why it works

- preserves original claim  
- extends argument  
- remains within domain  

## Insight

This shows the difference between:

- automatic text generation  
- controlled historical writing  

---

# 10. What these examples demonstrate

## Multi-layer system

The pipeline combines:

- corpus alignment  
- logical structure  
- rhetorical analysis  
- textual organization  
- documentary grounding  

---

## Function

It operates as a diagnostic tool for scholarly writing.

---

## Limitation

Outputs are:

- model-dependent  
- probabilistic  
- non-metric  
- dependent on composition of corpus
- quality of outputs dependant on model used

---

# Final note

This system does not replace historical reasoning.

It supports it by:

> making implicit structures visible and actionable
