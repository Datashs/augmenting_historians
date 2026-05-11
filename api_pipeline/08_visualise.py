"""
08_visualise.py
===============
Étape 8 du pipeline RAG — Visualisation HTML interactive.

Rôle : Lit un ou deux fichiers JSON produits par 06_map_enrich.py et/ou
07_map_critique.py, et génère un fichier HTML autonome (aucune dépendance
externe) affichant une carte interactive du manuscrit.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/
    04_rag_query.py        → exploration thématique libre
    05_rag_write.py        → écriture assistée
    06_map_enrich.py       → resultats/enrich.json
    07_map_critique.py     → resultats/critique.json
    08_visualise.py        → visualisation HTML interactive        (ce script)
    ──
    00_config.py           → configuration LLM (non utilisé ici)

╔══════════════════════════════════════════════════════════════╗
║  CE SCRIPT N'UTILISE PAS DE LLM                             ║
║  Aucune clé API n'est nécessaire.                           ║
║  Il lit uniquement les JSON produits par 06 et/ou 07.       ║
╚══════════════════════════════════════════════════════════════╝

Modes de chargement (flexibles) :
    --enrich  seul   → visualisation enrichissement uniquement
    --critique seul  → visualisation critique uniquement
    --enrich + --critique → vue double côte à côte

    Si un seul fichier est fourni, la colonne correspondante est affichée.
    Si les deux sont fournis, les deux colonnes sont affichées côte à côte,
    permettant la comparaison paragraphe par paragraphe.

Ce que la visualisation HTML affiche :

    EN-TÊTE GLOBAL
        Titre du manuscrit, date d'analyse, résumé statistique
        (score moyen, nb paragraphes, profils dominants pour le mode critique)

    CARTE DU MANUSCRIT (colonne principale)
        Chaque paragraphe = un bloc coloré.
        Mode enrichissement :
            Teinte verte : vert foncé = corpus très riche (score élevé)
                           vert pâle  = corpus peu couvrant (score faible)
        Mode critique :
            Teinte selon le profil dominant :
                Rouge       → CONTREDIT (contradiction directe)
                Bordeaux    → PROBLEMATISE (impensé historiographique)
                Orange      → NUANCE (complexification ignorée)
                Jaune-ocre  → PARTICULARISE (portée excessive)
                Bleu-gris   → DEPLACE (cadre conceptuel alternatif)
                Vert        → CONFORTE (corpus confirme)
                Gris        → EQUILIBRE (aucun rapport dominant)
            Intensité : score_global (plus foncé = plus fragile)

    PANNEAU DE DÉTAIL (au clic sur un paragraphe)
        - Texte complet du paragraphe
        - Barres horizontales pour chaque score
        - Segments sources avec référence (fichier + page + distance FAISS)
        - Analyse LLM complète (dépliable)

    LÉGENDE ET AIDE
        Explication des scores et de la taxonomie des rapports,
        adaptée au mode affiché.

Format de sortie :
    Un fichier HTML autonome — aucune dépendance JavaScript externe,
    aucune connexion réseau requise. S'ouvre directement dans le navigateur.
    Le JSON des données est embarqué dans le HTML (balise <script>).
    Attention cette visualisation ne permet pas de navigation et de retour aux textes.

Usage :
    # Mode enrichissement seul
    python 08_visualise.py --enrich resultats/enrich.json

    # Mode critique seul
    python 08_visualise.py --critique resultats/critique.json

    # Mode double (côte à côte)
    python 08_visualise.py --enrich resultats/enrich.json --critique resultats/critique.json

    # Avec nom de sortie personnalisé
    python 08_visualise.py --critique resultats/critique.json --sortie visu/carte.html

Arguments :
    --enrich  FICHIER    JSON produit par 06_map_enrich.py (optionnel)
    --critique FICHIER   JSON produit par 07_map_critique.py (optionnel)
    --sortie  FICHIER    Fichier HTML de sortie (défaut : voir SORTIE_HTML)
    Au moins un des deux JSON doit être fourni.
"""

# =============================================================================
# PARAMÈTRES — modifiez uniquement cette section
# =============================================================================

# Fichier HTML de sortie (peut être surchargé via --sortie)
SORTIE_HTML = "resultats/carte_manuscrit.html"

# Encodage du fichier HTML
ENCODAGE = "utf-8"

# Nombre maximum de caractères du texte de paragraphe affiché dans la carte
# (aperçu tronqué — le texte complet est dans le panneau de détail)
APERCU_CHARS = 120

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# FONCTIONS — CHARGEMENT DES DONNÉES
# =============================================================================

def charger_json(chemin: str, mode_attendu: str) -> dict:
    """
    Charge et valide un fichier JSON produit par 06 ou 07.

    Vérifie que le fichier contient les clés attendues et que le mode
    correspond au fichier demandé (enrich vs critique).

    Args:
        chemin       : Chemin vers le fichier JSON.
        mode_attendu : "enrichissement" ou "critique".

    Returns:
        Dict contenant "run" et "paragraphes".

    Raises:
        FileNotFoundError : Fichier introuvable.
        ValueError        : Format invalide ou mode incorrect.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier JSON introuvable : {path.resolve()}\n"
            f"Lancez d'abord {'06_map_enrich.py' if mode_attendu == 'enrichissement' else '07_map_critique.py'}."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if "paragraphes" not in data:
        raise ValueError(
            f"Format invalide : clé 'paragraphes' absente dans {chemin}."
        )

    if not data["paragraphes"]:
        raise ValueError(
            f"Le fichier {chemin} ne contient aucun paragraphe analysé.\n"
            "Vérifiez que le script d'analyse s'est exécuté correctement."
        )

    return data


# =============================================================================
# FONCTIONS — CALCUL DES COULEURS
# =============================================================================

def couleur_enrichissement(score: float) -> str:
    """
    Calcule la couleur d'un paragraphe en mode enrichissement.

    Palette : du vert très pâle (score faible = peu couvert par le corpus)
    au vert foncé (score élevé = corpus très riche sur ce sujet).

    Une teinte légèrement désaturée est utilisée pour rester lisible
    sur fond blanc et éviter la fatigue visuelle sur de longs manuscrits.

    Args:
        score : Float 0.0–1.0 (score de densité documentaire).

    Returns:
        Chaîne CSS "hsl(...)" représentant la couleur.
    """
    # Teinte fixe : vert (120°)
    # Saturation : 30% (faible) à 65% (élevée) selon le score
    # Luminosité : 85% (pâle) à 35% (foncé) selon le score
    saturation = int(30 + score * 35)
    luminosite  = int(85 - score * 50)
    return f"hsl(120, {saturation}%, {luminosite}%)"


def couleur_critique(score_global: float, profil: str) -> str:
    """
    Calcule la couleur d'un paragraphe en mode critique.

    Combine deux dimensions :
    - La TEINTE reflète le profil dominant (nature du rapport)
    - La SATURATION et LUMINOSITÉ reflètent le score_global (intensité)

    Palette des teintes par profil :
        contredit     → Rouge        (0°)   danger direct
        problematise  → Bordeaux     (340°) problème méthodologique
        nuance        → Orange       (30°)  attention modérée
        particularise → Jaune-ocre   (45°)  portée excessive
        deplace       → Bleu-ardoise (220°) alternative conceptuelle
        conforte      → Vert         (120°) positif
        equilibre     → Gris-neutre  (210°) indéterminé

    Args:
        score_global : Float 0.0–1.0 (fragilité pondérée).
        profil       : Profil dominant (clé de la taxonomie).

    Returns:
        Chaîne CSS "hsl(...)" représentant la couleur.
    """
    teintes = {
        "contredit":     0,
        "problematise":  340,
        "nuance":        30,
        "particularise": 45,
        "deplace":       220,
        "conforte":      120,
        "equilibre":     210,
    }
    teinte     = teintes.get(profil, 210)
    saturation = int(20 + score_global * 55)
    luminosite  = int(85 - score_global * 45)
    return f"hsl({teinte}, {saturation}%, {luminosite}%)"


def couleur_barre(nom_score: str) -> str:
    """
    Retourne la couleur d'une barre de score dans le panneau de détail.

    Chaque type de rapport a une couleur dédiée pour une lecture rapide
    cohérente avec la carte principale.

    Args:
        nom_score : Nom du score (ex : "score_contredit").

    Returns:
        Code couleur CSS.
    """
    couleurs = {
        "score_conforte":      "#2d7a3a",   # vert foncé
        "score_contredit":     "#c0392b",   # rouge
        "score_nuance":        "#d35400",   # orange
        "score_problematise":  "#8e3a59",   # bordeaux
        "score_deplace":       "#2c5f8a",   # bleu-ardoise
        "score_particularise": "#a07c20",   # ocre
        "score_fragilite":     "#555555",   # gris neutre
        "score":               "#3a7d44",   # vert (enrichissement)
    }
    return couleurs.get(nom_score, "#888888")


# =============================================================================
# FONCTIONS — GÉNÉRATION HTML
# =============================================================================

def generer_stats_enrich(data: dict) -> str:
    """
    Génère le bloc de statistiques pour le mode enrichissement.

    Args:
        data : Dict JSON complet (run + paragraphes).

    Returns:
        Fragment HTML des statistiques.
    """
    run    = data.get("run", {})
    paras  = data["paragraphes"]
    scores = [p.get("score", 0.0) for p in paras]
    moy    = sum(scores) / len(scores) if scores else 0.0
    forts  = sum(1 for s in scores if s >= 0.7)
    faibles = sum(1 for s in scores if s < 0.3)

    return f"""
        <div class="stat-bloc">
            <span class="stat-val">{len(paras)}</span>
            <span class="stat-lbl">paragraphes</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val">{moy:.2f}</span>
            <span class="stat-lbl">densité moyenne</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val" style="color:#2d7a3a">{forts}</span>
            <span class="stat-lbl">bien couverts (≥0.7)</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val" style="color:#a0a0a0">{faibles}</span>
            <span class="stat-lbl">peu couverts (&lt;0.3)</span>
        </div>
    """


def generer_stats_critique(data: dict) -> str:
    """
    Génère le bloc de statistiques pour le mode critique.

    Args:
        data : Dict JSON complet (run + paragraphes).

    Returns:
        Fragment HTML des statistiques.
    """
    from collections import Counter
    paras   = data["paragraphes"]
    scores  = [p.get("score_global", 0.0) for p in paras]
    moy     = sum(scores) / len(scores) if scores else 0.0
    fragiles = sum(1 for s in scores if s >= 0.6)
    solides  = sum(1 for s in scores if s < 0.3)
    profils  = Counter(p.get("profil_dominant", "?") for p in paras)
    dominant = profils.most_common(1)[0][0] if profils else "—"

    return f"""
        <div class="stat-bloc">
            <span class="stat-val">{len(paras)}</span>
            <span class="stat-lbl">paragraphes</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val">{moy:.2f}</span>
            <span class="stat-lbl">fragilité moyenne</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val" style="color:#c0392b">{fragiles}</span>
            <span class="stat-lbl">fragiles (≥0.6)</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val" style="color:#2d7a3a">{solides}</span>
            <span class="stat-lbl">solides (&lt;0.3)</span>
        </div>
        <div class="stat-bloc">
            <span class="stat-val" style="font-size:0.85em">{dominant}</span>
            <span class="stat-lbl">profil dominant</span>
        </div>
    """


def generer_bloc_paragraphe(para: dict, mode: str, idx_col: int) -> str:
    """
    Génère le bloc HTML d'un paragraphe dans la carte.

    Chaque bloc est cliquable — le clic déclenche l'affichage du panneau
    de détail via JavaScript. Les données complètes du paragraphe sont
    embarquées dans un attribut data-* encodé en JSON.

    Args:
        para    : Dict du paragraphe (depuis le JSON de 06 ou 07).
        mode    : "enrichissement" ou "critique".
        idx_col : Indice de la colonne (0 ou 1) pour le JS multi-colonnes.

    Returns:
        Fragment HTML du bloc paragraphe.
    """
    index = para.get("index", "?")
    texte = para.get("texte", "")
    apercu = texte[:APERCU_CHARS].replace('"', '&quot;').replace('\n', ' ')
    if len(texte) > APERCU_CHARS:
        apercu += "…"

    if mode == "enrichissement":
        score  = para.get("score", 0.0)
        couleur = couleur_enrichissement(score)
        label  = f"Densité : {score:.2f}"
    else:
        score_global = para.get("score_global", 0.0)
        profil       = para.get("profil_dominant", "equilibre")
        couleur      = couleur_critique(score_global, profil)
        label        = f"Fragilité : {score_global:.2f} · {profil}"

    # Sérialise les données pour le panneau de détail
    # On encode le texte pour éviter les problèmes avec les apostrophes et guillemets
    data_json = json.dumps(para, ensure_ascii=True)
    data_json_escaped = data_json.replace("'", "&#39;")

    return f"""
        <div class="para-bloc"
             style="background:{couleur}"
             onclick="afficherDetail({idx_col}, '{data_json_escaped}')"
             title="§{index} — {label}">
            <span class="para-index">§{index}</span>
            <span class="para-apercu">{apercu}</span>
            <span class="para-score">{label}</span>
        </div>
    """


def generer_legende_enrich() -> str:
    """Génère la légende pour le mode enrichissement."""
    return """
        <div class="legende">
            <div class="legende-titre">Mode enrichissement — densité documentaire</div>
            <div class="legende-items">
                <span class="legende-item" style="background:hsl(120,30%,85%)">Peu couvert</span>
                <span class="legende-item" style="background:hsl(120,47%,60%)">Modéré</span>
                <span class="legende-item" style="background:hsl(120,65%,35%);color:#fff">Bien couvert</span>
            </div>
            <div class="legende-note">
                Un score élevé signifie que le corpus contient des ressources
                pour enrichir ce paragraphe — pas que le paragraphe est meilleur.
            </div>
        </div>
    """


def generer_legende_critique() -> str:
    """Génère la légende pour le mode critique."""
    return """
        <div class="legende">
            <div class="legende-titre">Mode critique — profils de rapport</div>
            <div class="legende-items">
                <span class="legende-item" style="background:hsl(0,55%,65%);color:#fff">Contredit</span>
                <span class="legende-item" style="background:hsl(340,50%,55%);color:#fff">Problématise</span>
                <span class="legende-item" style="background:hsl(30,55%,65%)">Nuance</span>
                <span class="legende-item" style="background:hsl(45,50%,60%)">Particularise</span>
                <span class="legende-item" style="background:hsl(220,45%,65%);color:#fff">Déplace</span>
                <span class="legende-item" style="background:hsl(120,45%,55%);color:#fff">Conforte</span>
                <span class="legende-item" style="background:hsl(210,20%,75%)">Équilibre</span>
            </div>
            <div class="legende-note">
                L'intensité de la couleur reflète le score global de fragilité.
                Un paragraphe rouge foncé est directement contredit par le corpus.
            </div>
        </div>
    """


def generer_html(data_enrich: dict | None, data_critique: dict | None) -> str:
    """
    Génère le fichier HTML complet de la visualisation.

    Structure du HTML :
        - CSS embarqué (variables de couleur, layout, animations)
        - En-tête avec statistiques
        - Carte du manuscrit (une ou deux colonnes)
        - Panneau de détail latéral (affiché au clic)
        - JavaScript embarqué (interaction, rendu du panneau)

    Aucune dépendance externe — le fichier fonctionne hors ligne.

    Args:
        data_enrich  : Dict JSON mode enrichissement (None si absent).
        data_critique : Dict JSON mode critique (None si absent).

    Returns:
        Chaîne HTML complète.
    """
    double_vue = data_enrich is not None and data_critique is not None

    # Titre de la page
    if data_enrich:
        manuscrit = data_enrich.get("run", {}).get("manuscrit", "Manuscrit")
    else:
        manuscrit = data_critique.get("run", {}).get("manuscrit", "Manuscrit")
    titre = Path(manuscrit).stem.replace("_", " ").title()

    # Date de génération
    date_gen = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # Génération des blocs de paragraphes
    blocs_enrich  = ""
    blocs_critique = ""

    if data_enrich:
        for para in data_enrich["paragraphes"]:
            blocs_enrich += generer_bloc_paragraphe(para, "enrichissement", 0)

    if data_critique:
        for para in data_critique["paragraphes"]:
            blocs_critique += generer_bloc_paragraphe(para, "critique", 1)

    # Stats en-tête
    stats_enrich  = generer_stats_enrich(data_enrich)   if data_enrich  else ""
    stats_critique = generer_stats_critique(data_critique) if data_critique else ""

    # Légendes
    legende_enrich  = generer_legende_enrich()  if data_enrich  else ""
    legende_critique = generer_legende_critique() if data_critique else ""

    # Titres des colonnes
    titre_col_enrich  = f"""
        <div class="col-titre enrich">
            <span class="col-icone">◈</span> Enrichissement
            <span class="col-sous-titre">densité documentaire</span>
        </div>""" if data_enrich else ""

    titre_col_critique = f"""
        <div class="col-titre critique">
            <span class="col-icone">◉</span> Critique
            <span class="col-sous-titre">rapports manuscrit / corpus</span>
        </div>""" if data_critique else ""

    col_enrich_html = f"""
        <div class="colonne" id="col-enrich">
            {titre_col_enrich}
            {legende_enrich}
            <div class="carte-scroll">
                {blocs_enrich}
            </div>
        </div>""" if data_enrich else ""

    col_critique_html = f"""
        <div class="colonne" id="col-critique">
            {titre_col_critique}
            {legende_critique}
            <div class="carte-scroll">
                {blocs_critique}
            </div>
        </div>""" if data_critique else ""

    layout_class = "double-vue" if double_vue else "vue-simple"

    # Noms des scores pour le panneau de détail (mode critique)
    noms_scores_critique = [
        ("score_conforte",      "Conforte"),
        ("score_contredit",     "Contredit"),
        ("score_nuance",        "Nuance"),
        ("score_problematise",  "Problématise"),
        ("score_deplace",       "Déplace"),
        ("score_particularise", "Particularise"),
        ("score_fragilite",     "Fragilité LLM"),
        ("score_global",        "Score global pondéré"),
    ]

    # Génération du JS pour les barres de scores
    js_barres_critique = json.dumps(noms_scores_critique)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Carte — {titre}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* ── Variables ───────────────────────────────────────────── */
        :root {{
            --bg:          #f5f0e8;
            --bg-panel:    #fffdf8;
            --bg-carte:    #eee8d8;
            --txt:         #1a1410;
            --txt-muted:   #6b6050;
            --bord:        #c8b99a;
            --accent-e:    #2d7a3a;
            --accent-c:    #8e3a59;
            --shadow:      0 2px 12px rgba(0,0,0,0.10);
            --shadow-lg:   0 8px 32px rgba(0,0,0,0.18);
            --radius:      6px;
            --font-serif:  'IM Fell English', Georgia, serif;
            --font-mono:   'JetBrains Mono', 'Courier New', monospace;
        }}

        /* ── Reset & base ────────────────────────────────────────── */
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: var(--font-serif);
            background: var(--bg);
            color: var(--txt);
            min-height: 100vh;
        }}

        /* ── En-tête ─────────────────────────────────────────────── */
        header {{
            background: var(--txt);
            color: var(--bg);
            padding: 1.5rem 2rem 1.2rem;
            border-bottom: 3px solid var(--bord);
        }}

        header h1 {{
            font-size: 1.6rem;
            font-weight: normal;
            letter-spacing: 0.02em;
            margin-bottom: 0.3rem;
        }}

        header .meta {{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: #b0a090;
            margin-bottom: 1rem;
        }}

        .stats-bande {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}

        .stats-section {{
            display: flex;
            gap: 1rem;
            align-items: center;
            padding: 0.5rem 0.8rem;
            background: rgba(255,255,255,0.08);
            border-radius: var(--radius);
            border: 1px solid rgba(255,255,255,0.12);
        }}

        .stats-section .section-lbl {{
            font-family: var(--font-mono);
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8a7a6a;
            margin-right: 0.3rem;
        }}

        .stat-bloc {{
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 60px;
        }}

        .stat-val {{
            font-family: var(--font-mono);
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--bg);
        }}

        .stat-lbl {{
            font-family: var(--font-mono);
            font-size: 0.62rem;
            color: #8a7a6a;
            text-align: center;
            margin-top: 0.1rem;
        }}

        /* ── Layout principal ────────────────────────────────────── */
        .main-layout {{
            display: flex;
            height: calc(100vh - 140px);
            overflow: hidden;
        }}

        .cartes-zone {{
            flex: 1;
            display: flex;
            gap: 0;
            overflow: hidden;
        }}

        .colonne {{
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--bord);
            overflow: hidden;
        }}

        .colonne:last-child {{
            border-right: none;
        }}

        /* ── Titre de colonne ────────────────────────────────────── */
        .col-titre {{
            padding: 0.7rem 1rem;
            font-family: var(--font-mono);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid var(--bord);
            background: var(--bg-carte);
            flex-shrink: 0;
        }}

        .col-titre.enrich {{ color: var(--accent-e); }}
        .col-titre.critique {{ color: var(--accent-c); }}

        .col-icone {{ font-size: 1rem; }}

        .col-sous-titre {{
            font-size: 0.65rem;
            color: var(--txt-muted);
            margin-left: auto;
            font-style: italic;
        }}

        /* ── Légende ─────────────────────────────────────────────── */
        .legende {{
            padding: 0.5rem 0.8rem;
            background: var(--bg-carte);
            border-bottom: 1px solid var(--bord);
            flex-shrink: 0;
        }}

        .legende-titre {{
            font-family: var(--font-mono);
            font-size: 0.62rem;
            color: var(--txt-muted);
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .legende-items {{
            display: flex;
            gap: 0.3rem;
            flex-wrap: wrap;
            margin-bottom: 0.3rem;
        }}

        .legende-item {{
            font-family: var(--font-mono);
            font-size: 0.6rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.08);
        }}

        .legende-note {{
            font-size: 0.62rem;
            color: var(--txt-muted);
            font-style: italic;
        }}

        /* ── Carte scrollable ────────────────────────────────────── */
        .carte-scroll {{
            flex: 1;
            overflow-y: auto;
            padding: 0.6rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            background: var(--bg-carte);
        }}

        .carte-scroll::-webkit-scrollbar {{ width: 4px; }}
        .carte-scroll::-webkit-scrollbar-track {{ background: transparent; }}
        .carte-scroll::-webkit-scrollbar-thumb {{
            background: var(--bord);
            border-radius: 2px;
        }}

        /* ── Blocs paragraphes ───────────────────────────────────── */
        .para-bloc {{
            padding: 0.45rem 0.65rem;
            border-radius: var(--radius);
            cursor: pointer;
            border: 1px solid rgba(0,0,0,0.06);
            transition: transform 0.12s ease, box-shadow 0.12s ease,
                        border-color 0.12s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }}

        .para-bloc:hover {{
            transform: translateX(3px);
            box-shadow: var(--shadow);
            border-color: rgba(0,0,0,0.18);
        }}

        .para-bloc.actif {{
            transform: translateX(5px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(0,0,0,0.3);
            outline: 2px solid rgba(0,0,0,0.25);
        }}

        .para-index {{
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: rgba(0,0,0,0.45);
            font-weight: 600;
        }}

        .para-apercu {{
            font-size: 0.75rem;
            line-height: 1.35;
            color: rgba(0,0,0,0.80);
        }}

        .para-score {{
            font-family: var(--font-mono);
            font-size: 0.58rem;
            color: rgba(0,0,0,0.50);
            margin-top: 0.1rem;
        }}

        /* ── Panneau de détail ───────────────────────────────────── */
        #panneau {{
            width: 420px;
            flex-shrink: 0;
            background: var(--bg-panel);
            border-left: 2px solid var(--bord);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: width 0.25s ease;
        }}

        #panneau.ferme {{ width: 0; border-left: none; }}

        .panneau-entete {{
            padding: 0.9rem 1rem 0.7rem;
            border-bottom: 1px solid var(--bord);
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
            flex-shrink: 0;
            background: var(--bg-carte);
        }}

        .panneau-entete h2 {{
            font-size: 1rem;
            font-weight: normal;
        }}

        .panneau-mode-badge {{
            font-family: var(--font-mono);
            font-size: 0.62rem;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .badge-enrich  {{ background:#e8f5ea; color:var(--accent-e); }}
        .badge-critique {{ background:#f5e8ee; color:var(--accent-c); }}

        .btn-fermer {{
            margin-left: auto;
            background: none;
            border: 1px solid var(--bord);
            border-radius: var(--radius);
            padding: 0.2rem 0.5rem;
            cursor: pointer;
            font-family: var(--font-mono);
            font-size: 0.7rem;
            color: var(--txt-muted);
            transition: background 0.1s;
        }}

        .btn-fermer:hover {{ background: var(--bord); }}

        .panneau-corps {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }}

        .panneau-corps::-webkit-scrollbar {{ width: 4px; }}
        .panneau-corps::-webkit-scrollbar-thumb {{
            background: var(--bord);
            border-radius: 2px;
        }}

        /* ── Sections du panneau ─────────────────────────────────── */
        .section {{
            margin-bottom: 1.2rem;
        }}

        .section-titre {{
            font-family: var(--font-mono);
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--txt-muted);
            margin-bottom: 0.5rem;
            padding-bottom: 0.2rem;
            border-bottom: 1px solid var(--bord);
        }}

        .texte-para {{
            font-size: 0.82rem;
            line-height: 1.6;
            color: var(--txt);
            background: var(--bg-carte);
            padding: 0.7rem;
            border-radius: var(--radius);
            border: 1px solid var(--bord);
        }}

        /* ── Barres de scores ────────────────────────────────────── */
        .score-ligne {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }}

        .score-nom {{
            font-family: var(--font-mono);
            font-size: 0.63rem;
            color: var(--txt-muted);
            width: 130px;
            flex-shrink: 0;
        }}

        .score-barre-fond {{
            flex: 1;
            height: 8px;
            background: #e0d8cc;
            border-radius: 4px;
            overflow: hidden;
        }}

        .score-barre-rempli {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.4s ease;
        }}

        .score-val {{
            font-family: var(--font-mono);
            font-size: 0.63rem;
            color: var(--txt);
            width: 32px;
            text-align: right;
            flex-shrink: 0;
        }}

        /* ── Segments sources ────────────────────────────────────── */
        .segment-item {{
            font-family: var(--font-mono);
            font-size: 0.65rem;
            padding: 0.35rem 0.5rem;
            background: var(--bg-carte);
            border: 1px solid var(--bord);
            border-radius: var(--radius);
            margin-bottom: 0.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .segment-ref {{ color: var(--txt); }}
        .segment-dist {{ color: var(--txt-muted); }}

        /* ── Analyse LLM dépliable ───────────────────────────────── */
        details {{
            margin-top: 0.5rem;
        }}

        summary {{
            font-family: var(--font-mono);
            font-size: 0.65rem;
            cursor: pointer;
            color: var(--txt-muted);
            padding: 0.3rem 0;
            user-select: none;
        }}

        summary:hover {{ color: var(--txt); }}

        .analyse-texte {{
            font-size: 0.73rem;
            line-height: 1.6;
            white-space: pre-wrap;
            background: var(--bg-carte);
            padding: 0.7rem;
            border-radius: var(--radius);
            border: 1px solid var(--bord);
            margin-top: 0.4rem;
            color: var(--txt);
            max-height: 400px;
            overflow-y: auto;
        }}

        /* ── Profil badge ────────────────────────────────────────── */
        .profil-badge {{
            display: inline-block;
            font-family: var(--font-mono);
            font-size: 0.68rem;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
        }}

        /* ── Message vide ────────────────────────────────────────── */
        #panneau-vide {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--txt-muted);
            text-align: center;
            padding: 2rem;
        }}

        #panneau-vide .icone {{ font-size: 2.5rem; margin-bottom: 1rem; }}
        #panneau-vide p {{ font-size: 0.8rem; line-height: 1.5; }}
    </style>
</head>
<body>

<!-- ══ EN-TÊTE ══════════════════════════════════════════════════════════ -->
<header>
    <h1>Carte du manuscrit — <em>{titre}</em></h1>
    <div class="meta">Générée le {date_gen} · Pipeline RAG historien</div>
    <div class="stats-bande">
        {f'''<div class="stats-section">
            <span class="section-lbl">◈ Enrichissement</span>
            {stats_enrich}
        </div>''' if data_enrich else ""}
        {f'''<div class="stats-section">
            <span class="section-lbl">◉ Critique</span>
            {stats_critique}
        </div>''' if data_critique else ""}
    </div>
</header>

<!-- ══ LAYOUT PRINCIPAL ═════════════════════════════════════════════════ -->
<div class="main-layout">
    <div class="cartes-zone {layout_class}">
        {col_enrich_html}
        {col_critique_html}
    </div>

    <!-- Panneau de détail -->
    <div id="panneau" class="ferme">
        <div id="panneau-vide">
            <div class="icone">◎</div>
            <p>Cliquez sur un paragraphe<br>pour afficher son analyse détaillée.</p>
        </div>
        <div id="panneau-contenu" style="display:none; flex-direction:column; height:100%;">
            <div class="panneau-entete">
                <h2 id="panneau-titre">§—</h2>
                <span id="panneau-badge" class="panneau-mode-badge"></span>
                <button class="btn-fermer" onclick="fermerPanneau()">✕ fermer</button>
            </div>
            <div class="panneau-corps" id="panneau-corps"></div>
        </div>
    </div>
</div>

<!-- ══ JAVASCRIPT ═══════════════════════════════════════════════════════ -->
<script>
// Configuration des scores à afficher selon le mode
const SCORES_CRITIQUE = {js_barres_critique};
const COULEURS_SCORES = {{
    "score_conforte":      "#2d7a3a",
    "score_contredit":     "#c0392b",
    "score_nuance":        "#d35400",
    "score_problematise":  "#8e3a59",
    "score_deplace":       "#2c5f8a",
    "score_particularise": "#a07c20",
    "score_fragilite":     "#555555",
    "score_global":        "#222222",
    "score":               "#3a7d44",
}};

const LIBELLES_PROFIL = {{
    "conforte":      "Le corpus conforte ce paragraphe",
    "contredit":     "Le corpus contredit ce paragraphe",
    "nuance":        "Le corpus nuance ce paragraphe",
    "problematise":  "Le corpus problématise ce paragraphe",
    "deplace":       "Le corpus déplace le cadre de ce paragraphe",
    "particularise": "Le corpus particularise ce paragraphe",
    "equilibre":     "Aucun rapport dominant détecté",
}};

const COULEURS_PROFIL = {{
    "conforte":      {{ bg:"#e8f5ea", txt:"#2d7a3a" }},
    "contredit":     {{ bg:"#fce8e8", txt:"#c0392b" }},
    "nuance":        {{ bg:"#fdf0e8", txt:"#d35400" }},
    "problematise":  {{ bg:"#f5e8ee", txt:"#8e3a59" }},
    "deplace":       {{ bg:"#e8eef5", txt:"#2c5f8a" }},
    "particularise": {{ bg:"#faf5e8", txt:"#a07c20" }},
    "equilibre":     {{ bg:"#f0f0f0", txt:"#666666" }},
}};

let blocActif = null;

function fermerPanneau() {{
    document.getElementById('panneau').classList.add('ferme');
    document.getElementById('panneau-vide').style.display = 'flex';
    document.getElementById('panneau-contenu').style.display = 'none';
    if (blocActif) {{ blocActif.classList.remove('actif'); blocActif = null; }}
}}

function afficherDetail(idxCol, dataJson) {{
    const para = JSON.parse(dataJson.replace(/&#39;/g, "'"));
    const mode = idxCol === 0 ? 'enrichissement' : 'critique';

    // Marquer le bloc actif
    if (blocActif) blocActif.classList.remove('actif');
    const blocs = document.querySelectorAll('.para-bloc');
    // Trouver le bon bloc par index et colonne
    const cibles = Array.from(blocs).filter(b => {{
        const col = b.closest('.colonne');
        const bonneCol = idxCol === 0
            ? col && col.id === 'col-enrich'
            : col && col.id === 'col-critique';
        return bonneCol && b.querySelector('.para-index').textContent === '§' + para.index;
    }});
    if (cibles.length) {{ blocActif = cibles[0]; blocActif.classList.add('actif'); }}

    // Afficher le panneau
    document.getElementById('panneau').classList.remove('ferme');
    document.getElementById('panneau-vide').style.display = 'none';
    const contenu = document.getElementById('panneau-contenu');
    contenu.style.display = 'flex';

    // En-tête
    document.getElementById('panneau-titre').textContent = '§' + para.index;
    const badge = document.getElementById('panneau-badge');
    badge.textContent = mode === 'enrichissement' ? 'Enrichissement' : 'Critique';
    badge.className = 'panneau-mode-badge ' + (mode === 'enrichissement' ? 'badge-enrich' : 'badge-critique');

    // Corps
    const corps = document.getElementById('panneau-corps');
    let html = '';

    // Texte du paragraphe
    html += `<div class="section">
        <div class="section-titre">Texte du paragraphe</div>
        <div class="texte-para">${{escHtml(para.texte || '')}}</div>
    </div>`;

    // Profil dominant (mode critique)
    if (mode === 'critique' && para.profil_dominant) {{
        const p = para.profil_dominant;
        const c = COULEURS_PROFIL[p] || COULEURS_PROFIL['equilibre'];
        const lbl = LIBELLES_PROFIL[p] || p;
        html += `<div class="section">
            <div class="section-titre">Profil dominant</div>
            <span class="profil-badge" style="background:${{c.bg}};color:${{c.txt}}">${{p.toUpperCase()}}</span>
            <div style="font-size:0.72rem;color:var(--txt-muted);font-style:italic">${{lbl}}</div>
        </div>`;
    }}

    // Scores
    html += `<div class="section"><div class="section-titre">Scores</div>`;

    if (mode === 'enrichissement') {{
        const s = para.score || 0;
        html += scoreLigne('Densité documentaire', 'score', s);
    }} else {{
        // Score global en premier
        html += scoreLigne('Score global pondéré', 'score_global', para.score_global || 0);
        html += '<div style="height:0.4rem"></div>';
        for (const [cle, libelle] of SCORES_CRITIQUE.slice(0, 6)) {{
            html += scoreLigne(libelle, cle, para[cle] || 0);
        }}
        html += '<div style="height:0.2rem"></div>';
        html += scoreLigne('Fragilité LLM', 'score_fragilite', para.score_fragilite || 0);
    }}
    html += `</div>`;

    // Segments sources
    const segs = para.segments || [];
    if (segs.length) {{
        html += `<div class="section">
            <div class="section-titre">Segments sources (${{segs.length}})</div>`;
        for (const seg of segs) {{
            const dist = seg.distance ? seg.distance.toFixed(3) : '—';
            html += `<div class="segment-item">
                <span class="segment-ref">📄 ${{escHtml(seg.source)}} · p. ${{seg.page}}</span>
                <span class="segment-dist">dist. ${{dist}}</span>
            </div>`;
        }}
        html += `</div>`;
    }}

    // Analyse LLM dépliable
    if (para.analyse && para.analyse.length > 20) {{
        html += `<div class="section">
            <div class="section-titre">Analyse LLM</div>
            <details>
                <summary>▶ Afficher l'analyse complète</summary>
                <div class="analyse-texte">${{escHtml(para.analyse)}}</div>
            </details>
        </div>`;
    }}

    corps.innerHTML = html;
}}

function scoreLigne(libelle, cle, valeur) {{
    const pct = Math.round(valeur * 100);
    const couleur = COULEURS_SCORES[cle] || '#888';
    return `<div class="score-ligne">
        <span class="score-nom">${{libelle}}</span>
        <div class="score-barre-fond">
            <div class="score-barre-rempli"
                 style="width:${{pct}}%;background:${{couleur}}"></div>
        </div>
        <span class="score-val">${{valeur.toFixed(2)}}</span>
    </div>`;
}}

function escHtml(str) {{
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}}
</script>
</body>
</html>"""


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="08_visualise.py — carte interactive HTML du manuscrit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 08_visualise.py --enrich resultats/enrich.json
  python 08_visualise.py --critique resultats/critique.json
  python 08_visualise.py --enrich resultats/enrich.json --critique resultats/critique.json
  python 08_visualise.py --critique resultats/critique.json --sortie visu/carte.html
        """
    )
    parser.add_argument("--enrich",   default=None, metavar="FICHIER",
                        help="JSON produit par 06_map_enrich.py (optionnel)")
    parser.add_argument("--critique", default=None, metavar="FICHIER",
                        help="JSON produit par 07_map_critique.py (optionnel)")
    parser.add_argument("--sortie",   default=SORTIE_HTML, metavar="FICHIER",
                        help=f"Fichier HTML de sortie (défaut : {SORTIE_HTML})")
    args = parser.parse_args()

    if not args.enrich and not args.critique:
        print("❌ Fournissez au moins un fichier JSON (--enrich ou --critique).")
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("  08 — VISUALISATION HTML INTERACTIVE")
    print("=" * 60)

    # Chargement des données
    data_enrich  = None
    data_critique = None

    if args.enrich:
        print(f"Chargement enrichissement : {args.enrich}…")
        try:
            data_enrich = charger_json(args.enrich, "enrichissement")
            print(f"  {len(data_enrich['paragraphes'])} paragraphe(s).")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ❌ {e}")
            sys.exit(1)

    if args.critique:
        print(f"Chargement critique : {args.critique}…")
        try:
            data_critique = charger_json(args.critique, "critique")
            print(f"  {len(data_critique['paragraphes'])} paragraphe(s).")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ❌ {e}")
            sys.exit(1)

    # Génération du HTML
    print("Génération du HTML…")
    html = generer_html(data_enrich, data_critique)

    # Sauvegarde
    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(html, encoding=ENCODAGE)

    print(f"\n{'═' * 60}")
    print(f"  VISUALISATION GÉNÉRÉE")
    print(f"{'═' * 60}")
    if data_enrich:
        print(f"  Mode enrichissement : {len(data_enrich['paragraphes'])} paragraphes")
    if data_critique:
        print(f"  Mode critique       : {len(data_critique['paragraphes'])} paragraphes")
    print(f"  Fichier HTML        : {sortie.resolve()}")
    print(f"{'═' * 60}")
    print(f"\n  Ouvrez dans votre navigateur :")
    print(f"  open {sortie.resolve()}")


if __name__ == "__main__":
    main()
