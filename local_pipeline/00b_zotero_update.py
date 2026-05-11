#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00b_zotero_update.py
====================
Voie A — Mise à jour incrémentale de l'index depuis Zotero.

Rôle : Détecte les nouveaux items dans le CSV Zotero par rapport à
l'index existant, extrait leur texte, les découpe en chunks, calcule
leurs embeddings, et les ajoute à l'index FAISS sans reconstruire
l'index complet.

À lancer quand le chercheur a enrichi sa bibliothèque Zotero et veut
intégrer les nouvelles références sans refaire tout le pipeline.

POSITION DANS LE PIPELINE (voie A — mise à jour)
──────────────────────────────────────────────────
  [première fois]
  00a_html_to_pdf.py      → conversion HTML
  00_zotero_import.py     → import initial
  01/02/03                → extraction, chunking, index complet

  [15 jours plus tard — nouvelles références ajoutées dans Zotero]
  00a_html_to_pdf.py      → si nouveaux HTML (optionnel)
  00b_zotero_update.py    → détecte les nouveautés, met à jour l'index
  ↑ ce script — les scripts 06 à D peuvent être relancés immédiatement après

CE QUE CE SCRIPT FAIT ET NE FAIT PAS
──────────────────────────────────────
Il FAIT :
  - Comparer le nouveau CSV Zotero avec metadata_refs.json (déjà indexé)
  - Identifier les items dont la clé Zotero (Key) est absente de l'index
  - Extraire le texte des nouveaux PDFs
  - Les découper en chunks avec les mêmes paramètres que 02_chunk_corpus.py
  - Calculer leurs embeddings avec le même modèle que 03_build_embeddings.py
  - Ajouter les vecteurs à l'index FAISS existant (index.add())
  - Mettre à jour metadata_refs.json et metadata.json

Il NE FAIT PAS :
  - Supprimer des items retirés de Zotero (ignorés silencieusement)
  - Mettre à jour les métadonnées d'items modifiés (ignorés)
  - Reconstruire l'index si le modèle d'embeddings a changé

POURQUOI PAS DE SUPPRESSION ?
───────────────────────────────
FAISS ne supporte pas la suppression de vecteurs individuels sans
reconstruire l'index. Supprimer un item de Zotero ne fait donc
aucun dommage : ses chunks restent dans l'index mais ne seront plus
mis à jour. Un rebuild complet (python 03_build_embeddings.py) est
nécessaire si la suppression d'items est importante. C'est documenté
ici comme une limite assumée.

COHÉRENCE DU MODÈLE D'EMBEDDINGS
──────────────────────────────────
Ce script utilise le même modèle d'embeddings que 03_build_embeddings.py
(lu depuis rag_config_local.py ou config_00.py). Si le modèle a changé
entre les deux runs, les vecteurs seront incohérents. Le script le
détecte et refuse de continuer si les dimensions ne correspondent pas.

UTILISATION
───────────
  python 00b_zotero_update.py
  python 00b_zotero_update.py --csv MaBiblio.csv
  python 00b_zotero_update.py --dry-run   # voir les nouveautés sans indexer
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

import os

CSV_ZOTERO     = "IAHistoire.csv"
ZOTERO_STORAGE = os.path.expanduser("~/Zotero/storage")
PDF_DIR        = "pdfs"
METADATA_REFS  = "metadata_refs.json"

# Chemins du vector store (identiques à 03_build_embeddings.py)
VECTOR_DIR     = "vector_store"
INDEX_FILE     = "vector_store/faiss.index"
METADATA_FILE  = "vector_store/metadata.json"

# Paramètres de chunking (doivent être identiques à 02_chunk_corpus.py)
MAX_TOKENS_CHUNK = 800
OVERLAP_TOKENS   = 150
TOKENIZER_ENC    = "cl100k_base"

# Longueur minimale de texte extrait pour qu'un PDF soit indexé
MIN_TEXT_CHARS = 200

MODE_FICHIER = "copie"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# RÉUTILISATION DES FONCTIONS DE 00_zotero_import.py
# =============================================================================
# Ces fonctions sont dupliquées ici pour que 00b soit autonome
# (pas d'import relatif entre scripts du pipeline).

def parser_auteurs(champ: str) -> list[dict]:
    if not champ.strip():
        return []
    auteurs = []
    for a in champ.split(";"):
        a = a.strip()
        if not a:
            continue
        if "," in a:
            parties = a.split(",", 1)
            nom, prenom = parties[0].strip(), parties[1].strip()
        else:
            nom, prenom = a, ""
        initiale = prenom[0] + "." if prenom else ""
        auteurs.append({"nom": nom, "prenom": prenom, "initiale": initiale})
    return auteurs


def auteurs_court(auteurs):
    if not auteurs:
        return "Anonyme"
    if len(auteurs) == 1:
        return auteurs[0]["nom"]
    if len(auteurs) == 2:
        return f"{auteurs[0]['nom']} & {auteurs[1]['nom']}"
    return f"{auteurs[0]['nom']} et al."


def auteurs_long(auteurs):
    if not auteurs:
        return "Anonyme"
    parties = [f"{a['nom']} {a['initiale']}".strip() for a in auteurs]
    if len(parties) == 1:
        return parties[0]
    if len(parties) == 2:
        return f"{parties[0]} & {parties[1]}"
    return ", ".join(parties[:-1]) + f" & {parties[-1]}"


def ref_courte(auteurs, annee):
    noms = auteurs_court(auteurs)
    return f"{noms} ({annee})" if annee else noms


def ref_longue(row, auteurs):
    noms  = auteurs_long(auteurs)
    titre = row.get("Title", "").strip()
    annee = row.get("Publication Year", "").strip()
    revue = row.get("Publication Title", "").strip()
    vol   = row.get("Volume", "").strip()
    num   = row.get("Issue", "").strip()
    pages = row.get("Pages", "").strip()
    edit  = row.get("Publisher", "").strip()
    lieu  = row.get("Place", "").strip()
    type_ = row.get("Item Type", "").strip()
    eds   = row.get("Editor", "").strip()
    vol_num = f"{vol}({num})" if vol and num else vol
    pp = f"pp. {pages}" if pages else ""
    def join(*p): return ", ".join(x for x in p if x)
    if type_ == "journalArticle":
        return join(noms, f'"{titre}"', revue, vol_num, annee, pp)
    elif type_ == "book":
        return join(noms, titre, edit, lieu, annee)
    elif type_ == "bookSection":
        dirs = auteurs_long(parser_auteurs(eds)) + " (dir.)" if eds else ""
        in_l = f"in {dirs}, {revue}" if dirs and revue else (
               f"in {revue}" if revue else "")
        return join(noms, f'"{titre}"', in_l, edit, annee, pp)
    elif type_ == "conferencePaper":
        return join(noms, f'"{titre}"', revue, annee, pp)
    elif type_ in ("preprint", "report"):
        label = "preprint" if type_ == "preprint" else "rapport"
        return join(noms, titre, f"[{label}]", annee)
    else:
        return join(noms, titre, annee)


def nom_fichier(auteurs, annee, existants):
    import unicodedata
    def norm(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-zA-Z0-9]", "", s)
    base = "".join(norm(a["nom"]) for a in auteurs[:2])
    base += annee if annee else "SansDate"
    candidat = base + ".pdf"
    if candidat not in existants:
        return candidat
    for lettre in "abcdefghijklmnopqrstuvwxyz":
        candidat = base + lettre + ".pdf"
        if candidat not in existants:
            return candidat
    return base + str(len(existants)) + ".pdf"


def cle_storage(att):
    m = re.search(r"storage[/\\]([A-Z0-9]{8})[/\\]", att, re.IGNORECASE)
    return m.group(1) if m else None


def trouver_pdf(att, zotero_storage):
    if att:
        chemins = [c.strip() for c in att.split(";") if c.strip()]
        for c in chemins:
            if c.lower().endswith(".pdf") and Path(c).exists():
                return Path(c)
        cle = cle_storage(att)
        if cle:
            dossier = zotero_storage / cle
            if dossier.exists():
                pdfs = list(dossier.glob("*.pdf"))
                if pdfs:
                    return pdfs[0]
    return None


# =============================================================================
# EXTRACTION DE TEXTE
# =============================================================================

def extraire_texte_pdf(chemin_pdf: Path) -> str:
    """
    Extrait le texte d'un PDF via pdfminer.six.
    Même méthode que 01_extract_text.py pour cohérence.
    """
    try:
        from pdfminer.high_level import extract_text
        texte = extract_text(str(chemin_pdf))
        return texte.strip() if texte else ""
    except ImportError:
        print("  ⚠ pdfminer.six non installé. Lancer : pip install pdfminer.six")
        return ""
    except Exception as e:
        print(f"  ⚠ Extraction échouée pour {chemin_pdf.name} : {e}")
        return ""


# =============================================================================
# CHUNKING
# =============================================================================

def chunker_texte(texte: str, source: str) -> list[dict]:
    """
    Découpe le texte en chunks avec chevauchement.
    Même logique que 02_chunk_corpus.py.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(TOKENIZER_ENC)
    except ImportError:
        print("  ⚠ tiktoken non installé. Lancer : pip install tiktoken")
        # Fallback : approximation 4 chars / token
        tokens = texte.split()
        enc    = None

    if enc:
        tokens = enc.encode(texte)
    else:
        tokens = texte.split()  # fallback mots

    chunks = []
    debut  = 0
    while debut < len(tokens):
        fin     = min(debut + MAX_TOKENS_CHUNK, len(tokens))
        segment = tokens[debut:fin]
        if enc:
            texte_chunk = enc.decode(segment)
        else:
            texte_chunk = " ".join(segment)

        if len(texte_chunk.strip()) > MIN_TEXT_CHARS:
            chunks.append({
                "source" : source,
                "page"   : len(chunks) + 1,
                "texte"  : texte_chunk.strip(),
            })
        debut += MAX_TOKENS_CHUNK - OVERLAP_TOKENS

    return chunks


# =============================================================================
# EMBEDDINGS ET MISE À JOUR FAISS
# =============================================================================

def charger_modele_embeddings():
    """Charge le modèle d'embeddings — même que 03_build_embeddings.py."""
    try:
        from rag_config_local import EMBEDDING_MODEL
        model_name = EMBEDDING_MODEL
    except ImportError:
        try:
            from config_00 import EMBEDDING_MODEL
            model_name = EMBEDDING_MODEL
        except ImportError:
            model_name = "paraphrase-multilingual-mpnet-base-v2"
            print(f"  ⚠ Config non trouvée — modèle par défaut : {model_name}")

    from sentence_transformers import SentenceTransformer
    print(f"  Chargement du modèle d'embeddings : {model_name}…")
    return SentenceTransformer(model_name), model_name


def ajouter_a_index(
    chunks_nouveaux: list[dict],
    metadata_refs: dict,
    nom_fichier_pdf: str,
) -> int:
    """
    Calcule les embeddings des nouveaux chunks et les ajoute à l'index FAISS.

    Vérifie que la dimension des vecteurs est cohérente avec l'index existant
    avant d'ajouter — refuse si incohérent (modèle changé).

    Returns:
        Nombre de chunks ajoutés.
    """
    import faiss
    import numpy as np

    index_path    = Path(INDEX_FILE)
    metadata_path = Path(METADATA_FILE)

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {index_path}\n"
            "Lancez d'abord 03_build_embeddings.py."
        )

    # Charger index et métadonnées existants
    index    = faiss.read_index(str(index_path))
    with open(metadata_path, encoding="utf-8") as f:
        metadata_chunks = json.load(f)

    # Charger modèle et calculer les embeddings
    model, _ = charger_modele_embeddings()
    textes   = [c["texte"] for c in chunks_nouveaux]
    vecteurs = model.encode(textes, convert_to_numpy=True).astype("float32")

    # Vérification de cohérence dimensionnelle
    dim_index  = index.d
    dim_vect   = vecteurs.shape[1]
    if dim_index != dim_vect:
        raise ValueError(
            f"Incohérence de dimension : index={dim_index}, "
            f"nouveaux vecteurs={dim_vect}.\n"
            "Le modèle d'embeddings a probablement changé.\n"
            "Reconstruire l'index complet : python 03_build_embeddings.py"
        )

    # Ajouter à l'index
    index.add(vecteurs)

    # Mettre à jour les métadonnées chunks
    ref = metadata_refs.get(nom_fichier_pdf, {})
    for chunk in chunks_nouveaux:
        metadata_chunks.append({
            "source"    : nom_fichier_pdf,
            "page"      : chunk["page"],
            "ref_courte": ref.get("ref_courte", nom_fichier_pdf),
            "ref_longue": ref.get("ref_longue", nom_fichier_pdf),
        })

    # Sauvegarder
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_chunks, f, ensure_ascii=False, indent=2)

    return len(chunks_nouveaux)


# =============================================================================
# DÉTECTION DES NOUVEAUTÉS
# =============================================================================

def detecter_nouveautes(
    chemin_csv: Path,
    metadata_refs: dict,
) -> list[dict]:
    """
    Compare le CSV Zotero avec les clés déjà indexées dans metadata_refs.json.

    Un item est considéré comme nouveau si sa clé Zotero (Key) n'apparaît
    pas dans metadata_refs.json. Les items modifiés et supprimés sont ignorés.

    Returns:
        Liste de rows CSV des items nouveaux qui ont un PDF.
    """
    # Clés déjà indexées
    cles_existantes = {
        v["key"] for v in metadata_refs.values() if "key" in v
    }

    nouveaux = []
    with open(chemin_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row.get("Key", "").strip()
            if key and key not in cles_existantes:
                nouveaux.append(row)

    return nouveaux


# =============================================================================
# RAPPORT
# =============================================================================

def generer_rapport(
    ajoutes: list, sans_pdf: list, erreurs: list, timestamp: str
) -> Path:
    lignes = [
        "# Rapport de mise à jour Zotero",
        "",
        f"**Timestamp** : {timestamp}  ",
        f"**Nouveaux items indexés** : {len(ajoutes)}  ",
        f"**Nouveaux sans PDF** : {len(sans_pdf)}  ",
        f"**Erreurs** : {len(erreurs)}  ",
        "",
        "---",
        "",
    ]
    if ajoutes:
        lignes += ["## Ajoutés à l'index", ""]
        for i in ajoutes:
            lignes.append(
                f"- `{i['nom']}` ({i['n_chunks']} chunks) — {i['ref_courte']}"
            )
        lignes.append("")
    if sans_pdf:
        lignes += ["## Nouveaux sans PDF (ignorés)", ""]
        for i in sans_pdf:
            a = i.get('auteur','').split(';')[0].strip()[:35]
            lignes.append(f"- {a} ({i.get('annee','')}) — {i.get('titre','')[:55]}")
        lignes.append("")
    if erreurs:
        lignes += ["## Erreurs", ""]
        for i in erreurs:
            lignes.append(f"- ❌ {i.get('titre','')[:55]} — {i.get('erreur','')}")
        lignes.append("")
    lignes += [
        "---", "",
        "## Note sur les suppressions", "",
        "Les items retirés de Zotero restent dans l'index FAISS.",
        "Pour nettoyer l'index : reconstruire complètement avec",
        "`python 00_zotero_import.py` + `python 03_build_embeddings.py`.",
        "",
        "*Rapport généré par `00b_zotero_update.py`.*",
    ]
    chemin = Path(f"rapport_update_zotero_{timestamp}.md")
    chemin.write_text("\n".join(lignes), encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="00b_zotero_update — Mise à jour incrémentale de l'index."
    )
    parser.add_argument("--csv", type=str, default=CSV_ZOTERO)
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les nouveautés sans indexer.")
    args = parser.parse_args()

    chemin_csv     = Path(args.csv)
    zotero_storage = Path(ZOTERO_STORAGE)
    pdf_dir        = Path(PDF_DIR)
    metadata_path  = Path(METADATA_REFS)
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  MISE À JOUR INCRÉMENTALE — Zotero → index FAISS")
    print("=" * 60)
    print(f"  CSV     : {chemin_csv}")
    print(f"  Mode    : {'dry-run' if args.dry_run else 'indexation'}")
    print()

    # Vérifications préalables
    for chemin, label in [
        (chemin_csv, "CSV Zotero"),
        (metadata_path, "metadata_refs.json"),
        (Path(INDEX_FILE), "index FAISS"),
    ]:
        if not chemin.exists():
            print(f"❌ {label} introuvable : {chemin.resolve()}")
            if label == "metadata_refs.json":
                print("   Lancer d'abord : python 00_zotero_import.py")
            if "FAISS" in label:
                print("   Lancer d'abord : python 03_build_embeddings.py")
            sys.exit(1)

    # Chargement des métadonnées existantes
    with open(metadata_path, encoding="utf-8") as f:
        metadata_refs = json.load(f)

    print(f"  Items déjà indexés : {len(metadata_refs)}")

    # Détection des nouveautés
    nouveaux_rows = detecter_nouveautes(chemin_csv, metadata_refs)
    print(f"  Nouveaux items détectés : {len(nouveaux_rows)}")

    if not nouveaux_rows:
        print("\n  Rien à ajouter — l'index est à jour.\n")
        sys.exit(0)

    if args.dry_run:
        print(f"\n  Mode --dry-run : {len(nouveaux_rows)} items seraient indexés :\n")
        for row in nouveaux_rows[:20]:
            aut = row.get("Author","").split(";")[0].strip()[:35]
            print(f"    {aut} ({row.get('Publication Year','')}) — "
                  f"{row.get('Title','')[:50]}")
        if len(nouveaux_rows) > 20:
            print(f"    … et {len(nouveaux_rows)-20} autres")
        sys.exit(0)

    # Traitement des nouveaux items
    pdf_dir.mkdir(parents=True, exist_ok=True)
    noms_existants = set(metadata_refs.keys())

    ajoutes  = []
    sans_pdf = []
    erreurs  = []

    for i, row in enumerate(nouveaux_rows, 1):
        key     = row.get("Key", "")
        att     = row.get("File Attachments", "").strip()
        auteurs = parser_auteurs(row.get("Author", ""))
        annee   = row.get("Publication Year", "").strip()
        titre   = row.get("Title", "").strip()

        print(f"  [{i:2d}/{len(nouveaux_rows)}] "
              f"{auteurs_court(auteurs)} ({annee}) — {titre[:45]}")

        # Trouver le PDF
        pdf_src = trouver_pdf(att, zotero_storage)
        if pdf_src is None:
            print(f"    ⚠ Pas de PDF — ignoré.")
            sans_pdf.append({"auteur": row.get("Author",""),
                             "titre": titre, "annee": annee})
            continue

        # Copier dans pdfs/
        nom = nom_fichier(auteurs, annee, noms_existants)
        noms_existants.add(nom)
        dest = pdf_dir / nom
        try:
            if not dest.exists():
                if MODE_FICHIER == "lien":
                    dest.symlink_to(pdf_src.resolve())
                else:
                    shutil.copy2(pdf_src, dest)
        except Exception as e:
            erreurs.append({"titre": titre, "erreur": str(e)})
            continue

        # Extraire le texte
        texte = extraire_texte_pdf(dest)
        if len(texte) < MIN_TEXT_CHARS:
            print(f"    ⚠ Texte trop court ({len(texte)} chars) — ignoré.")
            sans_pdf.append({"auteur": row.get("Author",""),
                             "titre": titre, "annee": annee})
            continue

        # Chunker
        chunks = chunker_texte(texte, nom)
        print(f"    {len(chunks)} chunks")

        # Construire les références et mettre à jour metadata_refs
        rc = ref_courte(auteurs, annee)
        rl = ref_longue(row, auteurs)

        metadata_refs[nom] = {
            "key"       : key,
            "auteur"    : row.get("Author", ""),
            "titre"     : titre,
            "annee"     : annee,
            "editeur"   : row.get("Publisher", ""),
            "revue"     : row.get("Publication Title", ""),
            "type_item" : row.get("Item Type", ""),
            "ref_courte": rc,
            "ref_longue": rl,
            "pdf_source": str(pdf_src),
        }

        # Ajouter à l'index FAISS
        try:
            n = ajouter_a_index(chunks, metadata_refs, nom)
            print(f"    ✅ {n} vecteurs ajoutés à l'index")
            ajoutes.append({
                "nom": nom, "ref_courte": rc,
                "n_chunks": n, "key": key,
            })
        except Exception as e:
            print(f"    ❌ Erreur indexation : {e}")
            erreurs.append({"titre": titre, "erreur": str(e)})

    # Sauvegarder metadata_refs mis à jour
    Path(METADATA_REFS).write_text(
        json.dumps(metadata_refs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Rapport
    rapport = generer_rapport(ajoutes, sans_pdf, erreurs, timestamp)

    print(f"\n{'═'*60}")
    print(f"  Ajoutés à l'index : {len(ajoutes)}")
    print(f"  Sans PDF          : {len(sans_pdf)}")
    print(f"  Erreurs           : {len(erreurs)}")
    print(f"{'═'*60}")
    print(f"\n  Rapport : {rapport}")
    print(f"\n  L'index est à jour — vous pouvez relancer 06, 07, etc.\n")


if __name__ == "__main__":
    main()
