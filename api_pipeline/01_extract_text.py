#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================================
SCRIPT 01 : EXTRACTION DE TEXTE DEPUIS LES PDF
===============================================================================

Position dans la chaîne RAG :
    Ce script est le PREMIER maillon de la chaîne de traitement.
    Il transforme des fichiers PDF bruts en texte exploitable pour
    les étapes suivantes : nettoyage, chunking, indexation, requête.

    Chaîne complète :
        01_extract_text.py   ← CE SCRIPT  : PDF → texte brut + métadonnées
        02_chunk_corpus.py                       : découpage en segments
        03_build_embeddings.py                       : vectorisation (embeddings)
       
        04_rag_query.py                       : requête RAG

Ce que fait ce script :
    - Parcourt le dossier PDF_DIR et traite tous les fichiers .pdf trouvés
    - Détecte automatiquement si chaque PDF est natif (texte extractible)
      ou scanné (images uniquement)
    - Extrait le texte page par page avec PyMuPDF pour les PDF natifs
    - Applique Tesseract OCR pour les PDF scannés (si disponible)
    - Produit deux fichiers de sortie :
        corpus.txt   : texte brut concaténé de tous les PDF, avec marqueurs
                       de page [fichier — page N] pour la traçabilité
        metadata.json: métadonnées structurées par document et par page
                       (source, numéro de page, nb caractères, type de page,
                       méthode d'extraction) — utilisées par les scripts
                       suivants pour la citation des sources dans le RAG

Pourquoi deux fichiers de sortie ?
    Le fichier texte est optimisé pour la lisibilité et le traitement
    séquentiel (nettoyage, chunking). Le fichier JSON est optimisé pour
    la traçabilité : quand le RAG cite un passage, il peut remonter
    jusqu'au PDF source et à la page exacte. Séparer ces deux usages
    évite d'encombrer le texte avec des métadonnées qui perturbent
    les traitements NLP (traitement automatique du langage naturel).
    L'expression est un peu barbare, elle désigne
    l' ensemble des techniques informatiques qui permettent à une machine 
    de comprendre, analyser, manipuler ou produire du langage humain

Pourquoi PyMuPDF plutôt que pdfplumber ou pypdf ?
    PyMuPDF (fitz) offre le meilleur rapport qualité/performance pour
    les articles scientifiques : extraction texte robuste, gestion des
    encodages complexes, et accès natif aux images.
    Pour les documents avec tableaux complexes, pdfplumber serait
    préférable — mais les articles scientifiques sont majoritairement
    du texte linéaire.

Détection natif vs scanné :
    Une page est considérée scannée si le ratio texte/surface est
    inférieur à SEUIL_SCAN_CHARS_PAR_PAGE caractères. Ce seuil est
    configurable — une page de titre ou une page blanche peuvent
    déclencher le fallback OCR inutilement si le seuil est trop haut.
    La détection se fait page par page : un PDF peut avoir des pages
    natives et des pages scannées (cas fréquent dans les documents
    numérisés partiellement).

Dépendances :
    - PyMuPDF (pip install PyMuPDF) — extraction texte natif + images
    - pytesseract (pip install pytesseract) — OCR, optionnel
    - Tesseract installé sur le système (brew install tesseract
      ou apt install tesseract-ocr) — requis pour l'OCR

Structure attendue :
    MonProjet/
        01_extract_text.py   ← ce script, lancer depuis ici
        pdfs/                ← dossier source des PDF (PDF_DIR)
            article1.pdf
            article2.pdf
        extracted_text/      ← dossier de sortie (OUTPUT_DIR, créé auto)
            corpus.txt
            metadata.json

USAGE :
    python 01_extract_text.py
    python 01_extract_text.py --dossier mes_pdfs/
    python 01_extract_text.py --dossier mes_pdfs/ --langue fra
    python 01_extract_text.py --stats

ARGUMENTS :
    --dossier DOSSIER    Dossier contenant les PDF (défaut : pdfs/)
    --langue LANG        Langue pour Tesseract OCR (défaut : fra+eng)
                         Exemples : fra, eng, deu, fra+eng
    --stats              Afficher le détail par PDF après traitement

EXEMPLES :
    python 01_extract_text.py
    python 01_extract_text.py --dossier archives/ --langue fra
    python 01_extract_text.py --stats

Pièges Python et points d'attention :
    1. ENCODAGE UTF-8 :
       PyMuPDF retourne du texte en UTF-8. Toujours ouvrir les fichiers
       de sortie avec encoding='utf-8' dans les scripts suivants.

    2. PDF PROTÉGÉS :
       Certains PDF sont protégés contre l'extraction de texte.
       PyMuPDF lève une exception ou retourne un texte vide.
       Ce script signale ces fichiers sans interrompre le traitement.

    3. TESSERACT ET LES LANGUES :
       Tesseract doit avoir les modèles de langue installés séparément.
       'fra' nécessite tesseract-lang-fra (ou tesseract-ocr-fra sur apt).
       Si la langue demandée n'est pas disponible, Tesseract lève une
       exception — le script signale l'erreur et passe à la page suivante.

    4. PAGES MIXTES :
       Un article avec une figure pleine page déclenche le fallback OCR
       sur cette page — l'OCR d'une figure n'extrait que les légendes.
       C'est un comportement acceptable pour le RAG (la figure n'est
       pas du texte pertinent de toute façon).

    5. PERFORMANCES :
       L'OCR est lent (~2-5 secondes par page). Pour un corpus de 100
       articles avec quelques pages scannées, compter 5-10 minutes.
       Pour un corpus majoritairement scanné, envisager un pipeline
       OCR dédié (ocrmypdf) en prétraitement.

===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF

# =============================================================================
# PARAMÈTRES CONFIGURABLES
# =============================================================================
# Tous les paramètres ajustables se trouvent ici.
# Ne pas modifier le code en dessous de cette section pour un usage courant.
#
# Structure attendue :
#
#   MonProjet/
#       01_extract_text.py   ← lancer depuis ici
#       pdfs/                ← PDF_DIR
#       extracted_text/      ← OUTPUT_DIR (créé automatiquement)

# Dossier source contenant les PDF à traiter
PDF_DIR = Path("pdfs")

# Dossier de sortie (créé automatiquement s'il n'existe pas)
OUTPUT_DIR = Path("extracted_text")

# Nom du fichier texte concaténé
OUTPUT_TEXTE = "corpus.txt"

# Nom du fichier de métadonnées JSON
OUTPUT_META = "metadata.json"

# Nombre minimum de caractères par page pour considérer qu'elle est native
# En dessous de ce seuil, la page est traitée comme scannée → OCR
# Réduire si des pages de titre (peu de texte) déclenchent l'OCR inutilement
SEUIL_SCAN_CHARS_PAR_PAGE = 50

# Langue(s) pour Tesseract OCR
# Exemples : 'fra' (français), 'eng' (anglais), 'fra+eng' (les deux)
# Liste des langues installées : tesseract --list-langs
TESSERACT_LANGUE = "fra+eng"

# Séparateur entre les pages dans le fichier texte
# Modifiable selon les besoins du script de chunking suivant
SEPARATEUR_PAGE = "\n\n"

# Encodage de sortie
ENCODAGE = "utf-8"

# =============================================================================
# DÉTECTION ET EXTRACTION
# =============================================================================

def detecter_type_page(page: fitz.Page) -> tuple:
    r"""
    Détermine si une page est native (texte extractible) ou scannée.

    Retourne (type_page, texte) où type_page est 'natif' ou 'scanné'.

    Stratégie :
        On extrait d'abord le texte avec PyMuPDF. Si le résultat
        dépasse SEUIL_SCAN_CHARS_PAR_PAGE caractères significatifs
        (hors espaces et sauts de ligne), la page est native.
        Sinon elle est considérée scannée — soit parce que c'est une
        vraie image, soit parce que c'est une page quasi-vide (figure,
        page de titre minimaliste).

    Ce choix page par page (et non document par document) permet de
    gérer les PDF mixtes — fréquents dans les archives numérisées
    partiellement.
    """
    texte = page.get_text("text").strip()
    chars_significatifs = len(texte.replace(" ", "").replace("\n", ""))

    if chars_significatifs >= SEUIL_SCAN_CHARS_PAR_PAGE:
        return "natif", texte
    else:
        return "scanné", ""


def ocr_page(page: fitz.Page, langue: str) -> str:
    r"""
    Applique Tesseract OCR sur une page scannée.

    Convertit d'abord la page en image (300 DPI pour une qualité OCR
    acceptable), puis passe l'image à Tesseract.

    300 DPI est le minimum recommandé pour l'OCR. En dessous, le taux
    de reconnaissance chute significativement sur les petits caractères.

    Retourne le texte extrait, ou une chaîne vide en cas d'erreur.
    La gestion silencieuse des erreurs est délibérée : une page OCR
    ratée ne doit pas interrompre le traitement de tout le corpus.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Convertir la page en image à 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)  # 72 DPI natif → 300 DPI
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        texte = pytesseract.image_to_string(img, lang=langue)
        return texte.strip()

    except ImportError:
        return ""  # pytesseract ou PIL non installé
    except Exception as e:
        print(f"      ⚠️  OCR échoué : {e}")
        return ""


def extraire_pdf(pdf_path: Path, langue_ocr: str) -> tuple:
    r"""
    Extrait le texte d'un PDF, page par page.

    Retourne (pages_texte, pages_meta) :
      pages_texte : liste de chaînes (une par page)
      pages_meta  : liste de dicts avec les métadonnées de chaque page

    Métadonnées par page :
      - source    : nom du fichier PDF
      - page      : numéro de page (1-indexé)
      - type      : 'natif' ou 'scanné'
      - methode   : 'pymupdf' ou 'tesseract'
      - nb_chars  : nombre de caractères extraits
      - vide      : True si aucun texte extrait
    """
    pages_texte = []
    pages_meta  = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"   ❌ Impossible d'ouvrir {pdf_path.name} : {e}")
        return [], []

    n_natif  = 0
    n_scanne = 0

    for num_page, page in enumerate(doc, start=1):
        type_page, texte = detecter_type_page(page)

        if type_page == "natif":
            methode = "pymupdf"
            n_natif += 1
        else:
            # Fallback OCR
            texte   = ocr_page(page, langue_ocr)
            methode = "tesseract" if texte else "aucune"
            type_page = "scanné"
            n_scanne += 1

        pages_texte.append(texte)
        pages_meta.append({
            "source"   : pdf_path.name,
            "page"     : num_page,
            "type"     : type_page,
            "methode"  : methode,
            "nb_chars" : len(texte),
            "vide"     : len(texte.strip()) == 0,
        })

    doc.close()

    if n_scanne > 0:
        print(f"      {n_natif} page(s) native(s), "
              f"{n_scanne} page(s) scannée(s) → OCR")
    else:
        print(f"      {n_natif} page(s) native(s)")

    return pages_texte, pages_meta


# =============================================================================
# ÉCRITURE DES SORTIES
# =============================================================================

def ecrire_corpus(output_path: Path,
                  tous_textes: list,
                  tous_meta: list):
    r"""
    Écrit le fichier corpus.txt avec les marqueurs de page.

    Format d'un bloc page :
        [nom_fichier — page N]
        texte de la page

    Les marqueurs sont conçus pour être reconnaissables par le script
    de chunking suivant (02_chunk.py), qui peut s'en servir comme
    délimiteurs naturels ou les ignorer selon la stratégie choisie.

    Les pages vides sont incluses avec un commentaire explicite —
    leur présence dans le corpus permet de maintenir la correspondance
    entre les numéros de page du corpus.txt et les pages du PDF source.
    """
    with open(output_path, 'w', encoding=ENCODAGE) as f:
        for texte, meta in zip(tous_textes, tous_meta):
            # Marqueur de page — tracabilité pour le RAG
            f.write(f"[{meta['source']} — page {meta['page']}]\n")

            if meta['vide']:
                f.write("[page vide ou non extractible]\n")
            else:
                f.write(texte)

            f.write(SEPARATEUR_PAGE)


def ecrire_metadata(output_path: Path,
                    tous_meta: list,
                    stats_globales: dict):
    r"""
    Écrit le fichier metadata.json.

    Structure :
    {
        "extraction": {
            "date": "...",
            "nb_pdf": N,
            "nb_pages_total": N,
            "nb_pages_natives": N,
            "nb_pages_scannees": N,
            "nb_pages_vides": N,
        },
        "documents": [
            {
                "source": "article.pdf",
                "nb_pages": N,
                "pages": [
                    {"source": "...", "page": 1, "type": "natif", ...},
                    ...
                ]
            },
            ...
        ]
    }

    Ce format est pensé pour le script de chunking : il peut retrouver
    rapidement les métadonnées d'une page donnée via
    metadata["documents"][i]["pages"][j], et les attacher à chaque
    chunk pour la citation de sources dans le RAG.
    """
    # Regrouper par document
    docs = {}
    for meta in tous_meta:
        src = meta["source"]
        if src not in docs:
            docs[src] = []
        docs[src].append(meta)

    documents = [
        {"source": src, "nb_pages": len(pages), "pages": pages}
        for src, pages in docs.items()
    ]

    sortie = {
        "extraction" : {
            "date"              : datetime.now().strftime("%Y-%m-%d %H:%M"),
            **stats_globales,
        },
        "documents"  : documents,
    }

    with open(output_path, 'w', encoding=ENCODAGE) as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="01 : extraction de texte PDF pour pipeline RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Exemples :
  python 01_extract_text.py
  python 01_extract_text.py --dossier archives/ --langue fra
  python 01_extract_text.py --stats
        """
    )
    parser.add_argument('--dossier', default=str(PDF_DIR),
                        metavar='DOSSIER',
                        help=f"Dossier source des PDF (défaut : {PDF_DIR})")
    parser.add_argument('--langue', default=TESSERACT_LANGUE,
                        metavar='LANG',
                        help=f"Langue Tesseract OCR (défaut : {TESSERACT_LANGUE})")
    parser.add_argument('--stats', action='store_true',
                        help="Afficher le détail par PDF")
    args = parser.parse_args()

    pdf_dir = Path(args.dossier)
    if not pdf_dir.exists():
        print(f"❌ Dossier introuvable : {pdf_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_texte = OUTPUT_DIR / OUTPUT_TEXTE
    output_meta  = OUTPUT_DIR / OUTPUT_META

    # En-tête
    print("=" * 60)
    print("  EXTRACTION DE TEXTE PDF — pipeline RAG étape 1/6")
    print("=" * 60)
    print(f"\n  Source  : {pdf_dir.resolve()}")
    print(f"  Sortie  : {OUTPUT_DIR.resolve()}")
    print(f"  OCR     : Tesseract ({args.langue})")

    # Inventaire des PDF
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ Aucun fichier .pdf dans {pdf_dir}")
        sys.exit(1)
    print(f"\n  {len(pdf_files)} fichier(s) PDF trouvé(s)\n")

    # Traitement
    tous_textes = []
    tous_meta   = []

    nb_natif = nb_scanne = nb_vide = 0

    for pdf_file in pdf_files:
        print(f"  📄 {pdf_file.name}")
        pages_texte, pages_meta = extraire_pdf(pdf_file, args.langue)

        if not pages_meta:
            continue

        tous_textes.extend(pages_texte)
        tous_meta.extend(pages_meta)

        # Compteurs
        nb_natif  += sum(1 for m in pages_meta if m['type'] == 'natif')
        nb_scanne += sum(1 for m in pages_meta if m['type'] == 'scanné')
        nb_vide   += sum(1 for m in pages_meta if m['vide'])

        if args.stats:
            chars_total = sum(m['nb_chars'] for m in pages_meta)
            print(f"      {len(pages_meta)} pages — {chars_total:,} caractères")

    if not tous_meta:
        print("\n❌ Aucun texte extrait.")
        sys.exit(1)

    # Écriture des sorties
    print(f"\n💾 Écriture du corpus...")
    ecrire_corpus(output_texte, tous_textes, tous_meta)
    print(f"   → {output_texte} "
          f"({output_texte.stat().st_size:,} octets)")

    print(f"💾 Écriture des métadonnées...")
    stats_globales = {
        "nb_pdf"             : len(pdf_files),
        "nb_pages_total"     : len(tous_meta),
        "nb_pages_natives"   : nb_natif,
        "nb_pages_scannees"  : nb_scanne,
        "nb_pages_vides"     : nb_vide,
    }
    ecrire_metadata(output_meta, tous_meta, stats_globales)
    print(f"   → {output_meta}")

    # Bilan
    print(f"\n{'═'*60}")
    print(f"  BILAN")
    print(f"{'═'*60}")
    print(f"  PDF traités          : {len(pdf_files)}")
    print(f"  Pages total          : {len(tous_meta)}")
    print(f"    Natives (PyMuPDF)  : {nb_natif}")
    print(f"    Scannées (OCR)     : {nb_scanne}")
    print(f"    Vides              : {nb_vide}")
    print(f"  Caractères extraits  : "
          f"{sum(m['nb_chars'] for m in tous_meta):,}")
    print(f"{'═'*60}")
    print(f"\n  Fichiers produits :")
    print(f"    {output_texte.name:<20} texte brut pour les étapes suivantes")
    print(f"    {output_meta.name:<20} métadonnées pour la traçabilité RAG")
    print(f"\n  Étape suivante : python 02_clean_text.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
