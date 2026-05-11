

# Sorties RAG via API — Exemples analytiques

Ce fichier présente une sélection d'extraits courts (1 à 3 phrases) tirés de sorties de modèles via API.
L'objectif est de documenter les propriétés des sorties issues de la phase de test.

⚠️ Ces exemples ne nourrissent **pas des évaluations**.
Ils mettent en évidence des **patterns**, incluant points forts, ambiguïtés et limites.

---

## 1. Lissage argumentatif vs. ancrage dans le corpus


**Signal** : Claim sans backing explicite

**Extrait**
> « Cette distinction nouvelle s'est accompagnée d'une judiciarisation progressive de l'extradition, qui, tout en restant un acte de souveraineté, s'est inscrite dans un cadre légal plus rigoureux… »

**Interprétation**
Le passage formule un claim argumentatif clair (judiciarisation), mais le lien avec des segments précis du corpus reste implicite.

**Points forts**
- Cohérence rhétorique solide
- Progression argumentative claire

**Limites**
- Faible traçabilité vers les sources
- Compression de références hétérogènes en un énoncé généralisé

---

## 2. Dérive temporelle et complétion implicite

**Type** : Alignement avec le corpus (mode d'échec)
**Signal** : Extension au-delà du cadre temporel demandé

**Extrait**
> « Initialement, sous l'Ancien Régime… Cette conception se manifeste par des traités anciens, tels que celui conclu en 1174… »

**Interprétation**
Le modèle reconstruit un arrière-plan de longue durée pour stabiliser le récit, en introduisant des exemples situés hors de la période XIXe–XXe siècle demandée.

**Points forts**
- Apporte une profondeur contextuelle
- Renforce la continuité narrative

**Limites**
- Décalage par rapport au périmètre demandé
- Brouillage entre preuves récupérées et arrière-plan inféré

---

## 3. Surgénéralisation rhétorique


**Signal** : Abstraction de haut niveau et cadrage par le consensus

**Extrait**
> « Cette évolution traduit une tension persistante entre la nécessité de respecter les principes juridiques et les enjeux politiques sous-jacents… »

**Interprétation**
L'extrait produit un énoncé synthétique fort, organisant le matériau autour d'un cadre interprétatif très général.

**Points forts**
- Synthèse efficace
- Formulation lisible et réutilisable

**Limites**
- Perte de granularité analytique
- Faible différenciation entre cas et périodes

---

## 4. Inversion conceptuelle comme heuristique

**Type** : Cadrage argumentatif
**Signal** : Introduction d'un opérateur conceptuel structurant

**Extrait**
> « Ce n'est qu'à partir des Lumières… que la qualification d'« infraction politique » s'est inversée… »

**Interprétation**
La notion d'« inversion » offre une heuristique puissante pour organiser l'évolution, mais elle simplifie un processus vraisemblablement graduel et hétérogène.

**Points forts**
- Haute valeur analytique comme hypothèse structurante
- Fort potentiel explicatif

**Limites**
- Risque de sur-discrétisation
- Appui explicite sur les sources limité

---

## 5. Alignement localisé avec précision documentaire

**Type** : Enrichissement documentaire
**Signal** : Exemple précis avec référence explicite

**Extrait**
> « le traité franco-belge de 1869… marquant un élargissement progressif… »

**Interprétation**
Le modèle combine une référence historique précise avec un cadrage interprétatif, produisant une unité analytique équilibrée.

**Points forts**
- Bonne traçabilité
- Exemple concret étayant l'interprétation

**Limites**
- L'interprétation reste générale (« élargissement progressif »)

---

## 6. Subjectivation du raisonnement juridique

**Type** : Abstraction argumentative avancée
**Signal** : Glissement conceptuel (objet → sujet)

**Extrait**
> « l'extradition politique s'est déplacée de l'objet — l'infraction — au sujet… »

**Interprétation**
L'extrait condense une évolution jurisprudentielle complexe en une distinction analytique compacte.

**Points forts**
- Fort pouvoir explicatif
- Formulation analytique réutilisable

**Limites**
- Absence d'étapes de raisonnement intermédiaires
- Ancrage empirique immédiat limité

---

## Observations transversales sur les sorties via API

### 1. Cohérence plutôt que traçabilité
Les modèles via API tendent à privilégier la continuité discursive, parfois au détriment de l'alignement explicite avec les sources.

### 2. Comblement implicite des lacunes
Les maillons manquants sont souvent reconstruits sans justification explicite, produisant des effets de complétude.

### 3. Stabilisation rhétorique
Recours fréquent à des expressions généralisantes :
- « tension persistante »
- « évolution progressive »
- « transformation profonde »

Ces formules contribuent à un lissage interprétatif.

### 4. Micro-formulations fortes
Le modèle produit des expressions analytiques compactes et réutilisables, mais souvent sans décomposition détaillée.

---

## Usages suggérés

Ces exemples peuvent servir à :

- comparer le **comportement des modèles locaux et des modèles via API**
- alimenter des **flux de lecture critique assistée**
- guider les **étapes de post-traitement ou de validation humaine**