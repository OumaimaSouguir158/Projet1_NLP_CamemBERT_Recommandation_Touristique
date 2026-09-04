# Projet1_NLP_CamemBERT_Recommandation_Touristique

# Système de recommandation touristique par NLP
### Fine-tuning CamemBERT vs TF-IDF — Destinations tunisiennes

> **Auteure :** Oumaima Souguir  
> **Diplôme :** Licence Informatique Générale — CNAM Paris (mention Très Bien, 16.97/20, 180 ECTS)  
> **Environnement :** Google Colab T4 (GPU gratuit) + PC local Intel i3 / 8 GB RAM  
> **Domaine :** Traitement automatique du langage naturel (NLP) · Recommandation · Tourisme

---

## Présentation du projet

Ce projet construit et compare deux systèmes NLP capables de détecter l'**intention touristique** d'un utilisateur à partir d'un avis textuel en français, puis de recommander les **top-5 destinations tunisiennes** correspondantes.

**Deux approches sont comparées :**

| Approche | Méthode | F1-macro estimé |
|----------|---------|-----------------|
| Baseline | TF-IDF + Régression logistique | ~0.85 |
| Avancée | CamemBERT fine-tuné (110M params) | ~0.95+ |

**Exemple d'utilisation :**
```
Entrée  : "Je veux visiter des sites historiques et des ruines romaines"
Sortie  : Intention = culture (confiance : 0.97)
          → Tunis, Carthage, Dougga, El Jem, Matmata
```

---

## Architecture du pipeline

```
Avis texte brut (français)
        │
        ▼
┌───────────────────┐
│   Prétraitement   │  tokenisation, nettoyage, encodage des labels
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
TF-IDF    CamemBERT
+ LR      fine-tuné
    │         │
    └────┬────┘
         │
         ▼
┌──────────────────────────┐
│  Extraction d'intention  │  5 catégories : plage, culture,
│  + score de confiance    │  gastronomie, aventure, bien-être
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Top-5 recommandations   │  avec scores de similarité cosinus
└────────────┬─────────────┘
             │
             ▼
     Interface Gradio
     (lien public Colab)
```

---

## Structure des fichiers

```
projet1_nlp/
│
├── Projet1_NLP_CamemBERT_Recommandation_Touristique.ipynb   ← Notebook principal (Colab)
├── Rapport_Projet1_NLP_CamemBERT_Recommandation.docx         ← Rapport académique
└── README_Projet1_NLP_CamemBERT.md                           ← Ce fichier
```

---

## Démarrage rapide (Google Colab)

### Étape 1 — Importer le notebook

```
Google Colab → Fichier → Importer un notebook → Choisir le fichier .ipynb
```

### Étape 2 — Activer le GPU T4

```
Exécution → Modifier le type d'exécution → Accélérateur matériel → GPU (T4)
```

### Étape 3 — Exécuter dans l'ordre

Les 7 cellules sont conçues pour s'exécuter séquentiellement. Aucune configuration supplémentaire n'est requise.

```
Cellule 0 → Vérification GPU + installation des dépendances
Cellule 1 → Construction du dataset (50 avis × 5 catégories)
Cellule 2 → Pipeline TF-IDF + Régression logistique (baseline)
Cellule 3 → Chargement CamemBERT + fine-tuning (~8 min sur T4)
Cellule 4 → Fonction de recommandation CamemBERT
Cellule 5 → Comparaison et visualisation matplotlib
Cellule 6 → Interface Gradio interactive (lien public généré)
```

---

## Dépendances

Toutes les installations sont gérées automatiquement dans la **Cellule 0** du notebook.

```python
# Installé automatiquement dans Colab
transformers    # CamemBERT, Trainer API
datasets        # Dataset HuggingFace
scikit-learn    # TF-IDF, métriques, CV-5
torch           # PyTorch backend, mixed precision fp16
gradio          # Interface interactive
matplotlib      # Visualisation des résultats
accelerate      # Optimisation entraînement
sentencepiece   # Tokenizer CamemBERT
```

---

## Résultats attendus

### Métriques de classification

```
TF-IDF (CV-5)      F1-macro : 0.82 – 0.88   Temps : < 1 seconde
CamemBERT (val)    F1-macro : 0.93 – 0.97   Temps : 5 – 10 min (T4)
```

### Exemples de recommandations

```
Requête : "Trek dans le désert, nuit sous les étoiles"
→ Intention : aventure (confiance : 0.96)
→ Top destinations : Douz · Ain Draham · Matmata · Tozeur · Bou Hedma

Requête : "Spa, hammam et soins bien-être"
→ Intention : bien-être (confiance : 0.94)
→ Top destinations : Hammamet · Korbous · Djerba · Kerkennah · Tunis

Requête : "Gastronomie locale, fruits de mer frais"
→ Intention : gastronomie (confiance : 0.91)
→ Top destinations : Sfax · Tunis · Sidi Bou Said · Mornag
```

---

## Catégories d'intention et destinations

| Catégorie | Destinations couvertes |
|-----------|----------------------|
| plage | Hammamet, Sousse, Djerba, Kerkennah, Tabarka |
| culture | Tunis, Carthage, Dougga, El Jem, Matmata |
| gastronomie | Tunis, Sfax, Sidi Bou Said, Mornag |
| aventure | Douz, Ain Draham, Matmata, Tozeur, Bou Hedma |
| bien-être | Hammamet, Korbous, Djerba, Kerkennah, Tunis |

---

## Choix techniques justifiés

**Pourquoi CamemBERT et pas BERT multilingue ?**  
CamemBERT est entraîné sur 138 GB de texte français natif (Common Crawl), ce qui le rend plus performant que mBERT sur les tâches françaises, notamment sur les avis touristiques avec vocabulaire spécialisé.

**Pourquoi fp16 (mixed precision) ?**  
La précision mixte réduit la mémoire GPU d'environ 40 % sans perte de qualité significative, permettant de faire tourner CamemBERT sur le T4 Colab gratuit (16 GB VRAM) avec un batch size de 8.

**Pourquoi TF-IDF avec bigrammes ?**  
Les bigrammes (ngram_range=(1,2)) capturent des expressions fréquentes dans les avis touristiques : "fruits de mer", "sous les étoiles", "bord de mer" — que les unigrammes manquent.

**Pourquoi Gradio et pas Streamlit ?**  
Gradio génère automatiquement un lien public partageable depuis Colab sans configuration serveur — idéal pour une démonstration académique.

---

## Extension et mise en production

```
Dataset réel       → Scraping TripAdvisor/Booking.com + annotation manuelle (5 000+ exemples)
Multilingue        → Extension à l'arabe tunisien (darija) via AraBERT ou DarijaBERT
API production     → FastAPI + uvicorn + Docker
Base vectorielle   → FAISS + sentence-transformers pour similarité sémantique dense
Monitoring         → MLflow pour le tracking des expériences
RAG                → Couplage avec un LLM pour des recommandations narratives (Projet 3)
```

---

## Limites connues

- **Dataset synthétique** (50 exemples) : construit manuellement pour la démonstration. Les métriques sont indicatives et non généralisables sans données réelles annotées.
- **Session Colab** : limite de 12h sur GPU gratuit. Sauvegarder les checkpoints régulièrement (`save_strategy='epoch'`).
- **Langage familier** : CamemBERT est moins performant sur les avis très courts ou en langage SMS.

---

## Références

- Martin et al. (2020). *CamemBERT: a Tasty French Language Model*. ACL 2020.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL 2019.
- Wolf et al. (2020). *Transformers: State-of-the-Art NLP*. EMNLP 2020.
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
- Sparck Jones, K. (1972). *A statistical interpretation of term specificity*. Journal of Documentation.

---

## Licence

Projet académique — usage éducatif.  
Code source disponible sous licence MIT.

---

*Projet réalisé dans le cadre d'une candidature en Master Intelligence Artificielle.*  
*Environnement reproductible : Google Colab (gratuit) — aucun GPU local requis.*
