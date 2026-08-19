# Diagnostic des maladies foliaires par Deep Learning — PlantVillage (Sujet 9)

Mini-mémoire de Master 1 IA & Data Science (ISM). Un outil qui, à partir d'une **photo de feuille**, prédit parmi **38 classes** (14 cultures, maladies + feuilles saines) avec un **indice de confiance** et un **message d'abstention** si la prédiction est incertaine.

**Binôme :** Edem Kokou Emmanuel ASSILA · Said Mmadi SAANDA
**Enseignant :** M. Ahmeth Bachir DIOUF · Année 2025-2026

- **Application en ligne :** https://plant-disease-diagnostic-sujet-9.streamlit.app
- **Dépôt GitHub :** https://github.com/Emadesko/plant-disease-diagnostic
- **Mémoire (PDF) :** [`memoire/Memoire_PlantVillage.pdf`](memoire/Memoire_PlantVillage.pdf)

---

## Démarche

On entraîne et on compare **trois modèles** sur PlantVillage (~54 305 images, 38 classes) :

1. **CNN entraîné de zéro** (baseline) — 4 blocs convolutifs, ~262 k paramètres.
2. **Transfert d'apprentissage** — MobileNetV2 pré-entraîné sur ImageNet, extracteur gelé + tête à 38 sorties.
3. **Réglage fin (fine-tuning)** — 30 dernières couches de MobileNetV2 dégelées, learning rate 1e-5.

L'idée centrale : mesurer non seulement la performance sur le jeu de test propre, mais surtout la **généralisation à des photos réelles** hors du dataset.

## Résultats

**Sur le test PlantVillage (~8 130 images propres) :**

| Modèle | Accuracy | F1 macro |
|---|---|---|
| CNN from-scratch | 92,2 % | 0,901 |
| Transfert (gelé) | 93,8 % | 0,924 |
| Transfert + fine-tuning | 93,4 % | 0,920 |

**Sur 71 photos réelles (résultat-clé) :**

| Origine | Fine-tuné | From-scratch |
|---|---|---|
| KAG (proches du dataset, 18 img.) | 17/18 (94 %) | 17/18 (94 %) |
| NET (web, réelles, 53 img.) | 22/53 (42 %) | 9/53 (17 %) |
| Ensemble (71 img.) | 39/71 (55 %) | 26/71 (37 %) |

Sur les images propres, les trois modèles se valent (~93 %). Mais sur les vraies photos web, le transfert (**42 %**) généralise environ **2,5× mieux** que le CNN de zéro (**17 %**) : la performance sur un benchmark contrôlé ne garantit pas la robustesse sur le terrain (*domain gap*).

## Structure du dépôt

```
.
├── README.md
├── app.py                         # application Streamlit (à la racine = déploiement direct)
├── requirements.txt
├── class_names.json               # 38 classes (ordre du modèle)
├── plantvillage_mobilenet_finetuned.keras
├── .streamlit/
│   └── config.toml                # thème vert
├── assets/                        # captures d'écran de la démo (pour le README)
├── notebook/
│   └── Memoire_CNN.ipynb          # pipeline complet (données, 3 modèles, éval)
├── images_test/                   # photos réelles KAG/NET pour reproduire la Partie G
└── memoire/
    └── Memoire_PlantVillage.pdf
```

## Reproduire le notebook (Google Colab)

1. Ouvrir `notebook/Memoire_CNN.ipynb` dans Google Colab (Runtime GPU).
2. Fournir un `kaggle.json` (Kaggle → Settings → API) pour télécharger PlantVillage.
3. Exécuter les cellules dans l'ordre. Les modèles s'entraînent (ou se rechargent depuis le Drive si déjà entraînés).
4. **Partie G (test terrain) :** cloner ce dépôt à côté du notebook, ou uploader le dossier `images_test/`. La cellule lit directement les photos et affiche le tableau comparatif des deux modèles — aucun upload manuel nécessaire.

## Lancer l'application en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Puis ouvrir une photo de feuille (upload ou caméra). L'app affiche le diagnostic, l'indice de confiance, un conseil, et s'abstient sous 60 % de confiance.

## Avertissement

Outil d'**aide au dépistage de première ligne** — ce n'est pas un diagnostic définitif. En cas de doute, consulter un conseiller phytosanitaire.
