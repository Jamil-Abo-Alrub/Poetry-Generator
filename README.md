# Générateur de Poésie  📜✍️

## 🎯 Objectif du Projet

Ce projet universitaire vise à créer un générateur de texte capable de produire des poèmes. Il utilise un modèle d'intelligence artificielle (Réseau de Neurones Récurrent - RNN) entraîné sur un corpus de poésie pour apprendre et imiter un style d'écriture. L'application est interactive grâce à une interface web simple d'utilisation.

---

## ✨ Fonctionnalités Principales

* **Génération de poèmes :** À partir d'une phrase d'amorce, l'IA compose la suite d'un poème.
* **Complétion de vers :** L'IA peut proposer une fin pour un vers de poème inachevé.
* **Interface web interactive :** Permet de tester facilement le générateur.

---

## 🛠️ Technologies Clés

* **Python :** Langage principal de développement.
* **TensorFlow (Keras) :** Pour la création et l'entraînement du modèle d'IA.
* **SpaCy :** Pour le traitement du langage naturel (analyse de texte) lors de la préparation des données d'entraînement.
* **Streamlit :** Pour construire l'interface web de démonstration.

---

## 🗂️ Organisation du Projet (Fichiers Importants)

* `streamlit_app.py`: Lance l'interface web.
* `train_poetry_model.py`: Script pour entraîner le modèle IA (si vous souhaitez le ré-entraîner).
* `config.py`: Contient les chemins vers le modèle et les données.
* `model_loader.py`, `text_generator.py`: Modules gérant le modèle et la génération.
* `french_poetry_generator_model.keras` (ou similaire) : Le modèle IA pré-entraîné.
* `french_tokenizer.pickle` (ou similaire) : Le "vocabulaire" utilisé par le modèle.
* `french_poetry_corpus.txt` (ou similaire) : Le corpus de poèmes français utilisé pour l'entraînement.

---

## 🚀 Démarrage Rapide

### Prérequis

* Python 3 (version 3.8 ou plus récente recommandée).
* `pip` (l'installateur de paquets Python).

### Installation des Dépendances

1.  **(Optionnel mais conseillé) Créez un environnement virtuel Python pour isoler les dépendances du projet.**
2.  Ouvrez un terminal ou une invite de commande dans le dossier du projet.
3.  Installez les bibliothèques nécessaires :
    ```bash
    pip install tensorflow numpy streamlit spacy
    ```
4.  Téléchargez le modèle linguistique SpaCy pour le français (utilisé lors de l'entraînement) :
    ```bash
    python -m spacy download fr_core_news_sm
    ```

### Utiliser l'Application avec le Modèle Pré-entraîné

C'est la manière la plus simple de voir le projet en action.

1.  **Vérifiez la configuration :** Assurez-vous que le fichier `config.py` contient les bons noms pour vos fichiers de modèle et de tokenizer pré-entraînés (par exemple, `french_poetry_generator_model.keras` et `french_tokenizer.pickle`).
2.  **Lancez l'application Streamlit :**
    ```bash
    streamlit run streamlit_app.py
    ```
    Un onglet devrait s'ouvrir dans votre navigateur web avec l'interface du générateur de poésie.

### Entraîner un Nouveau Modèle (Optionnel)

Si vous souhaitez entraîner le modèle vous-même (cela peut prendre du temps) :

1.  **Préparez un corpus :** Placez un fichier `.txt` contenant des poèmes français dans le dossier (par exemple, `french_poetry_corpus.txt`).
2.  **Modifiez `train_poetry_model.py` :** Vérifiez que les chemins et paramètres (nom du corpus, nom du modèle à sauvegarder, modèle SpaCy `fr_core_news_sm`) sont corrects.
3.  **Lancez l'entraînement :**
    ```bash
    python train_poetry_model.py
    ```
    Cela créera de nouveaux fichiers `.keras` (modèle) et `.pickle` (tokenizer).

---