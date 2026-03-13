# RAG Technical Documentation Assistant

Système de question-réponse sur documentation technique utilisant RAG (Retrieval-Augmented Generation).

## Description

Projet développé dans le cadre de LJ WebData pour permettre l'interrogation de documentation technique en langage naturel. Le système indexe des documents PDF et génère des réponses contextuelles avec sources.

## Stack technique

- **LangChain** : Orchestration du pipeline RAG
- **OpenAI GPT-3.5** : Génération des réponses
- **FAISS** : Base vectorielle pour recherche sémantique
- **Streamlit** : Interface utilisateur

## Installation

### Prérequis

- Python 3.11+
- Clé API OpenAI

### Setup

```bash
# Cloner le repo
git clone https://github.com/[username]/rag-technical-docs.git
cd rag-technical-docs

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Ajouter vos PDFs dans le dossier data/
mkdir data
# Copier vos fichiers PDF dans data/
Lancement
Copystreamlit run app.py
L'application sera accessible sur http://localhost:8501

Utilisation
Entrer votre clé API OpenAI dans la sidebar
Les documents du dossier data/ sont automatiquement indexés
Poser vos questions dans le chat
Les sources sont affichées sous chaque réponse

### Configuration
Paramètres modifiables dans app.py :

chunk_size : Taille des chunks (défaut: 1000)
chunk_overlap : Chevauchement (défaut: 200)
k : Nombre de documents récupérés (défaut: 3)
temperature : Créativité du modèle (défaut: 0)
TODO
 Support de formats supplémentaires (Word, Markdown)
 Amélioration du prompt system
 Export des conversations
 Base vectorielle persistante
Contact
Oumayma Lamjar
lamjar.oumayma@gmail.com
LinkedIn"# Assistant-IA-RAG" 
