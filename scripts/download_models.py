# scripts/download_models.py - Run on Docker build for prod
import nltk
import spacy

# NLTK data (essential for tokenization/stopwords)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)  # For lemmatization fallback
nltk.download('omw-1.4', quiet=True)

# spaCy model (small English for prod balance)
spacy.cli.download("en_core_web_sm", quiet=True)

print("Models downloaded successfully.")