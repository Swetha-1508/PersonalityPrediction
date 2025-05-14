import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    MODEL_CACHE_TIMEOUT = int(os.getenv('MODEL_CACHE_TIMEOUT', 3600))
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    ONTOLOGY_PATH = 'data/learning_ontology.owl'

    # SMOTE Configuration
    SMOTE_SAMPLING_STRATEGY = 'auto'
    SMOTE_K_NEIGHBORS = 5
    SMOTE_RANDOM_STATE = 42

    # Cross-Validation
    CV_FOLDS = 5
    CV_STRATIFIED = True
