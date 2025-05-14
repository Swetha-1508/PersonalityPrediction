import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from owlready2 import get_ontology
from config import Config
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningModel:
    def __init__(self):
        self.dropout_model = None
        self.style_model = None
        self.tokenizer = None
        self.bert_model = None
        self.ontology = None
        self.scaler = None  # Will be loaded from file
        self.load_models()

    def load_models(self):
        try:
            # Load all models and scaler
            self.dropout_model = joblib.load('data/dropout_predictor.pkl')
            self.style_model = joblib.load('data/learning_style_predictor.pkl')
            self.scaler = joblib.load('data/scaler.pkl')  # Load pre-fitted scaler

            # Load BERT components
            self.tokenizer = DistilBertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_model = DistilBertModel.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_model.eval()

            # Load ontology
            self.ontology = get_ontology(Config.ONTOLOGY_PATH).load()

            logger.info("All models and resources loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            raise RuntimeError("Initialization failed")

    def prepare_features(self, input_data):
        try:
            # Feature mapping - must match training data exactly
            features = np.array([
                float(input_data['age']),
                1 if input_data['gender'].lower() == 'male' else 0,
                ['high school', 'bachelor\'s', 'master\'s', 'phd'].index(input_data['education'].lower()),
                float(input_data['time_spent']),
                float(input_data['quiz_attempts']),
                float(input_data['quiz_scores']),
                float(input_data['forum_participation']),
                float(input_data['assignment_completion']),
                float(input_data['feedback_score'])
            ]).reshape(1, -1)

            # Scale features using the pre-fitted scaler
            return self.scaler.transform(features)

        except Exception as e:
            logger.error(f"Feature preparation error: {str(e)}")
            raise ValueError(f"Invalid input data: {str(e)}")

    # ... keep other methods the same ...