import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from owlready2 import *
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data(n_samples=10000):
    """Generate synthetic training data with consistent feature space"""
    np.random.seed(42)
    
    data = {
        'age': np.clip(np.random.normal(22, 5, n_samples), 16, 40),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
        'education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], 
                                    n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'time_spent': np.clip(np.random.normal(15, 5, n_samples), 5, 30),
        'quiz_attempts': np.random.poisson(5, n_samples),
        'quiz_scores': np.clip(np.random.normal(75, 15, n_samples), 0, 100),
        'forum_participation': np.random.poisson(3, n_samples),
        'assignment_completion': np.clip(np.random.normal(80, 15, n_samples), 0, 100),
        'feedback_score': np.clip(np.random.normal(7, 1.5, n_samples), 1, 10),
        'Dropout': np.random.binomial(1, 0.15, n_samples),  # 15% dropout rate
        'Learning_Style': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }
    return pd.DataFrame(data)

def preprocess_features(df):
    """Convert categorical features to numerical and return X, y"""
    # Gender mapping
    gender_map = {'Male': 1, 'Female': 0, 'Other': 0.5}
    
    # Education mapping (must match what's in model.py)
    education_map = {
        'High School': 0,
        'Bachelor\'s': 1,
        'Master\'s': 2,
        'PhD': 3
    }
    
    X = pd.DataFrame({
        'age': df['age'],
        'gender': df['gender'].map(gender_map),
        'education': df['education'].map(education_map),
        'time_spent': df['time_spent'],
        'quiz_attempts': df['quiz_attempts'],
        'quiz_scores': df['quiz_scores'],
        'forum_participation': df['forum_participation'],
        'assignment_completion': df['assignment_completion'],
        'feedback_score': df['feedback_score']
    })
    
    return X, df['Dropout'], df['Learning_Style']

def train_dropout_model(X, y):
    """Train dropout prediction model with proper feature scaling"""
    pipeline = ImbPipeline([
        ('smote', SMOTE(
            sampling_strategy='auto',
            k_neighbors=5,
            random_state=42
        )),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        ))
    ])
    
    # Train model
    pipeline.fit(X, y)
    
    # Save the scaler separately for use in prediction
    joblib.dump(pipeline.named_steps['scaler'], 'data/dropout_scaler.pkl')
    
    return pipeline

def train_style_model(X, y):
    """Train learning style classifier"""
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline.fit(X, y)
    joblib.dump(pipeline.named_steps['scaler'], 'data/style_scaler.pkl')
    
    return pipeline

def create_ontology():
    """Create learning style ontology"""
    onto = get_ontology("http://example.org/learning_ontology.owl")
    
    with onto:
        class ContentType(Thing): pass
        class LearningStyle(Thing): pass
        
        class prefersContentType(ObjectProperty):
            domain = [LearningStyle]
            range = [ContentType]
        
        # Define learning styles and their preferred content
        styles = {
            "Visual": ["VideoLectures", "Infographics", "Diagrams"],
            "Auditory": ["Podcasts", "AudioLectures", "DiscussionGroups"],
            "Kinesthetic": ["InteractiveLabs", "Simulations", "HandsOnActivities"],
            "ReadingWriting": ["Textbooks", "Articles", "WritingAssignments"]
        }
        
        for style, contents in styles.items():
            style_inst = LearningStyle(style)
            for content in contents:
                content_inst = ContentType(content)
                style_inst.prefersContentType.append(content_inst)
    
    # Save ontology
    os.makedirs('data', exist_ok=True)
    onto.save(file="data/learning_ontology.owl")

def main():
    """Main training workflow"""
    logger.info("Generating training data...")
    df = generate_training_data()
    
    logger.info("Preprocessing features...")
    X, y_dropout, y_style = preprocess_features(df)
    
    # Train models
    logger.info("Training dropout model...")
    dropout_model = train_dropout_model(X, y_dropout)
    
    logger.info("Training learning style model...")
    style_model = train_style_model(X, y_style)
    
    # Create ontology
    logger.info("Creating learning ontology...")
    create_ontology()
    
    # Save models
    os.makedirs('data', exist_ok=True)
    joblib.dump(dropout_model, 'data/dropout_predictor.pkl')
    joblib.dump(style_model, 'data/learning_style_predictor.pkl')
    
    # Evaluate models
    logger.info("\nDropout Model Evaluation:")
    y_pred = dropout_model.predict(X)
    print(classification_report(y_dropout, y_pred))
    print(f"Accuracy: {accuracy_score(y_dropout, y_pred):.2f}")
    
    logger.info("\nLearning Style Model Evaluation:")
    y_pred = style_model.predict(X)
    print(classification_report(y_style, y_pred))
    print(f"Accuracy: {accuracy_score(y_style, y_pred):.2f}")
    
    logger.info("Training complete! Models saved to data/ directory")

if __name__ == '__main__':
    main()