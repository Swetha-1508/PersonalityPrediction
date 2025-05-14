import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, 
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib
import os
import matplotlib.pyplot as plt
from owlready2 import *
import logging
from collections import Counter
import sys

# Configure logging to show output in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_realistic_data(n_samples=20000):
    """Generate synthetic data with meaningful relationships"""
    logger.info("Starting data generation...")
    try:
        np.random.seed(42)
        
        # Base features
        data = {
            'age': np.clip(np.random.normal(22, 5, n_samples), 16, 40),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                        n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'time_spent': np.clip(np.random.normal(15, 5, n_samples), 5, 30),
            'quiz_attempts': np.random.poisson(5, n_samples),
            'quiz_scores': np.clip(np.random.normal(75, 15, n_samples), 0, 100),
            'forum_participation': np.random.poisson(3, n_samples),
            'assignment_completion': np.clip(np.random.normal(80, 15, n_samples), 0, 100),
            'feedback_score': np.clip(np.random.normal(7, 1.5, n_samples), 1, 10),
        }
        
        df = pd.DataFrame(data)
        
        # Create meaningful dropout relationships (10% dropout rate)
        dropout_prob = (
            0.4 * (df['quiz_scores'] < 50).astype(float) + 
            0.3 * (df['assignment_completion'] < 60).astype(float) +
            0.2 * (df['forum_participation'] < 2).astype(float) +
            np.random.normal(0, 0.1, n_samples)
        )
        df['Dropout'] = (dropout_prob > 0.5).astype(int)
        
        # Create meaningful learning style relationships
        def get_learning_style(row):
            if row['forum_participation'] > 4 and row['quiz_scores'] < 70:
                return 1  # Auditory
            elif row['quiz_attempts'] > 6 and row['quiz_scores'] > 80:
                return 3  # Reading/Writing
            elif row['time_spent'] > 20 and row['quiz_scores'] < 60:
                return 2  # Kinesthetic
            else:
                return 0  # Visual
            
        df['Learning_Style'] = df.apply(get_learning_style, axis=1)
        
        # Ensure minimum samples per class
        min_samples = n_samples // 20
        for style in range(4):
            if sum(df['Learning_Style'] == style) < min_samples:
                indices = np.random.choice(df.index, min_samples, replace=False)
                df.loc[indices, 'Learning_Style'] = style
                
        logger.info("Data generation completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error in data generation: {str(e)}")
        raise

def preprocess_features(df):
    """Convert categorical features to numerical"""
    try:
        gender_map = {'Male': 1, 'Female': 0, 'Other': 0.5}
        education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        
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
    except Exception as e:
        logger.error(f"Error in feature preprocessing: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, name=""):
    """Comprehensive model evaluation"""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"{name} Model Evaluation (TEST SET)")
        logger.info('='*50)
        
        y_pred = model.predict(X_test)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
        
        logger.info(f"\nBalanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}")
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{name} Confusion Matrix")
        os.makedirs('data', exist_ok=True)
        plt.savefig(f'data/{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def train_dropout_model(X_train, X_test, y_train, y_test):
    """Train dropout prediction model"""
    try:
        logger.info("\nTraining Dropout Model...")
        
        # Create separate scaler for the dropout model
        dropout_scaler = StandardScaler()
        X_train_scaled = dropout_scaler.fit_transform(X_train)
        X_test_scaled = dropout_scaler.transform(X_test)
        
        pipeline = ImbPipeline([ 
            ('smote', SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced_subsample',
                max_depth=10,
                random_state=42
            ))
        ])
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5, scoring='balanced_accuracy')
        logger.info(f"CV Balanced Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Train final model
        pipeline.fit(X_train_scaled, y_train)
        evaluate_model(pipeline, X_test_scaled, y_test, "Dropout Prediction")
        
        # Save the scaler separately
        joblib.dump(dropout_scaler, 'data/dropout_scaler.pkl')
        
        return pipeline
    except Exception as e:
        logger.error(f"Error in dropout model training: {str(e)}")
        raise

def train_style_model(X_train, X_test, y_train, y_test):
    """Train learning style classification model"""
    try:
        logger.info("\nTraining Learning Style Model...")
        
        # Create separate scaler for the style model
        style_scaler = StandardScaler()
        X_train_scaled = style_scaler.fit_transform(X_train)
        X_test_scaled = style_scaler.transform(X_test)
        
        pipeline = ImbPipeline([
            ('classifier', RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5, scoring='balanced_accuracy')
        logger.info(f"CV Balanced Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Train final model
        pipeline.fit(X_train_scaled, y_train)
        evaluate_model(pipeline, X_test_scaled, y_test, "Learning Style")
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        importances = pipeline.named_steps['classifier'].feature_importances_
        features = X_train.columns
        indices = np.argsort(importances)[::-1]
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.savefig('data/learning_style_feature_importances.png')
        plt.close()
        
        # Save the scaler separately
        joblib.dump(style_scaler, 'data/style_scaler.pkl')
        
        return pipeline
    except Exception as e:
        logger.error(f"Error in style model training: {str(e)}")
        raise

def create_ontology():
    """Create learning style ontology"""
    try:
        logger.info("Creating learning ontology...")
        onto = get_ontology("http://example.org/learning_ontology.owl")
        
        with onto:
            class ContentType(Thing): pass
            class LearningStyle(Thing): pass
            
            class prefersContentType(ObjectProperty):
                domain = [LearningStyle]
                range = [ContentType]
            
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
        
        os.makedirs('data', exist_ok=True)
        onto.save(file="data/learning_ontology.owl")
        logger.info("Ontology created successfully")
    except Exception as e:
        logger.error(f"Error creating ontology: {str(e)}")
        raise

def main():
    """Main training workflow"""
    try:
        logger.info("Starting model training pipeline...")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        df = generate_realistic_data(20000)
        logger.info("\nClass Distribution:")
        logger.info(f"Dropout: {Counter(df['Dropout'])}")
        logger.info(f"Learning Style: {Counter(df['Learning_Style'])}")
        
        X, y_dropout, y_style = preprocess_features(df)
        X_train, X_test, y_train_dropout, y_test_dropout = train_test_split(
            X, y_dropout, test_size=0.2, stratify=y_dropout, random_state=42
        )
        X_train_style, X_test_style, y_train_style, y_test_style = train_test_split(
            X, y_style, test_size=0.2, stratify=y_style, random_state=42
        )
        
        dropout_model = train_dropout_model(X_train, X_test, y_train_dropout, y_test_dropout)
        style_model = train_style_model(X_train_style, X_test_style, y_train_style, y_test_style)
        
        # Save models
        joblib.dump(dropout_model, 'data/dropout_model.pkl')
        joblib.dump(style_model, 'data/style_model.pkl')
        
        create_ontology()
        logger.info("Pipeline completed. All model files saved in 'data' directory.")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()tra