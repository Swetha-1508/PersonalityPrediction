
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from owlready2 import *
import joblib

# Download NLTK data
nltk.download('wordnet', quiet=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ====================== 1. Data Loading & Preprocessing ======================
def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    
    print("\nOriginal Data Columns:")
    print(df.columns)
    print("\nData Types:")
    print(df.dtypes)
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns.drop(['Course_Name', 'Student_ID'], errors='ignore')
    
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# ====================== 2. SMOTE Analysis & Class Balancing ======================
def analyze_class_balance(y, title="Class Distribution"):
    """Visualize class distribution."""
    counts = Counter(y)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# ====================== 3. BERT Embeddings ======================
def get_bert_embeddings(texts, batch_size=16):
    """Generate BERT embeddings for text data."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT Embeddings"):
        batch = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    
    return np.vstack(embeddings)

# ====================== 4. Model Training & Evaluation ======================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance with multiple metrics."""
    y_pred = model.predict(X_test)
    
    print(f"\n=== {model_name} Evaluation ===")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    
    # ROC-AUC for binary classification
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {roc_auc:.3f}")

def train_model(X, y, model, model_name, target_name, apply_smote=True):
    """Train and evaluate a model with optional SMOTE."""
    # Analyze class distribution
    print(f"\n=== Class Distribution for {target_name} ===")
    analyze_class_balance(y)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    if apply_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Final training
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    evaluate_model(pipeline, X_test, y_test, model_name)
    
    return pipeline

# ====================== 5. Ontology Update ======================
def update_ontology(output_path):
    """Update the learning ontology with new styles and content types."""
    try:
        # Try to load existing ontology or create new
        onto = get_ontology("learning_ontology.owl").load()
    except FileNotFoundError:
        onto = get_ontology("http://example.org/learning_ontology.owl")
        with onto:
            # Define classes if creating new ontology
            class ContentType(Thing):
                pass
            
            class LearningStyle(Thing):
                pass
            
            class prefersContentType(ObjectProperty):
                domain = [LearningStyle]
                range = [ContentType]
    
    # Ensure classes exist in loaded ontology
    with onto:
        if not hasattr(onto, 'ContentType'):
            class ContentType(Thing):
                pass
        
        if not hasattr(onto, 'LearningStyle'):
            class LearningStyle(Thing):
                pass
        
        if not hasattr(onto, 'prefersContentType'):
            class prefersContentType(ObjectProperty):
                domain = [LearningStyle]
                range = [ContentType]
    
    # Get references to the classes
    ContentType = onto.ContentType
    LearningStyle = onto.LearningStyle
    
    # Define learning styles and content mappings
    learning_styles = {
        "Visual": ["VideoLecture", "Infographic"],
        "Auditory": ["AudioLecture", "Podcast"],
        "Kinesthetic": ["InteractiveSimulation", "HandsOnLab"],
        "ReadingWriting": ["TextBook", "Article"]
    }
    
    # Add all styles and content types
    for style, content_types in learning_styles.items():
        # Get or create the learning style
        style_inst = onto.search_one(iri=f"*{style}")
        if not style_inst:
            style_inst = LearningStyle(style)
        
        # Add each content type
        for content in content_types:
            content_inst = onto.search_one(iri=f"*{content}")
            if not content_inst:
                content_inst = ContentType(content)
            
            # Add relationship if not exists
            if content_inst not in style_inst.prefersContentType:
                style_inst.prefersContentType.append(content_inst)
    
    # Save the ontology
    onto.save(file=output_path, format="rdfxml")
    print(f"Ontology successfully updated at {output_path}")

# ====================== 6. Main Execution ======================
if __name__ == "__main__":
    print("Starting Personalized Learning System...")
    
    # 1. Load and preprocess data
    df, label_encoders = load_and_preprocess_data("personalized_learning_dataset.csv")
    
    # 2. Generate BERT embeddings for course names
    print("\nGenerating BERT embeddings for course names...")
    text_embeddings = get_bert_embeddings(df['Course_Name'].fillna("").astype(str))
    text_features = pd.DataFrame(text_embeddings, columns=[f'bert_{i}' for i in range(text_embeddings.shape[1])])
    df = pd.concat([df.drop(['Course_Name', 'Student_ID'], axis=1), text_features], axis=1)
    
    # 3. Train Dropout Prediction Model (MLP)
    print("\nTraining Dropout Prediction Model...")
    X_dropout = df.drop('Dropout_Likelihood', axis=1)
    y_dropout = df['Dropout_Likelihood']
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    mlp_pipeline = train_model(X_dropout, y_dropout, mlp, "MLP - Dropout Prediction", "Dropout_Likelihood", apply_smote=True)
    
    # 4. Train Learning Style Model (Random Forest)
    print("\nTraining Learning Style Model...")
    X_style = df.drop('Learning_Style', axis=1)
    y_style = df['Learning_Style']
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        random_state=42
    )
    rf_pipeline = train_model(X_style, y_style, rf, "Random Forest - Learning Style", "Learning_Style", apply_smote=False)
    
    # 5. Update Ontology
    print("\nUpdating Learning Ontology...")
    update_ontology("learning_ontology_updated.owl")
    
    # 6. Save Models
    joblib.dump(mlp_pipeline, 'dropout_predictor.pkl')
    joblib.dump(rf_pipeline, 'learning_style_predictor.pkl')
    print("\nModels saved successfully!")
    print("\nProcess completed successfully!")