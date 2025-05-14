from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class ModelHandler:
    """Wrapper class for model loading and prediction"""
    
    class DummyModel:
        """Dummy model class for testing"""
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])  # [P(Not Dropout), P(Dropout)]
        def predict(self, X):
            return np.array([1])  # Default to Auditory learning style
    
    class DummyScaler:
        """Dummy scaler class for testing"""
        def transform(self, X):
            return X  # No scaling
    
    def __init__(self):
        self.models = {
            'dropout': None,
            'style': None,
            'dropout_scaler': None,
            'style_scaler': None
        }
        self.load_models()

    def load_models(self):
        """Load all required models and scalers with proper error handling"""
        try:
            logger.info("Attempting to load models...")
            base_path = os.path.join(os.path.dirname(__file__), 'data')
            
            # Model files configuration
            model_files = {
                'dropout': 'dropout_model.pkl',
                'style': 'style_model.pkl',
                'dropout_scaler': 'dropout_scaler.pkl',
                'style_scaler': 'style_scaler.pkl'
            }

            # Load each model file
            for model_name, filename in model_files.items():
                file_path = os.path.join(base_path, filename)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Model file not found: {file_path}")
                
                self.models[model_name] = joblib.load(file_path)
                logger.info(f"Successfully loaded {model_name} from {filename}")

            logger.info("All models and scalers loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            # Create dummy models for testing if real models fail to load
            self._create_dummy_models()
            return False

    def _create_dummy_models(self):
        """Create dummy models if real models fail to load"""
        logger.warning("Creating dummy models for testing purposes")
        
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.7, 0.3]])  # [P(Not Dropout), P(Dropout)]
            def predict(self, X):
                return np.array([1])  # Default to Auditory learning style
        
        class DummyScaler:
            def transform(self, X):
                return X  # No scaling
        
        self.models['dropout'] = DummyModel()
        self.models['style'] = DummyModel()
        self.models['dropout_scaler'] = DummyScaler()
        self.models['style_scaler'] = DummyScaler()

    def prepare_features(self, input_data):
        """Prepare and validate input features"""
        try:
            # Gender and education mappings (must match training)
            gender_map = {'male': 1, 'female': 0, 'other': 0.5}
            education_map = {
                'high school': 0,
                'bachelor\'s': 1,
                'master\'s': 2,
                'phd': 3
            }
            
            # Convert all features to match training data format
            features = np.array([
                float(input_data['age']),
                gender_map.get(input_data['gender'].lower(), 0.5),
                education_map.get(input_data['education'].lower(), 0),
                float(input_data['time_spent']),
                float(input_data['quiz_attempts']),
                float(input_data['quiz_scores']),
                float(input_data['forum_participation']),
                float(input_data['assignment_completion']),
                float(input_data['feedback_score'])
            ]).reshape(1, -1)
            
            # Scale features using the appropriate scaler
            scaled_features = self.models['dropout_scaler'].transform(features)
            return scaled_features
            
        except (KeyError, ValueError) as e:
            logger.error(f"Feature preparation error: {str(e)}")
            raise ValueError(f"Invalid input data: {str(e)}")

    def _classify_risk(self, probability):
        """Classify dropout risk based on probability"""
        if probability > 0.7:
            return "High Risk"
        elif probability > 0.4:
            return "Medium Risk"
        return "Low Risk"

    def _get_style_label(self, style_code):
        """Convert numeric style code to label"""
        styles = ["Visual", "Auditory", "Kinesthetic", "ReadingWriting"]
        return styles[style_code] if 0 <= style_code < len(styles) else "Unknown"

    def _get_recommendations(self, risk_level, learning_style):
        """Generate personalized recommendations"""
        risk_based = {
            "High Risk": [
                "One-on-one tutoring sessions",
                "Simplified study materials", 
                "Weekly progress check-ins"
            ],
            "Medium Risk": [
                "Interactive quizzes",
                "Study group meetings",
                "Video explanations"
            ],
            "Low Risk": [
                "Advanced reading materials",
                "Challenge problems",
                "Research opportunities"
            ]
        }
        
        style_based = {
            "Visual": ["Video lectures", "Infographics", "Diagrams"],
            "Auditory": ["Podcasts", "Audio recordings", "Discussion groups"],
            "Kinesthetic": ["Hands-on labs", "Simulations", "Interactive exercises"],
            "ReadingWriting": ["Textbooks", "Articles", "Writing assignments"]
        }
        
        return {
            "risk_based": risk_based.get(risk_level, []),
            "style_based": style_based.get(learning_style, [])
        }

# Initialize model handler
model_handler = ModelHandler()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint for service health check"""
    status = 'ready' if all(model_handler.models.values()) else 'error'
    return jsonify({
        'status': status,
        'model_status': {
            'dropout': bool(model_handler.models['dropout']),
            'style': bool(model_handler.models['style']),
            'dropout_scaler': bool(model_handler.models['dropout_scaler']),
            'style_scaler': bool(model_handler.models['style_scaler'])
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate model initialization
        if not all(model_handler.models.values()):
            return jsonify({
                "error": "Model initialization failed",
                "message": "Some models failed to load"
            }), 503

        # Generate session ID
        session_id = str(uuid.uuid4())
        data = request.json
        
        # Validate input
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data received"}), 400
            
        required_fields = [
            'age', 'gender', 'education', 'time_spent',
            'quiz_attempts', 'quiz_scores', 'forum_participation',
            'assignment_completion', 'feedback_score'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Prepare features
        features = model_handler.prepare_features(data)
        logger.debug(f"Processed features: {features}")

        # Make predictions
        dropout_prob = model_handler.models['dropout'].predict_proba(features)[0][1]
        dropout_risk = model_handler._classify_risk(dropout_prob)
        
        style_pred = model_handler.models['style'].predict(features)[0]
        style_label = model_handler._get_style_label(style_pred)

        # Prepare response
        response = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": dropout_risk,
            "confidence": float(dropout_prob),
            "learning_style": style_label,
            "recommendations": model_handler._get_recommendations(dropout_risk, style_label),
            "model_status": "live" if not isinstance(model_handler.models['dropout'], 
                            model_handler.__class__.DummyModel) else "debug"
        }
        
        logger.info(f"Prediction successful for session {session_id}")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "error": "Invalid input data",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Endpoint for collecting feedback"""
    try:
        feedback_data = request.json
        if not feedback_data:
            return jsonify({"error": "No feedback data received"}), 400
            
        logger.info(f"Feedback received: {feedback_data}")
        return jsonify({
            "status": "success",
            "message": "Feedback recorded",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({
            "error": "Invalid feedback data",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)