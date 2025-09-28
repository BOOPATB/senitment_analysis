"""
Main sentiment analysis module with multiple model implementations.
Includes traditional ML models and transformer-based approaches.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from preprocessing import TextPreprocessor, load_sample_data, create_train_test_split


class TraditionalSentimentAnalyzer:
    """Traditional ML-based sentiment analyzer using scikit-learn."""
    
    def __init__(self, model_type: str = 'logistic_regression', 
                 vectorizer_type: str = 'tfidf', max_features: int = 5000):
        """
        Initialize the traditional sentiment analyzer.
        
        Args:
            model_type: Type of ML model ('logistic_regression', 'svm', 'naive_bayes', 'random_forest')
            vectorizer_type: Type of vectorizer ('tfidf', 'count')
            max_features: Maximum number of features for vectorization
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
        self.label_mapping = None
        
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the ML pipeline with vectorizer and model."""
        # Choose vectorizer
        if self.vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=self.max_features, 
                                       ngram_range=(1, 2), stop_words='english')
        else:
            vectorizer = CountVectorizer(max_features=self.max_features, 
                                       ngram_range=(1, 2), stop_words='english')
        
        # Choose model
        if self.model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            model = SVC(kernel='linear', probability=True, random_state=42)
        elif self.model_type == 'naive_bayes':
            model = MultinomialNB()
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
    
    def train(self, X_train: pd.Series, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the sentiment analysis model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            
        Returns:
            Training metrics
        """
        # Preprocess text data
        X_train_processed = X_train.apply(self.preprocessor.preprocess)
        
        # Create label mapping
        unique_labels = sorted(y_train.unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Convert labels to numeric
        y_train_numeric = y_train.map(self.label_mapping)
        
        # Fit the pipeline
        self.pipeline.fit(X_train_processed, y_train_numeric)
        
        # Calculate training accuracy
        y_train_pred = self.pipeline.predict(X_train_processed)
        train_accuracy = accuracy_score(y_train_numeric, y_train_pred)
        
        return {
            'train_accuracy': train_accuracy,
            'label_mapping': self.label_mapping,
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type
        }
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predicted sentiments
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Make predictions
        predictions_numeric = self.pipeline.predict(processed_texts)
        
        # Convert back to labels
        reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        predictions = [reverse_mapping[pred] for pred in predictions_numeric]
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment probabilities for given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        return self.pipeline.predict_proba(processed_texts)
    
    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X_test.tolist())
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'pipeline': self.pipeline,
            'label_mapping': self.label_mapping,
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.label_mapping = model_data['label_mapping']
        self.model_type = model_data['model_type']
        self.vectorizer_type = model_data['vectorizer_type']
        self.preprocessor = model_data['preprocessor']


class TransformerSentimentAnalyzer:
    """Transformer-based sentiment analyzer using Hugging Face models."""
    
    def __init__(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """
        Initialize the transformer sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained model from Hugging Face
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained transformer model."""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to default sentiment analysis model...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", return_all_scores=True)
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predicted sentiments
        """
        if self.sentiment_pipeline is None:
            raise ValueError("Model not loaded.")
        
        results = []
        for text in texts:
            # Get predictions
            predictions = self.sentiment_pipeline(text)[0]
            
            # Find the label with highest score
            best_prediction = max(predictions, key=lambda x: x['score'])
            label = best_prediction['label']
            
            # Map to standard labels if needed
            if label in self.label_mapping:
                label = self.label_mapping[label]
            
            results.append(label.lower())
        
        return results
    
    def predict_with_scores(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment with confidence scores.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of prediction dictionaries with scores
        """
        if self.sentiment_pipeline is None:
            raise ValueError("Model not loaded.")
        
        results = []
        for text in texts:
            predictions = self.sentiment_pipeline(text)[0]
            
            # Process predictions
            processed_preds = []
            for pred in predictions:
                label = pred['label']
                if label in self.label_mapping:
                    label = self.label_mapping[label]
                
                processed_preds.append({
                    'label': label.lower(),
                    'score': pred['score']
                })
            
            # Sort by score
            processed_preds.sort(key=lambda x: x['score'], reverse=True)
            results.append(processed_preds)
        
        return results


class SentimentAnalysisComparator:
    """Compare different sentiment analysis approaches."""
    
    def __init__(self):
        """Initialize the comparator."""
        self.models = {}
        self.results = {}
    
    def add_traditional_model(self, name: str, model_type: str, 
                            vectorizer_type: str = 'tfidf'):
        """Add a traditional ML model for comparison."""
        self.models[name] = TraditionalSentimentAnalyzer(
            model_type=model_type, 
            vectorizer_type=vectorizer_type
        )
    
    def add_transformer_model(self, name: str, model_name: str):
        """Add a transformer model for comparison."""
        self.models[name] = TransformerSentimentAnalyzer(model_name=model_name)
    
    def compare_models(self, X_train: pd.Series, y_train: pd.Series,
                      X_test: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare all models and return results.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Comparison results dataframe
        """
        comparison_results = []
        
        for name, model in self.models.items():
            print(f"Training and evaluating {name}...")
            
            try:
                if isinstance(model, TraditionalSentimentAnalyzer):
                    # Train traditional model
                    train_metrics = model.train(X_train, y_train)
                    eval_metrics = model.evaluate(X_test, y_test)
                    
                    accuracy = eval_metrics['accuracy']
                    
                elif isinstance(model, TransformerSentimentAnalyzer):
                    # Evaluate transformer model (pre-trained)
                    predictions = model.predict(X_test.tolist())
                    accuracy = accuracy_score(y_test, predictions)
                
                comparison_results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Model Type': type(model).__name__
                })
                
                # Store detailed results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy
                }
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                comparison_results.append({
                    'Model': name,
                    'Accuracy': None,
                    'Model Type': type(model).__name__
                })
        
        return pd.DataFrame(comparison_results).sort_values('Accuracy', ascending=False)
    
    def plot_comparison(self, results_df: pd.DataFrame):
        """Plot model comparison results."""
        plt.figure(figsize=(10, 6))
        
        # Remove None values
        results_clean = results_df.dropna(subset=['Accuracy'])
        
        sns.barplot(data=results_clean, x='Model', y='Accuracy')
        plt.title('Sentiment Analysis Model Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(results_clean['Accuracy']):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate sentiment analysis capabilities."""
    print("=== Sentiment Analysis Project Demo ===\n")
    
    # Load sample data
    print("1. Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    print(df.head())
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df, 'text', 'sentiment')
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_processed, 'text_processed', 'sentiment', test_size=0.3, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize comparator
    print("\n4. Setting up models for comparison...")
    comparator = SentimentAnalysisComparator()
    
    # Add traditional models
    comparator.add_traditional_model("Logistic Regression", "logistic_regression")
    comparator.add_traditional_model("SVM", "svm")
    comparator.add_traditional_model("Naive Bayes", "naive_bayes")
    
    # Add transformer model (will use default if specific model not available)
    comparator.add_transformer_model("Transformer", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Compare models
    print("\n5. Comparing models...")
    results = comparator.compare_models(X_train, y_train, X_test, y_test)
    
    print("\nComparison Results:")
    print(results)
    
    # Test individual predictions
    print("\n6. Testing individual predictions...")
    test_texts = [
        "I absolutely love this product!",
        "This is the worst thing ever.",
        "It's okay, nothing special.",
        "Amazing quality and great service!",
        "Terrible experience, very disappointed."
    ]
    
    # Get the best performing traditional model
    best_traditional = None
    best_accuracy = 0
    for name, result in comparator.results.items():
        if isinstance(result['model'], TraditionalSentimentAnalyzer) and result['accuracy'] > best_accuracy:
            best_traditional = result['model']
            best_accuracy = result['accuracy']
    
    if best_traditional:
        print(f"\nPredictions from best traditional model (Accuracy: {best_accuracy:.3f}):")
        predictions = best_traditional.predict(test_texts)
        for text, pred in zip(test_texts, predictions):
            print(f"Text: '{text}' → Sentiment: {pred}")


if __name__ == "__main__":
    main()
