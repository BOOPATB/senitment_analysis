"""
Sentiment Analysis Project

A comprehensive machine learning project for sentiment analysis using both 
traditional ML approaches and modern transformer models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .preprocessing import TextPreprocessor, load_sample_data, create_train_test_split
from .sentiment_analyzer import (
    TraditionalSentimentAnalyzer,
    TransformerSentimentAnalyzer,
    SentimentAnalysisComparator
)

__all__ = [
    "TextPreprocessor",
    "load_sample_data", 
    "create_train_test_split",
    "TraditionalSentimentAnalyzer",
    "TransformerSentimentAnalyzer", 
    "SentimentAnalysisComparator"
]
