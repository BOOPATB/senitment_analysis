"""
Data preprocessing utilities for sentiment analysis.
This module contains functions for text cleaning, tokenization, and data preparation.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """A comprehensive text preprocessing class for sentiment analysis."""
    
    def __init__(self, remove_stopwords: bool = True, use_stemming: bool = False, 
                 use_lemmatization: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.use_stemming:
            self.stemmer = PorterStemmer()
        
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media data)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespaces and line breaks
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers (optional - you might want to keep them)
        # text = re.sub(r'\d+', '', text)
        
        # Remove punctuation but keep emoticons
        emoticons = re.findall(r'[:;=8][\-o\*\']?[\)\]\(\[\dDpP/\:\}\{@\|\\]', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        
        # Add emoticons back
        for emoticon in emoticons:
            text += f" {emoticon}"
        
        return text
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize text and apply stemming/lemmatization if specified.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if specified
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization if specified
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove empty tokens and single characters
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_process(cleaned_text)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                           target_column: str = None) -> pd.DataFrame:
        """
        Preprocess an entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            target_column: Name of the target column (optional)
            
        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Apply preprocessing to text column
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(self.preprocess)
        
        # Remove empty processed texts
        df_processed = df_processed[df_processed[f'{text_column}_processed'].str.len() > 0]
        
        return df_processed


def load_sample_data() -> pd.DataFrame:
    """
    Create a sample dataset for testing sentiment analysis models.
    
    Returns:
        Sample dataframe with text and sentiment columns
    """
    sample_data = {
        'text': [
            "I love this movie! It's absolutely fantastic.",
            "This product is terrible. I hate it so much.",
            "The service was okay, nothing special.",
            "Amazing experience! Highly recommend to everyone.",
            "Worst purchase ever. Complete waste of money.",
            "It's alright, could be better but not bad.",
            "Fantastic quality! Will definitely buy again.",
            "Very disappointing. Expected much better.",
            "Good value for money. Satisfied with the purchase.",
            "Exceptional service! Staff was very helpful.",
            "Poor quality. Broke after one day.",
            "Love the design and functionality. Great product!",
            "Not worth the price. Too expensive for what it offers.",
            "Perfect! Exactly what I was looking for.",
            "Mediocre at best. Nothing impressive about it."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'positive', 'positive',
            'negative', 'positive', 'negative', 'positive', 'neutral'
        ]
    }
    
    return pd.DataFrame(sample_data)


def create_train_test_split(df: pd.DataFrame, text_column: str, 
                          target_column: str, test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                         pd.Series, pd.Series]:
    """
    Create train-test split for the dataset.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        target_column: Name of the target column
        test_size: Proportion of test set
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[text_column]
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, stratify=y)


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    # Test with sample text
    sample_text = "I really LOVE this amazing product!!! 😊 It's the best thing ever! http://example.com"
    processed_text = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed_text}")
    
    # Test with sample dataframe
    df = load_sample_data()
    df_processed = preprocessor.preprocess_dataframe(df, 'text', 'sentiment')
    print("\nSample processed data:")
    print(df_processed[['text', 'text_processed', 'sentiment']].head())
