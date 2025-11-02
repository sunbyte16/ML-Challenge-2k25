"""
Feature engineering utilities for text and image data
Author: Sunil Sharma
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract features from catalog_content text"""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic text features"""
        df_features = df.copy()
        
        # Text length features
        df_features['text_length'] = df_features['catalog_content'].str.len()
        df_features['word_count'] = df_features['catalog_content'].str.split().str.len()
        df_features['sentence_count'] = df_features['catalog_content'].str.count(r'[.!?]+')
        
        # Price-related keywords
        price_keywords = ['price', 'cost', 'cheap', 'expensive', 'affordable', 'premium', 'budget']
        df_features['price_keywords_count'] = df_features['catalog_content'].str.lower().apply(
            lambda x: sum(1 for keyword in price_keywords if keyword in str(x))
        )
        
        # Brand indicators (uppercase words)
        df_features['uppercase_words'] = df_features['catalog_content'].apply(
            lambda x: len(re.findall(r'\b[A-Z]{2,}\b', str(x)))
        )
        
        # Numbers in text (could indicate specifications)
        df_features['number_count'] = df_features['catalog_content'].apply(
            lambda x: len(re.findall(r'\d+', str(x)))
        )
        
        # Special characters
        df_features['special_char_count'] = df_features['catalog_content'].apply(
            lambda x: len(re.findall(r'[^\w\s]', str(x)))
        )
        
        logger.info("Basic text features extracted")
        return df_features
    
    def extract_tfidf_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Extract TF-IDF features"""
        # Combine text for fitting
        all_text = pd.concat([train_df['catalog_content'], test_df['catalog_content']])
        
        # Fit TF-IDF
        self.tfidf_vectorizer.fit(all_text.fillna(''))
        
        # Transform
        train_tfidf = self.tfidf_vectorizer.transform(train_df['catalog_content'].fillna(''))
        test_tfidf = self.tfidf_vectorizer.transform(test_df['catalog_content'].fillna(''))
        
        # Convert to DataFrame
        feature_names = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=feature_names, index=train_df.index)
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=feature_names, index=test_df.index)
        
        logger.info(f"TF-IDF features extracted: {train_tfidf.shape[1]} features")
        return train_tfidf_df, test_tfidf_df


class ImageFeatureExtractor:
    """Extract features from image URLs and metadata"""
    
    def extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from image URLs"""
        df_features = df.copy()
        
        # Has image
        df_features['has_image'] = (df_features['image_link'].notna() & 
                                   (df_features['image_link'] != '')).astype(int)
        
        # URL length
        df_features['url_length'] = df_features['image_link'].str.len().fillna(0)
        
        # Image format from URL
        df_features['is_jpg'] = df_features['image_link'].str.contains('.jpg', case=False, na=False).astype(int)
        df_features['is_png'] = df_features['image_link'].str.contains('.png', case=False, na=False).astype(int)
        df_features['is_webp'] = df_features['image_link'].str.contains('.webp', case=False, na=False).astype(int)
        
        # Amazon images (might indicate certain quality/type)
        df_features['is_amazon_image'] = df_features['image_link'].str.contains('amazon', case=False, na=False).astype(int)
        
        # Image ID/hash length (might indicate image complexity)
        df_features['image_id_length'] = df_features['image_link'].apply(
            lambda x: len(re.findall(r'[A-Za-z0-9]{10,}', str(x))) if pd.notna(x) else 0
        )
        
        logger.info("Image URL features extracted")
        return df_features


class FeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self, max_tfidf_features: int = 5000):
        self.text_extractor = TextFeatureExtractor(max_tfidf_features)
        self.image_extractor = ImageFeatureExtractor()
        self.feature_columns = []
        
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Engineer all features"""
        logger.info("Starting feature engineering...")
        
        # Extract basic text features
        train_text_features = self.text_extractor.extract_basic_features(train_df)
        test_text_features = self.text_extractor.extract_basic_features(test_df)
        
        # Extract TF-IDF features
        train_tfidf, test_tfidf = self.text_extractor.extract_tfidf_features(train_df, test_df)
        
        # Extract image features
        train_image_features = self.image_extractor.extract_url_features(train_df)
        test_image_features = self.image_extractor.extract_url_features(test_df)
        
        # Combine all features
        train_features = pd.concat([
            train_text_features,
            train_tfidf,
            train_image_features[['has_image', 'url_length', 'is_jpg', 'is_png', 'is_webp', 
                                'is_amazon_image', 'image_id_length']]
        ], axis=1)
        
        test_features = pd.concat([
            test_text_features,
            test_tfidf,
            test_image_features[['has_image', 'url_length', 'is_jpg', 'is_png', 'is_webp', 
                               'is_amazon_image', 'image_id_length']]
        ], axis=1)
        
        # Store feature columns (excluding original columns and target)
        exclude_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
        self.feature_columns = [col for col in train_features.columns if col not in exclude_cols]
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_columns)}")
        return train_features, test_features
    
    def get_feature_importance_analysis(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature importance if model supports it"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                logger.warning("Model doesn't support feature importance analysis")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_and_preprocess_data
    
    train_df, test_df = load_and_preprocess_data()
    
    engineer = FeatureEngineer(max_tfidf_features=1000)  # Smaller for testing
    train_features, test_features = engineer.engineer_features(train_df, test_df)
    
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Feature columns: {len(engineer.feature_columns)}")