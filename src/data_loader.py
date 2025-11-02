"""
Data loading and preprocessing utilities
Author: Sunil Sharma
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from config import TRAIN_FILE, TEST_FILE, SAMPLE_TEST_FILE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and basic preprocessing class"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.sample_test_data = None
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training data"""
        try:
            self.train_data = pd.read_csv(TRAIN_FILE)
            logger.info(f"Training data loaded: {self.train_data.shape}")
            return self.train_data
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data"""
        try:
            self.test_data = pd.read_csv(TEST_FILE)
            logger.info(f"Test data loaded: {self.test_data.shape}")
            return self.test_data
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def load_sample_test_data(self) -> pd.DataFrame:
        """Load sample test data"""
        try:
            self.sample_test_data = pd.read_csv(SAMPLE_TEST_FILE)
            logger.info(f"Sample test data loaded: {self.sample_test_data.shape}")
            return self.sample_test_data
        except Exception as e:
            logger.error(f"Error loading sample test data: {e}")
            raise
    
    def get_data_info(self) -> dict:
        """Get basic information about the datasets"""
        info = {}
        
        if self.train_data is not None:
            info['train'] = {
                'shape': self.train_data.shape,
                'columns': list(self.train_data.columns),
                'missing_values': self.train_data.isnull().sum().to_dict(),
                'price_stats': self.train_data['price'].describe().to_dict() if 'price' in self.train_data.columns else None
            }
        
        if self.test_data is not None:
            info['test'] = {
                'shape': self.test_data.shape,
                'columns': list(self.test_data.columns),
                'missing_values': self.test_data.isnull().sum().to_dict()
            }
        
        return info
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing steps"""
        df_processed = df.copy()
        
        # Handle missing values in catalog_content
        df_processed['catalog_content'] = df_processed['catalog_content'].fillna('')
        
        # Handle missing image links
        df_processed['image_link'] = df_processed['image_link'].fillna('')
        
        # Add text length feature
        df_processed['text_length'] = df_processed['catalog_content'].str.len()
        
        # Add has_image feature
        df_processed['has_image'] = (df_processed['image_link'] != '').astype(int)
        
        logger.info("Basic preprocessing completed")
        return df_processed


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load and preprocess all data"""
    loader = DataLoader()
    
    # Load data
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    
    # Basic preprocessing
    train_processed = loader.basic_preprocessing(train_df)
    test_processed = loader.basic_preprocessing(test_df)
    
    # Print data info
    info = loader.get_data_info()
    logger.info(f"Data info: {info}")
    
    return train_processed, test_processed


if __name__ == "__main__":
    # Test the data loader
    train_df, test_df = load_and_preprocess_data()
    print("Data loading completed successfully!")