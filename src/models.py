"""
Machine Learning models for price prediction
Author: Sunil Sharma
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100


class ModelTrainer:
    """Train and evaluate different ML models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models to train"""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=RANDOM_STATE),
            'lasso': Lasso(random_state=RANDOM_STATE),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
        }
        return models
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_name: str) -> Dict[str, float]:
        """Train a single model and return cross-validation scores"""
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=CV_FOLDS, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Store the model
            self.models[model_name] = model
            
            # Calculate metrics
            train_pred = model.predict(X_train)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_smape = smape(y_train.values, train_pred)
            
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Update best model
            if cv_mae < self.best_score:
                self.best_score = cv_mae
                self.best_model = model_name
            
            results = {
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'train_mae': train_mae,
                'train_smape': train_smape
            }
            
            logger.info(f"{model_name} - CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {}
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Train all models and return results"""
        logger.info("Starting model training...")
        
        models = self.get_models()
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            result = self.train_model(model, X_train, y_train, model_name)
            if result:
                result['model'] = model_name
                results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_mae')
        
        logger.info(f"Best model: {self.best_model} with CV MAE: {self.best_score:.4f}")
        return results_df
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str = None) -> Any:
        """Perform hyperparameter tuning for the best model"""
        if model_name is None:
            model_name = self.best_model
        
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No hyperparameter grid for {model_name}")
            return self.models[model_name]
        
        base_model = self.get_models()[model_name]
        
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=CV_FOLDS,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Update the model
        self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def save_model(self, model_name: str, filename: str = None):
        """Save a trained model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        if filename is None:
            filename = f"{model_name}_model.joblib"
        
        filepath = MODELS_DIR / filename
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load a saved model"""
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(self, model_name: str, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return np.array([])
        
        predictions = self.models[model_name].predict(X_test)
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 0.01)
        
        return predictions


class EnsembleModel:
    """Ensemble multiple models for better predictions"""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0 for name in models.keys()}
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        total_weight = sum(self.weights.values())
        
        for model_name, model in self.models.items():
            pred = model.predict(X_test)
            weight = self.weights.get(model_name, 1.0) / total_weight
            predictions.append(pred * weight)
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Ensure predictions are positive
        ensemble_pred = np.maximum(ensemble_pred, 0.01)
        
        return ensemble_pred


if __name__ == "__main__":
    # Test the model trainer
    from data_loader import load_and_preprocess_data
    from feature_engineering import FeatureEngineer
    
    # Load data
    train_df, test_df = load_and_preprocess_data()
    
    # Engineer features
    engineer = FeatureEngineer(max_tfidf_features=1000)  # Smaller for testing
    train_features, test_features = engineer.engineer_features(train_df, test_df)
    
    # Prepare training data
    X_train = train_features[engineer.feature_columns]
    y_train = train_features['price']
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train)
    
    print("Model training results:")
    print(results)