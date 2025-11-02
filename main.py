"""
Main script for ML Challenge 2025 - Smart Product Pricing
Author: Sunil Sharma
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Import custom modules
from src.data_loader import load_and_preprocess_data
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer, EnsembleModel
from config import OUTPUT_DIR, OUTPUT_FILE, RANDOM_STATE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training and prediction pipeline"""
    logger.info("🚀 Starting ML Challenge 2025 - Smart Product Pricing")
    logger.info(f"Random state: {RANDOM_STATE}")
    
    try:
        # 1. Load and preprocess data
        logger.info("📊 Loading and preprocessing data...")
        train_df, test_df = load_and_preprocess_data()
        
        # 2. Feature engineering
        logger.info("🔧 Engineering features...")
        engineer = FeatureEngineer(max_tfidf_features=args.max_features)
        train_features, test_features = engineer.engineer_features(train_df, test_df)
        
        # 3. Prepare training data
        X_train = train_features[engineer.feature_columns]
        y_train = train_features['price']
        X_test = test_features[engineer.feature_columns]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Target statistics: Mean={y_train.mean():.2f}, Std={y_train.std():.2f}")
        
        # 4. Train models
        logger.info("🤖 Training models...")
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, y_train)
        
        # Display results
        print("\n📈 Model Performance Results:")
        print("=" * 60)
        print(results.to_string(index=False))
        print("=" * 60)
        
        # 5. Hyperparameter tuning (optional)
        if args.tune_hyperparameters:
            logger.info("⚙️ Performing hyperparameter tuning...")
            best_model = trainer.hyperparameter_tuning(X_train, y_train)
        else:
            best_model = trainer.models[trainer.best_model]
        
        # 6. Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            logger.info("📊 Analyzing feature importance...")
            importance_df = engineer.get_feature_importance_analysis(
                best_model, engineer.feature_columns
            )
            if not importance_df.empty:
                print("\n🔍 Top 10 Most Important Features:")
                print("=" * 40)
                print(importance_df.head(10).to_string(index=False))
                print("=" * 40)
        
        # 7. Make predictions
        logger.info("🎯 Making predictions on test data...")
        if args.use_ensemble:
            # Create ensemble of top 3 models
            top_models = results.head(3)['model'].tolist()
            ensemble_models = {name: trainer.models[name] for name in top_models}
            ensemble = EnsembleModel(ensemble_models)
            predictions = ensemble.predict(X_test)
            logger.info(f"Using ensemble of: {top_models}")
        else:
            predictions = trainer.predict(trainer.best_model, X_test)
            logger.info(f"Using single model: {trainer.best_model}")
        
        # 8. Create submission file
        logger.info("📝 Creating submission file...")
        submission = pd.DataFrame({
            'sample_id': test_features['sample_id'],
            'price': predictions
        })
        
        # Ensure all predictions are positive
        submission['price'] = np.maximum(submission['price'], 0.01)
        
        # Save submission
        output_path = OUTPUT_DIR / OUTPUT_FILE
        submission.to_csv(output_path, index=False)
        
        logger.info(f"✅ Submission saved to: {output_path}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Price statistics: Min={submission['price'].min():.2f}, "
                   f"Max={submission['price'].max():.2f}, "
                   f"Mean={submission['price'].mean():.2f}")
        
        # 9. Save best model
        if args.save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"best_model_{timestamp}.joblib"
            trainer.save_model(trainer.best_model, model_filename)
        
        logger.info("🎉 Pipeline completed successfully!")
        
        return submission, results
        
    except Exception as e:
        logger.error(f"❌ Error in main pipeline: {e}")
        raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ML Challenge 2025 - Smart Product Pricing')
    
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum number of TF-IDF features (default: 5000)')
    
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning on best model')
    
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Use ensemble of top models instead of single best model')
    
    parser.add_argument('--save-model', action='store_true',
                       help='Save the best trained model')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced features for quick testing')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Adjust parameters for quick testing
    if args.quick_test:
        args.max_features = 1000
        logger.info("🚀 Running in quick test mode with reduced features")
    
    # Run main pipeline
    submission, results = main(args)
    
    print(f"\n🎯 Final submission preview:")
    print(submission.head(10))
    print(f"\n📊 Total predictions: {len(submission)}")
    print(f"💰 Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")