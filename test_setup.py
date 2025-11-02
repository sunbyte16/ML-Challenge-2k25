"""
Test script to verify project setup
Author: Sunil Sharma
"""

import sys
import pandas as pd
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import sklearn
        print("✅ Basic libraries imported successfully")
        
        # Test custom modules
        from src.data_loader import DataLoader
        from src.feature_engineering import FeatureEngineer
        from src.models import ModelTrainer
        from config import *
        print("✅ Custom modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    try:
        from src.data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test loading training data
        train_df = loader.load_train_data()
        print(f"✅ Training data loaded: {train_df.shape}")
        
        # Test loading test data
        test_df = loader.load_test_data()
        print(f"✅ Test data loaded: {test_df.shape}")
        
        # Basic validation
        assert 'sample_id' in train_df.columns, "sample_id column missing"
        assert 'catalog_content' in train_df.columns, "catalog_content column missing"
        assert 'image_link' in train_df.columns, "image_link column missing"
        assert 'price' in train_df.columns, "price column missing in training data"
        
        print("✅ Data validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from src.data_loader import load_and_preprocess_data
        from src.feature_engineering import FeatureEngineer
        
        # Load small sample for testing
        train_df, test_df = load_and_preprocess_data()
        
        # Use small sample for quick testing
        train_sample = train_df.head(100)
        test_sample = test_df.head(100)
        
        # Test feature engineering
        engineer = FeatureEngineer(max_tfidf_features=100)  # Small for testing
        train_features, test_features = engineer.engineer_features(train_sample, test_sample)
        
        print(f"✅ Features engineered - Train: {train_features.shape}, Test: {test_features.shape}")
        print(f"✅ Feature columns: {len(engineer.feature_columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\n🤖 Testing model training...")
    
    try:
        from src.data_loader import load_and_preprocess_data
        from src.feature_engineering import FeatureEngineer
        from src.models import ModelTrainer
        
        # Load very small sample for quick testing
        train_df, test_df = load_and_preprocess_data()
        train_sample = train_df.head(50)  # Very small sample
        test_sample = test_df.head(50)
        
        # Feature engineering
        engineer = FeatureEngineer(max_tfidf_features=50)  # Very small
        train_features, test_features = engineer.engineer_features(train_sample, test_sample)
        
        # Prepare data
        X_train = train_features[engineer.feature_columns]
        y_train = train_features['price']
        
        # Test single model training
        trainer = ModelTrainer()
        
        # Test just linear regression for speed
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        result = trainer.train_model(model, X_train, y_train, 'test_linear')
        
        print(f"✅ Model training test passed: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        'dataset', 'src', 'models', 'logs', 'output', 'notebooks'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🎯 ML Challenge 2025 - Setup Test")
    print("Created with ❤️ by Sunil Sharma")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready!")
        print("\n💡 Next steps:")
        print("1. Run 'python run_pipeline.py' for interactive pipeline")
        print("2. Run 'python main.py --quick-test' for quick test")
        print("3. Open notebooks/exploratory_data_analysis.ipynb for EDA")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check if dataset files exist in dataset/ folder")
        print("3. Ensure all directories are created properly")

if __name__ == "__main__":
    main()