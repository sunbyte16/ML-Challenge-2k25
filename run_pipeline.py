"""
Quick pipeline runner for ML Challenge 2025
Author: Sunil Sharma
"""

import subprocess
import sys
from pathlib import Path

def run_quick_test():
    """Run a quick test of the pipeline"""
    print("🚀 Running Quick Test Pipeline...")
    print("=" * 50)
    
    try:
        # Run main pipeline with quick test flag
        result = subprocess.run([
            sys.executable, "main.py", 
            "--quick-test",
            "--save-model"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Quick test completed successfully!")
        else:
            print("❌ Quick test failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def run_full_pipeline():
    """Run the full pipeline"""
    print("🚀 Running Full Pipeline...")
    print("=" * 50)
    
    try:
        # Run main pipeline with all features
        result = subprocess.run([
            sys.executable, "main.py", 
            "--tune-hyperparameters",
            "--use-ensemble",
            "--save-model"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Full pipeline completed successfully!")
        else:
            print("❌ Full pipeline failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def main():
    """Main runner function"""
    print("🎯 ML Challenge 2025 - Pipeline Runner")
    print("Created with ❤️ by Sunil Sharma")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Quick Test (reduced features, fast)")
        print("2. Full Pipeline (all features, hyperparameter tuning)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            success = run_quick_test()
        elif choice == "2":
            success = run_full_pipeline()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
            continue
        
        if success:
            print("\n🎉 Check the 'output' folder for your submission file!")
        else:
            print("\n💡 Check the logs for error details.")

if __name__ == "__main__":
    main()