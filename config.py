"""
Configuration file for ML Challenge 2025
Author: Sunil Sharma
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "dataset"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data file paths
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_TEST_FILE = DATA_DIR / "sample_test.csv"
SAMPLE_OUTPUT_FILE = DATA_DIR / "sample_test_out.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Image processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_IMAGES_TO_DOWNLOAD = 1000  # Limit for testing

# Text processing
MAX_TEXT_LENGTH = 512
VOCAB_SIZE = 10000

# Model training
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Output
OUTPUT_FILE = "test_out.csv"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"