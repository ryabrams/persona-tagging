# Makefile for Persona Classification System

# Default target - show help
.DEFAULT_GOAL := help

# Phony targets (not actual files)
.PHONY: help train predict clean all check setup

# Help - display available commands
help:
	@echo "Persona Classification System - Available Commands:"
	@echo "=================================================="
	@echo "  make train   - Train the model using training data"
	@echo "  make predict - Run predictions on input data"
	@echo "  make all     - Run training followed by prediction"
	@echo "  make check   - Check if all required files exist"
	@echo "  make clean   - Remove generated files"
	@echo "  make help    - Show this help message"

# Train the model
train: data/training_data.csv scripts/train_model.py
	@echo "Starting model training..."
	python scripts/train_model.py

# Run predictions (depends on trained model)
predict: model/persona_classifier.pkl data/input.csv scripts/predict.py
	@echo "Running predictions..."
	python scripts/predict.py

# Run full pipeline - train then predict
all: train predict
	@echo "✅ Full pipeline completed!"

# Check if all required files and directories exist
check:
	@echo "Checking system setup..."
	@echo "========================"
	@echo "Required directories:"
	@test -d data || echo "❌ Missing: data/"
	@test -d scripts || echo "❌ Missing: scripts/"
	@test -d model || echo "❌ Missing: model/"
	@test -d data && echo "✅ Found: data/"
	@test -d scripts && echo "✅ Found: scripts/"
	@test -d model && echo "✅ Found: model/"
	@echo ""
	@echo "Required scripts:"
	@test -f scripts/train_model.py || echo "❌ Missing: scripts/train_model.py"
	@test -f scripts/predict.py || echo "❌ Missing: scripts/predict.py"
	@test -f scripts/title_standardizer.py || echo "❌ Missing: scripts/title_standardizer.py"
	@test -f scripts/train_model.py && echo "✅ Found: scripts/train_model.py"
	@test -f scripts/predict.py && echo "✅ Found: scripts/predict.py"
	@test -f scripts/title_standardizer.py && echo "✅ Found: scripts/title_standardizer.py"
	@echo ""
	@echo "Required data files:"
	@test -f data/training_data.csv || echo "❌ Missing: data/training_data.csv (required for training)"
	@test -f data/input.csv || echo "❌ Missing: data/input.csv (required for prediction)"
	@test -f data/training_data.csv && echo "✅ Found: data/training_data.csv"
	@test -f data/input.csv && echo "✅ Found: data/input.csv"
	@echo ""
	@echo "Optional data files:"
	@test -f data/keyword_matching.csv || echo "⚠️  Missing: data/keyword_matching.csv (optional - for keyword rules)"
	@test -f data/title_reference.csv || echo "⚠️  Missing: data/title_reference.csv (optional - for title standardization)"
	@test -f data/keyword_matching.csv && echo "✅ Found: data/keyword_matching.csv"
	@test -f data/title_reference.csv && echo "✅ Found: data/title_reference.csv"
	@echo ""
	@echo "Model files:"
	@test -f model/persona_classifier.pkl || echo "⚠️  Missing: model/persona_classifier.pkl (will be created by 'make train')"
	@test -f model/persona_classifier.pkl && echo "✅ Found: model/persona_classifier.pkl"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f tagged_personas.csv
	@rm -f model/persona_classifier.pkl
	@rm -f model/model_metadata.txt
	@echo "✅ Cleaned generated files"

# Setup directories (helpful for initial setup)
setup:
	@echo "Setting up directory structure..."
	@mkdir -p data
	@mkdir -p scripts
	@mkdir -p model
	@echo "✅ Directory structure created"

# Dependency rules - these ensure files exist before running commands
model/persona_classifier.pkl: data/training_data.csv
	@echo "Model not found or training data updated. Training model..."
	python scripts/train_model.py

# Additional convenience targets

# Re-train model (force training even if model exists)
retrain:
	@echo "Force re-training model..."
	@rm -f model/persona_classifier.pkl
	python scripts/train_model.py

# Quick test - train on training data, predict on same data (for testing)
test: train
	@echo "Running test prediction on training data..."
	@cp data/training_data.csv data/input.csv
	python scripts/predict.py
	@echo "✅ Test completed - check tagged_personas.csv"