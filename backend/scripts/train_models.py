#!/usr/bin/env python
"""Script to train models from command line."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.ml.pipelines.training_pipeline import TrainingPipeline
from app.ml.pipelines.preprocessing import generate_sample_data


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["xgboost", "lightgbm", "random_forest", "logistic", "ensemble"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (parquet or csv)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic data if no data provided"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples for synthetic data"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing"
    )
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        if args.data_path.endswith(".parquet"):
            df = pd.read_parquet(args.data_path)
        else:
            df = pd.read_csv(args.data_path)
    elif args.generate_data:
        print(f"Generating {args.n_samples} synthetic samples")
        df = generate_sample_data(n_samples=args.n_samples)
        
        # Save generated data
        os.makedirs("./data/processed", exist_ok=True)
        df.to_parquet("./data/processed/churn_data.parquet")
        print("Saved generated data to ./data/processed/churn_data.parquet")
    else:
        # Try to load existing processed data
        default_path = "./data/processed/churn_data.parquet"
        if os.path.exists(default_path):
            print(f"Loading existing data from {default_path}")
            df = pd.read_parquet(default_path)
        else:
            print("No data provided. Generating synthetic data...")
            df = generate_sample_data(n_samples=args.n_samples)
            os.makedirs("./data/processed", exist_ok=True)
            df.to_parquet(default_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['churn'].value_counts(normalize=True)}\n")
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        model_type=args.model_type,
        model_path=args.output_path,
        test_size=args.test_size
    )
    
    # Train
    print(f"Training {args.model_type} model...")
    results = pipeline.train(df, target_column="churn")
    
    # Print results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Model Type: {results['model_type']}")
    print(f"Version: {results['version']}")
    print(f"Training Samples: {results['training_samples']}")
    print(f"Test Samples: {results['test_samples']}")
    print("\nMetrics:")
    
    for metric, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel saved to: {args.output_path}")


if __name__ == "__main__":
    main()
