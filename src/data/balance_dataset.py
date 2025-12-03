"""
موازنة البيانات - Balance Dataset
Creates a balanced dataset by undersampling the majority class
"""

import argparse
import pandas as pd
from pathlib import Path
from src.utils import logger


def balance_dataset(input_csv: str, output_csv: str, random_state: int = 42):
    """Équilibrer dataset par sous-échantillonnage"""
    df = pd.read_csv(input_csv)
    logger.info(f"Dataset loaded: {len(df)} articles")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    logger.info(f"Original distribution:\n{label_counts}")
    
    # Separate by label
    df_credible = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]
    
    # Find minimum count
    min_count = min(len(df_credible), len(df_fake))
    logger.info(f"Minimum class size: {min_count}")
    
    # Undersample to balance
    df_credible_sampled = df_credible.sample(n=min_count, random_state=random_state)
    df_fake_sampled = df_fake.sample(n=min_count, random_state=random_state)
    
    # Combine and shuffle
    df_balanced = pd.concat([df_credible_sampled, df_fake_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Balanced dataset: {len(df_balanced)} articles")
    logger.info(f"New distribution:\n{df_balanced['label'].value_counts()}")
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)
    logger.info(f"Saved balanced dataset to: {output_csv}")
    
    return df_balanced


def main():
    parser = argparse.ArgumentParser(description="Équilibrage du dataset")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for sampling')
    
    args = parser.parse_args()
    
    balance_dataset(args.input, args.output, args.random_state)
    logger.info("Terminé avec succès")


if __name__ == '__main__':
    main()
