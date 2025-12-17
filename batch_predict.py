import os
import pandas as pd
import argparse
from model_utils import EnsemblePredictor

# 启用 MPS 回退机制
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def process_csv(input_file, output_file, headline_col="headline", desc_col="short_description"):
    print(f"Loading models...")
    try:
        predictor = EnsemblePredictor()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print(f"Reading CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 检查列是否存在
    if headline_col not in df.columns and desc_col not in df.columns:
        print(f"Error: Neither '{headline_col}' nor '{desc_col}' columns found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"Processing {len(df)} rows...")
    
    predictions = []
    confidences = []

    for index, row in df.iterrows():
        headline = str(row[headline_col]) if headline_col in df.columns and pd.notna(row[headline_col]) else ""
        description = str(row[desc_col]) if desc_col in df.columns and pd.notna(row[desc_col]) else ""
        
        category, confidence = predictor.predict(headline, description)
        predictions.append(category)
        confidences.append(confidence)
        
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} rows...")

    df["predicted_category"] = predictions
    df["confidence"] = confidences

    print(f"Saving results to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch classify news from CSV file.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", help="Path to output CSV file (default: input_labeled.csv)", default=None)
    parser.add_argument("--headline_col", help="Column name for headline", default="headline")
    parser.add_argument("--desc_col", help="Column name for description", default="short_description")

    args = parser.parse_args()

    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_labeled{ext}"

    process_csv(args.input_csv, args.output_csv, args.headline_col, args.desc_col)
