from pathlib import Path

import pandas as pd
import argparse

def merge_jsonl_with_csv(jsonl_path, csv_path, output_path):
    tgt_df = pd.read_json(jsonl_path, lines=True)
    csv_data = pd.read_csv(csv_path)
    df = tgt_df.merge(csv_data, on='key')
    df['prompt'] = df['prompt_tgt']
    df = df.drop(columns=['prompt_src', 'prompt_tgt'])
    df.to_json(output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True, type=Path)
    parser.add_argument('--csv', required=True, help='Path to the input CSV file.')
    parser.add_argument('--out', required=True, help='Path to the output JSONL file.')

    args = parser.parse_args()
    merge_jsonl_with_csv(args.jsonl, args.csv, args.out)
