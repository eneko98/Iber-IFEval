import json
import argparse
from pathlib import Path

import pandas as pd


def parse_value(user_input):
    """Try to interpret input as JSON if possible, else return as string."""
    if not user_input.strip():
        return None
    if '[\'' in user_input or '["' in user_input:
        return json.loads(user_input)
    if user_input.isnumeric():
        return int(user_input)
    return user_input


def main(input_path: Path, output_path: Path):

    processed_keys = set()

    if output_path.exists():
        df_out = pd.read_json(output_path, lines=True)
        processed_keys = set(df_out['key'])

    df_in = pd.read_json(input_path, lines=True)

    with output_path.open('a', encoding='utf-8') as outfile:
        for _, row in df_in.iterrows():
            key = row['key']

            if key in processed_keys:
                continue

            print(f'\n==Prompt:==\n\n{row["prompt"]}\n\n')
            print(f'Instruction IDs: {row["instruction_id_list"]}')
            print(f'Current kwargs: {row["kwargs"]}')

            kwargs = row['kwargs']
            new_kwargs = []

            for kw in kwargs:
                updated_kw = {}
                for k, v in kw.items():
                    if k not in ('relation', 'frequency', 'language') and not any(k.startswith(x) for x in ['num_', 'let_', 'capital_', 'nth_']):
                        user_input = input(f'  {k} [{v}]: ').strip()
                    else:
                        user_input = None
                    updated_kw[k] = parse_value(user_input) if user_input else v
                new_kwargs.append(updated_kw)
            row['kwargs'] = new_kwargs

            outfile.write(row.to_json() + '\n')
            outfile.flush()

            processed_keys.add(key)
            print('✅ Saved.\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    main(args.input, args.output)
