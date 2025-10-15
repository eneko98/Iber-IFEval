import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a professional English (EN) to {language} ({language_iso}) translator. "
    "The user will give you an English instruction, which you must translate to {language}, respecting the provided format. "
    "Do not comment or add any new content, answer directly with the {language} translation only."
)


class Response(BaseModel):
    EU: str


iso2full = {
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'es': 'Spanish'
}


def load_data(input_path: Path, few_shot_path: Path, language_iso: str):
    language_full = iso2full[language_iso]
    df_input = pd.read_json(input_path, lines=True)
    if not few_shot_path.exists():
        few_shot_examples = []
    else:
        df_few_shot = pd.read_csv(few_shot_path, encoding="utf-8", lineterminator="\n")
        few_shot_examples = [
            [
                {'role': 'user', 'content': f'Translate the following text to {language_full}.\n\n{{"EN": {json.dumps(row["prompt_src"])}}}'},
                {'role': 'assistant', 'content': f'{{"{language_iso.upper()}": {json.dumps(row["prompt_tgt"])}}}'},
            ]
            for idx, row in df_few_shot.iterrows()
        ]
        few_shot_examples = [turn for pair in few_shot_examples for turn in pair]
    conversations = [
        [
            {'role': 'system', 'content': SYSTEM_PROMPT.replace('{language}', language_full).replace('{language_iso}', language_iso.upper())},
            *few_shot_examples,
            {'role': 'user', 'content': f'Translate the following text to {language_full}.\n\n{{"EN": {json.dumps(x)}}}'},
        ]
        for x in df_input['prompt'].tolist()
    ]
    return df_input, conversations


def main(input_path: Path, few_shot_path: Path, output_path: Path, model: str, api_key: str, language_iso: str, build_few_shot: bool = False):
    print(f'Preparing data for GPT-4o...')
    df_src, messages = load_data(input_path, few_shot_path, language_iso)
    print(f'{len(messages)} prompts loaded.')
    client = OpenAI(api_key=api_key)
    with output_path.open('w', encoding='utf-8') as wf:
        for i, conv in enumerate(tqdm(messages)):
            if build_few_shot and i == 5:
                sys.exit()
            try:
                response = client.responses.parse(input=conv, model=model, text_format=Response)
                completion = response.output_parsed.EU
                record = df_src.iloc[i].to_dict()
                record['prompt'] = completion
                wf.write(f'{json.dumps(record)}\n')
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                logging.exception(e)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('language', choices=list(iso2full.keys()))
    ap.add_argument('api_key', type=str)
    ap.add_argument('--build-few-shot', action='store_true')

    args = ap.parse_args()

    main(
        Path(__file__).parent.parent / 'data' / 'input_data.jsonl',
        Path(__file__).parent.parent / 'data' / f'input_data.{args.language}-few-shot.csv',
        Path(__file__).parent.parent / 'data' / f'input_data.{args.language}-gpt.jsonl',
        'gpt-4o-2024-08-06',
        args.api_key,
        args.language,
        args.build_few_shot
    )
