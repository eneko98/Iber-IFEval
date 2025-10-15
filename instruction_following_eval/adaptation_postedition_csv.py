from pathlib import Path

import pandas as pd

input_a = Path(__file__).parent.parent / 'data' / 'input_data.jsonl'

# input_b = Path(__file__).parent.parent  / 'data' / 'input_data.eu-few-shot.jsonl'
# output = Path(__file__).parent.parent  / 'data' / 'input_data.eu-few-shot.csv'

# input_b = Path(__file__).parent.parent  / 'data' / 'input_data.eu-gpt.jsonl'
# output = Path(__file__).parent.parent  / 'data' / 'input_data.eu-gpt-postediting.csv'

input_b = Path(__file__).parent.parent  / 'data' / 'input_data.gl-few-shot.jsonl'
output = Path(__file__).parent.parent  / 'data' / 'input_data.gl-few-shot.csv'

df_a = pd.read_json(input_a, lines=True)
df_b = pd.read_json(input_b, lines=True)

df = df_a.merge(df_b, how='left', on='key', suffixes=('_src', '_tgt'))[['key', 'prompt_src', 'prompt_tgt']]

df.to_csv(output, index=False)
