import csv
import statistics
from collections import defaultdict
from pathlib import Path

from instruction_following_eval import evaluation_lib


def run_and_parse(language, input_data, input_response, output_dir, log_file):

    inputs = evaluation_lib.read_prompt_list(input_data)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(input_response)

    results = defaultdict(dict)
    for func, output_file_name_id in [
        (evaluation_lib.test_instruction_following_strict, 'strict'),
        (evaluation_lib.test_instruction_following_loose, 'loose'),
    ]:
        print(f'Generating {output_file_name_id}...', file=log_file)
        outputs = []
        for inp in inputs:
            if inp.prompt in prompt_to_response:
                outputs.append(func(inp, prompt_to_response, language))
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        print(f'Accuracy: {accuracy}', file=log_file)

        output_file_name = output_dir / f'{output_file_name_id}.jsonl'
        evaluation_lib.write_outputs(output_file_name, outputs)
        print(f'Generated: {output_file_name}', file=log_file)

        print('=' * 64, file=log_file)
        print(f'{output_file_name} Accuracy Scores:', file=log_file)
        prompt_level, instruction_level = evaluation_lib.print_report(outputs, log_file)
        results[output_file_name_id]['prompt'] = prompt_level
        results[output_file_name_id]['instruction'] = instruction_level

    return results


if __name__ == '__main__':

    LANGS = ['en', 'es', 'ca', 'eu']
    DATA_DIR = Path(__file__).parent.parent / 'data'
    OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
    RESULTS_DIR = Path(__file__).parent.parent / 'results'

    SYSTEM_ORDER = [
        'llama-3.1-8b-joint-eu',
        'llama-3.1-8b-instruct',
        'llama-3.1-8b-merge-eu',
        'llama-3.1-8b-merge-gl',
        'llama-3.1-8b-merge-ca',
        'llama-3.1-8b-merge-es',
        'llama-3.1-8b-merge-multi-eu30-gl15-ca5-ins50',
        'qwen3-8b-instruct',
        'qwen3-8b-merge-eu',
        'qwen3-8b-merge-gl',
        'qwen3-8b-merge-ca',
        'qwen3-8b-merge-es',
        'qwen3-8b-merge-multi-eu30-gl15-ca5-ins50',
        'qwen3-14b-instruct',
        'qwen3-14b-merge-eu',
        'qwen3-14b-merge-gl',
        'qwen3-14b-merge-ca',
        'qwen3-14b-merge-es',
        'qwen3-14b-merge-multi-eu30-gl15-ca5-ins50',
    ]

    for lang in LANGS:

        results = defaultdict(dict)

        for run in ['seed0', 'seed1', 'seed2']:

            print(f'=== Evaluating language: {lang} ===')
            input_data = DATA_DIR / f'input_data.{lang}.jsonl'
            lang_output_root = OUTPUT_DIR / run / lang
            response_files = list(lang_output_root.rglob('*.jsonl'))

            if not response_files:
                print(f'(No response files found for {lang})')
                continue

            for resp_path in response_files:
                system_name = resp_path.stem

                output_dir = RESULTS_DIR / run / lang / system_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f'Running evaluation for {system_name}')
                log_file = (output_dir / 'results.txt').open('w')
                result_dict = run_and_parse(lang, input_data, resp_path, output_dir, log_file)
                log_file.close()

                if not any(x in system_name for x in ['instruct', 'joint', 'merge']):
                    continue

                if 'strict_instruction' not in results[system_name]:
                    results[system_name]['strict_instruction'] = []
                results[system_name]['strict_instruction'].append(result_dict['strict']['instruction'])

                if 'loose_instruction' not in results[system_name]:
                    results[system_name]['loose_instruction'] = []
                results[system_name]['loose_instruction'].append(result_dict['loose']['instruction'])

        results = [{
            'system': k,
            'strict_instruction mean': statistics.mean(v['strict_instruction']),
            'strict_instruction std': statistics.stdev(v['strict_instruction']),
            'loose_instruction mean': statistics.mean(v['loose_instruction']),
            'loose_instruction std': statistics.stdev(v['loose_instruction']),
        } for k, v in results.items() ]
        order_index = {name: i for i, name in enumerate(SYSTEM_ORDER)}
        results.sort(key=lambda x: order_index.get(x['system'], len(SYSTEM_ORDER)))

        csv_path = f'eval_summary_{lang}.csv'
        with open(csv_path, 'w', newline='') as wf:
            writer = csv.DictWriter(wf, fieldnames=['system', 'strict_instruction mean', 'strict_instruction std', 'loose_instruction mean', 'loose_instruction std'])
            writer.writeheader()
            writer.writerows(results)

        print(f'Saved: {csv_path} ({len(results)} systems)')
