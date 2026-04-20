import csv
import statistics
from collections import defaultdict
from pathlib import Path

from instruction_following_eval import evaluation_lib


def run_and_parse(language, input_data, input_response, output_dir, log_file):
    inputs = evaluation_lib.read_prompt_list(input_data)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(input_response)

    results = {}

    for func, name in [
        (evaluation_lib.test_instruction_following_strict, "strict"),
        (evaluation_lib.test_instruction_following_loose, "loose"),
    ]:
        print(f"Generating {name}...", file=log_file)

        outputs = []
        for inp in inputs:
            if inp.prompt in prompt_to_response:
                outputs.append(func(inp, prompt_to_response, language))

        follow_all = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all) / len(follow_all) if outputs else 0.0
        print(f"Accuracy: {accuracy}", file=log_file)

        out_path = output_dir / f"{name}.jsonl"
        evaluation_lib.write_outputs(out_path, outputs)
        print(f"Generated: {out_path}", file=log_file)

        prompt_level, instruction_level = evaluation_lib.print_report(outputs, log_file)
        results[name] = {
            "prompt": prompt_level,
            "instruction": instruction_level,
        }

        print("=" * 64, file=log_file)

    return results


if __name__ == "__main__":

    BASE_INPUTS = Path("/scratch/evalero/inference/inputs")
    BASE_OUTPUTS = Path("/scratch/evalero/inference/outputs")
    BASE_RESULTS = Path("/scratch/evalero/ifeval_results")

    LANGS = ["en", "es", "ca", "eu", "gl"]

    EXPERIMENTS = {
        "main": {
            "seeds": ["seed21", "seed22", "seed23"],
            "modes": ["merge", "original"],
            "output_root": lambda lang, mode: BASE_RESULTS / lang / mode,
        },
        # "ablation": {
        #     "seeds": ["seed3", "seed4", "seed5"],
        #     "modes": ["merge"],
        #     "output_root": lambda lang, mode: BASE_RESULTS / lang / "ablation",
        # },
    }

    for lang in LANGS:
        print(f"=== Evaluating language: {lang} ===")

        input_data = BASE_INPUTS / lang / f"input_data.{lang}.jsonl"
        if not input_data.exists():
            print(f"Missing input file for {lang}, skipping.")
            continue

        for exp_name, exp in EXPERIMENTS.items():
            print(f"--- Experiment: {exp_name} ---")

            for mode in exp["modes"]:
                aggregated = defaultdict(lambda: {"strict": [], "loose": []})

                lang_results_root = exp["output_root"](lang, mode)
                lang_results_root.mkdir(parents=True, exist_ok=True)

                for seed in exp["seeds"]:
                    seed_output_root = BASE_OUTPUTS / seed / mode / lang
                    if not seed_output_root.exists():
                        print(f"Missing output root: {seed_output_root}, skipping.")
                        continue

                    seed_results_root = lang_results_root / seed
                    seed_results_root.mkdir(parents=True, exist_ok=True)

                    for resp_path in seed_output_root.rglob("*.jsonl"):
                        rel_parts = resp_path.relative_to(seed_output_root).parts

                        # Expected structure:
                        # <family>/<thinking_mode>/<file>.jsonl
                        if len(rel_parts) < 3:
                            print(f"Skipping unexpected path: {resp_path}")
                            continue

                        family = rel_parts[0]
                        thinking_mode = rel_parts[1]
                        filename = resp_path.name

                        # with_thinking  -> only use *.cleaned.jsonl
                        # without_thinking -> only use normal *.jsonl
                        if thinking_mode == "with_thinking":
                            if not filename.endswith(".cleaned.jsonl"):
                                continue
                            model_name = filename[:-len(".cleaned.jsonl")]
                        elif thinking_mode == "without_thinking":
                            if filename.endswith(".cleaned.jsonl"):
                                continue
                            model_name = resp_path.stem
                        else:
                            print(f"Skipping unknown thinking mode path: {resp_path}")
                            continue

                        system_name = f"{family}/{thinking_mode}/{model_name}"

                        model_out_dir = seed_results_root / family / thinking_mode / model_name
                        model_out_dir.mkdir(parents=True, exist_ok=True)

                        print(f"Running {exp_name} | {mode} | {lang} | {seed} | {system_name}")

                        with (model_out_dir / "results.txt").open("w") as log_file:
                            res = run_and_parse(
                                lang,
                                input_data,
                                resp_path,
                                model_out_dir,
                                log_file,
                            )

                        aggregated[system_name]["strict"].append(
                            res["strict"]["instruction"]
                        )
                        aggregated[system_name]["loose"].append(
                            res["loose"]["instruction"]
                        )

                csv_path = lang_results_root / f"eval_summary_{lang}.csv"
                with csv_path.open("w", newline="") as wf:
                    writer = csv.DictWriter(
                        wf,
                        fieldnames=[
                            "system",
                            "strict_instruction_mean",
                            "strict_instruction_std",
                            "loose_instruction_mean",
                            "loose_instruction_std",
                        ],
                    )
                    writer.writeheader()

                    for system, vals in aggregated.items():
                        writer.writerow(
                            {
                                "system": system,
                                "strict_instruction_mean": statistics.mean(vals["strict"]),
                                "strict_instruction_std": (
                                    statistics.stdev(vals["strict"])
                                    if len(vals["strict"]) > 1 else 0.0
                                ),
                                "loose_instruction_mean": statistics.mean(vals["loose"]),
                                "loose_instruction_std": (
                                    statistics.stdev(vals["loose"])
                                    if len(vals["loose"]) > 1 else 0.0
                                ),
                            }
                        )

                print(f"Saved: {csv_path}")