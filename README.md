# Iber-IFEval

A modified version of the original **IFEval** codebase, adapted for multilingual evaluation.

This repository is used to evaluate instruction-following performance across several languages, seeds, and experiment settings, and to aggregate results into summary CSV files.

## Main script

The main script currently used in this repository is:

```bash
instruction_following_eval/runner.py
```

## What the runner does

The runner:

- reads language-specific prompt files
- reads model response files in JSONL format
- evaluates responses with:
  - **strict** instruction-following
  - **loose** instruction-following
- saves per-run outputs
- writes log files
- aggregates instruction-level scores across seeds
- exports summary CSV files

## Installation

Install the required dependencies with:

```bash
pip3 install -r requirements.txt
```

## How to run

From the repository root, run:

```bash
python3 -m instruction_following_eval.runner
```

You can also run it as a script:

```bash
python3 instruction_following_eval/runner.py
```

## Important: paths must be adapted

The current version of `runner.py` uses hardcoded cluster paths:

```python
BASE_INPUTS = Path("/scratch/evalero/inference/inputs")
BASE_OUTPUTS = Path("/scratch/evalero/inference/outputs")
BASE_RESULTS = Path("/scratch/evalero/ifeval_results")
```

If you want to reuse this repository on another machine or account, you will need to modify these paths.

You may also need to adapt:

- the list of languages
- the experiment configuration
- the seed names
- the expected directory structure for model outputs

## Current languages

The runner is currently configured for these languages:

- `en` — English
- `es` — Spanish
- `ca` — Catalan
- `eu` — Basque
- `gl` — Galician

## Required input datasets

To run IFEval, you need prompt data for each language in JSONL format.

The datasets currently used are:

- **English**: `google/IFEval`
- **Spanish**: `BSC-LT/IFEval_es`
- **Catalan**: `projecte-aina/IFEval_ca`
- **Basque**: created by us, available in the HiTZ collection
- **Galician**: created by us, available in the HiTZ collection

Basque and Galician can be downloaded from this Hugging Face collection:

<https://huggingface.co/collections/HiTZ/merge-and-conquer>

All JSONL files can be downloaded from their corresponding repositories or collections on Hugging Face.

This step is required: if the input files are not available in the expected paths, the runner will skip that language and the evaluation will not run.

## Expected prompt input structure

Prompt files are expected under:

```bash
<BASE_INPUTS>/<lang>/input_data.<lang>.jsonl
```

Example:

```bash
/scratch/evalero/inference/inputs/en/input_data.en.jsonl
/scratch/evalero/inference/inputs/es/input_data.es.jsonl
```

After downloading the datasets, the input files should be placed following that structure. For example:

```bash
/scratch/evalero/inference/inputs/en/input_data.en.jsonl
/scratch/evalero/inference/inputs/es/input_data.es.jsonl
/scratch/evalero/inference/inputs/ca/input_data.ca.jsonl
/scratch/evalero/inference/inputs/eu/input_data.eu.jsonl
/scratch/evalero/inference/inputs/gl/input_data.gl.jsonl
```

## Expected response structure

Model response files are expected under:

```bash
<BASE_OUTPUTS>/<seed>/<mode>/<lang>/<family>/<thinking_mode>/*.jsonl
```

Example:

```bash
/scratch/evalero/inference/outputs/seed21/merge/en/qwen/with_thinking/model.cleaned.jsonl
/scratch/evalero/inference/outputs/seed21/merge/en/qwen/without_thinking/model.jsonl
```

### Naming conventions

- `with_thinking` uses only files ending in `.cleaned.jsonl`
- `without_thinking` uses only standard `.jsonl` files
- unknown thinking-mode directory names are skipped

## Output structure

Results are written under:

```bash
<BASE_RESULTS>/<lang>/<mode>/
```

For each evaluated system, the runner stores:

- `strict.jsonl`
- `loose.jsonl`
- `results.txt`

It also generates one aggregated CSV per language and mode:

```bash
eval_summary_<lang>.csv
```

These CSV files contain:

- `system`
- `strict_instruction_mean`
- `strict_instruction_std`
- `loose_instruction_mean`
- `loose_instruction_std`

## Current experiment setup

The active configuration in `runner.py` is:

- **experiment name:** `main`
- **seeds:** `seed21`, `seed22`, `seed23`
- **modes:** `merge`, `original`

There is also commented code for additional ablation-style experiments.

## Notes

This repository is **not** a clean mirror of the original upstream project anymore. It contains custom modifications made for a personal evaluation pipeline and cluster setup.

If you want to adapt the workflow, the main file to inspect is:

```bash
instruction_following_eval/runner.py
```

## Original reference

This repository is based on the original IFEval codebase and paper:

**Instruction-Following Evaluation for Large Language Models**  
<https://arxiv.org/abs/2311.07911>

If you use the original work, please consider citing:

```bibtex
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```
