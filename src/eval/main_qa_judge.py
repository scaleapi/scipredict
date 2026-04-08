import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os
import json
import hashlib
from dotenv import load_dotenv
import logging
import re
import litellm
import asyncio
import aiofiles
import numpy as np
from typing import Dict, List
from copy import deepcopy

from src.api_client.api_conf import APIConfiguration
from src.api_client.client import LLMClient
from src.utils.prompts_utils import read_prompts
from src.utils.llm_utils import get_response_content
from src.utils.conf_utils import load_config_with_subset


def normalize_mcq_answer(answer: str) -> str:
    assert isinstance(answer, str), "Input MCQ answer must be a string."
    assert re.match(r'^\s*[\w\d][\.\)]?(\s*,\s*[\w\d][\.\)]?)*\s*$', answer.strip()), \
        "Invalid MCQ answer format: " + answer
    cleaned = re.sub(r'[\.\)\(\s]', '', answer.upper())
    parts = sorted([part for part in cleaned.split(',') if part])
    return ",".join(parts)


def compare_mcq(prediction: str, ground_truth: str) -> int:
    return 1 if normalize_mcq_answer(prediction) == normalize_mcq_answer(ground_truth) else 0


def is_numerical_in_range(prediction: str, gt_range_str: str) -> int:
    """
    Checks if a numerical prediction falls within a ground truth range.
    Handles formats like: 'X-Y', '<X', '>X', 'X', and comma-separated lists of these.
    Also handles scientific notation and negative numbers.
    Returns 1 if the prediction is valid and in ANY specified range, 0 otherwise.
    """
    pred_float, pred_low, pred_high = None, None, None

    if prediction == "_":
        return 0

    try:
        pred_float = float(prediction)
    except (ValueError, TypeError):
        try:
            lo, hi = prediction.split("-", 1)
            pred_low, pred_high = float(lo.strip()), float(hi.strip())
        except (ValueError, TypeError):
            raise ValueError(f"Prediction '{prediction}' is not a valid float.")

    if pd.isna(gt_range_str) or not str(gt_range_str).strip():
        raise ValueError("Input ground truth range string is empty or NaN.")

    # Split by comma for multiple possible valid ranges (e.g., "10-100")
    individual_ranges = str(gt_range_str).split(',')

    for part in individual_ranges:
        part = part.strip()
        if not part:
            continue

        try:
            # Case 1: Less than (e.g., "<0.01")
            if part.startswith('<'):
                bound = float(part[1:])
                if pred_float < bound:
                    return 1
            # Case 2: Greater than (e.g., ">100")
            elif part.startswith('>'):
                bound = float(part[1:])
                if pred_float > bound:
                    return 1
            # Case 3: A range (e.g., "89-99", "-0.40--0.25", "-8.7e-30-1.1e31")
            elif '-' in part and (
                    match := re.match(r'^(-?[\d.]+(?:[eE][+-]?\d+)?)-(-?[\d.]+(?:[eE][+-]?\d+)?)$', part)):
                lower_bound = float(match.group(1))
                upper_bound = float(match.group(2))
                if pred_float is not None and lower_bound <= pred_float <= upper_bound:
                    return 1
                if pred_low is not None and pred_high is not None and pred_low >= lower_bound and pred_high <= upper_bound:
                    return 1
            # Case 4: A single value (e.g., "100")
            else:
                gt_float = float(part)
                # Use a small tolerance for single float comparisons
                if np.isclose(pred_float, gt_float):
                    return 1
        except (ValueError, IndexError):
            # Ignore malformed range parts and continue to the next
            print(f"Warning: Could not parse ground truth range part: '{part}'")
            continue
    return 0


def get_criterion_hash_id(text: str) -> str:
    """Calculates the SHA3_256 hash of a string and returns the first 10 hex characters."""
    return hashlib.sha3_256(text.encode('utf-8')).hexdigest()[:10]


def get_criteria_with_hashes(row: pd.Series, implicit_rubrics: bool = False) -> (Dict[str, str], Dict[str, int]):
    if implicit_rubrics:
        generic_criterion_text = "Determine the rubrics based on the ground truth answer."
        generic_criterion_hash = get_criterion_hash_id(generic_criterion_text)
        return {generic_criterion_hash: generic_criterion_text}, {generic_criterion_hash: 0}

    criteria = {}
    hash_to_idx = {}
    for col in row.index:
        if col.startswith('criterion_') and pd.notna(row[col]):
            criterion_idx = int(re.match(r'criterion_(\d+)', col).group(1))
            criterion_text = str(row[col])
            criterion_hash = get_criterion_hash_id(criterion_text)
            criteria[criterion_hash] = criterion_text
            hash_to_idx[criterion_hash] = criterion_idx
    return criteria, hash_to_idx


def format_criteria_for_prompt(criteria_with_hashes: Dict[str, str]) -> str:
    return "\n".join(f"- Criterion {hash_id}: {text}" for hash_id, text in criteria_with_hashes.items())


def generate_dynamic_schema(base_schema: Dict, criteria_hashes: List[str]) -> Dict:
    """Generates a dynamic JSON schema with nested objects for each criterion."""
    new_schema = deepcopy(base_schema)
    new_schema["properties"] = new_schema.get("properties", {})
    new_schema["required"] = new_schema.get("required", [])
    new_schema["propertyOrdering"] = []

    for hash_id in criteria_hashes:
        new_schema["properties"][hash_id] = {
            "type": "object",
            "properties": {
                "satisfied": {
                    "type": "boolean"
                },
                "reasoning": {
                    "type": "string"
                }
            },
            "propertyOrdering": ["satisfied", "reasoning"],
            "required": ["satisfied", "reasoning"]
        }
        new_schema["propertyOrdering"].append(hash_id)
        new_schema["required"].append(hash_id)

    return new_schema


async def process_row(row, cfg: DictConfig, llm_client_factory, prompt_template, base_schema,
                      implicit_rubrics: bool = False):
    assert implicit_rubrics in [True, False]
    result = row.to_dict()
    pq_format = str(row.get('pq_format', '')).strip().lower()

    result["judgments"] = {}
    if pq_format in ['mcq', 'multiple-choice']:
        try:
            result['judgments']['score'] = compare_mcq(str(row.get('answer')), str(row.get('clean_gta')))
            result['eval_method'] = 'rule-based'
        except Exception as e:
            result['error'] = str(e)
            result['judgments']['score'] = 0
            result['eval_method'] = 'error'
        return result
    elif pq_format in ['numerical values', 'numerical']:
        try:
            result['judgments']['score'] = is_numerical_in_range(str(row.get('answer')), str(row.get('clean_gta')))
            result['eval_method'] = 'rule-based'
        except Exception as e:
            result['error'] = str(e)
            result['judgments']['score'] = 0
            result['eval_method'] = 'error'
        return result
    elif pq_format in ['free-format question', 'free-form']:
        if 'use_llm_judge' in cfg.run and not cfg.run.use_llm_judge:
            result['judgments']['score'], result['eval_method'] = 'skipped', 'skipped'
            return result

        criteria_with_hashes, hash_to_idx = get_criteria_with_hashes(row, implicit_rubrics=implicit_rubrics)

        if not criteria_with_hashes:
            result['judgments']['score'], result['eval_method'] = 'no_criteria', 'skipped'
            return result

        dynamic_schema = generate_dynamic_schema(base_schema, list(criteria_with_hashes.keys()))
        llm_client = llm_client_factory(dynamic_schema)

        context = {
            'domain': row.get('domain_x'),
            'field': row.get('field_x'),
            'rubric_criteria_lines': format_criteria_for_prompt(
                criteria_with_hashes),
            'outcome_prediction_question': row.get('outcome_prediction_question_x'),
            'predicted_answer': row.get('answer', 'No answer provided.'),
            'clean_gta': row.get('clean_gta', 'No ground truth provided.'),
            'experimental_setup': row.get('experimental_setup_x'),
            'measurement_taken': row.get('measurement_taken_x'),
        }

        sys_prompt = read_prompts(prompt_template=prompt_template, key='SYS', context=context)
        user_prompt = read_prompts(prompt_template=prompt_template, key='USER', context=context)

        retries = 3
        while retries > 0:
            retries -= 1
            try:
                response = await asyncio.to_thread(llm_client, system=sys_prompt, query=user_prompt)
                judge_json = get_response_content(response, to_json=True)
                for criterion_hash, criterion_value in judge_json.items():
                    criterion_value['index'] = hash_to_idx.get(criterion_hash, None)
                print(response)
                result['judgments'].update(judge_json)
                result['eval_method'] = 'llm-judge'

                satisfied_count = sum(
                    1 for criterion in judge_json.values()
                    if isinstance(criterion, dict) and criterion.get('satisfied') is True
                )
                total_criteria = len(criteria_with_hashes)
                result['judgments']['score'] = int(satisfied_count == total_criteria)
                result['judgments']['score_tot'] = satisfied_count

                print(f"Successfully judged row {row['task']}")
                break
            except litellm.RateLimitError:
                wait_time = 10
                retries += 1
                print(f"Rate limit error for row {row['task']}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Error judging row {row['task']}: {e}. Retrying...")
                if retries == 0:
                    result['error'], result['eval_method'] = str(e), 'error'
                    break
        return result
    else:
        result['eval_method'] = 'unsupported_format'
        return result


async def process_rows(df, cfg, llm_client_factory, prompt_template, base_schema, output_jsonl_path, implicit_rubrics):
    tasks = [process_row(row, cfg, llm_client_factory, prompt_template, base_schema, implicit_rubrics)
             for _, row in tqdm(df.iterrows(), total=len(df), desc="Submitting judging tasks")]

    print(f"Judged results will be progressively saved to {output_jsonl_path}")

    for completed_task in asyncio.as_completed(tasks):
        # async with aiofiles.open(output_jsonl_path, 'a') as jsonl_file:
        with open(output_jsonl_path, 'a') as jsonl_file:
            result = await completed_task
            jsonl_file.write(json.dumps(result, default=str) + '\n')


def main() -> None:
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../../configs/conf.yaml'))
    assert 'sub_confs' in cfg, "sub_confs key not found in the main config."
    curr_subconf_key = ''.join(os.path.basename(__file__).split('.')[:-1])
    assert curr_subconf_key in cfg.sub_confs, f"{curr_subconf_key} key not found in sub_confs."
    for key, path in cfg.sub_confs[curr_subconf_key].items():
        cfg = load_config_with_subset(cfg, key, path)
    # Merge command-line arguments into the final configuration
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    print(f"Using config: {cfg}")

    load_dotenv()
    logging.getLogger().setLevel(logging.DEBUG)
    for logger_name in logging.root.manager.loggerDict:
        print(f'Muting logs from {logger_name}')
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    assert 'input_predictions_path' in cfg.run, "run.input_predictions_path must be specified in the config. Please add config in command line using run.input_predictions_path=/path/to/qa_main_output.jsonl."

    input_filename = os.path.basename(cfg.run.input_predictions_path)
    input_dir = os.path.dirname(cfg.run.input_predictions_path)
    output_jsonl_path = os.path.join(input_dir, f"judged_{input_filename}")

    print(f"Input predictions: {cfg.run.input_predictions_path}")
    print(f"Output will be saved to: {output_jsonl_path}")

    try:
        df_preds = pd.read_json(cfg.run.input_predictions_path, lines=True)
    except Exception as e:
        print(f"Error loading predictions file: {e}");
        return

    implicit_rubrics = '--implicit-rubrics' in cli_conf or '--implicit_rubrics' in cli_conf
    print(f"Using implicit rubrics: {implicit_rubrics}")

    df_main = pd.read_csv(cfg.data.main)
    df_main.columns = df_main.columns.str.lower()

    if not implicit_rubrics:
        df_rubrics = pd.read_csv(cfg.data.rubrics)
        df_rubrics.columns = df_rubrics.columns.str.lower()

    df_preds['task'] = df_preds['task'].astype(str)
    df_main['task'] = df_main['task'].astype(str)
    if not implicit_rubrics:
        df_rubrics['task'] = df_rubrics['task'].astype(str)

    main_cols_needed = ['task', 'clean_gta', 'domain', 'field', 'outcome_prediction_question', 'experimental_setup',
                        'measurement_taken']
    df = pd.merge(df_preds, df_main[main_cols_needed], on='task', how='left', suffixes=('_preds', '_'))

    if not implicit_rubrics:
        df = pd.merge(df, df_rubrics, left_on='task', right_on='prev_task_id', how='left', suffixes=('', '_rubric'))
    print(df)
    criteria_cols = [c for c in df.columns if c.startswith('criterion_')]
    df[criteria_cols] = df[criteria_cols].fillna(value=np.nan)
    print('Merged data digest: ', hashlib.sha3_256(str(df).encode('utf-8')).hexdigest())

    print(f"Loaded and merged data. Total rows to process: {len(df)}.")
    if cfg.run.limit: df = df.iloc[:cfg.run.limit]; print(f"Processing a limited subset of {len(df)} rows.")

    api_conf = APIConfiguration(
        model_name=cfg.model.model_name, model_provider=cfg.model.model_provider,
        api_base=cfg.model.api_base, use_cache=cfg.model.use_cache)

    with open(cfg.run.prompts[cfg.run.key], 'r') as f:
        prompt_template = f.read()
    with open(cfg.run.schemas[cfg.run.key], 'r') as f:
        base_schema = json.load(f)

    def llm_client_factory(output_schema):
        return LLMClient(api_conf=api_conf, output_schema=output_schema, temperature=cfg.model.temperature)

    if os.path.exists(output_jsonl_path):
        os.remove(output_jsonl_path)

    asyncio.run(process_rows(
        df=df, cfg=cfg, llm_client_factory=llm_client_factory,
        prompt_template=prompt_template, base_schema=base_schema,
        output_jsonl_path=output_jsonl_path, implicit_rubrics=implicit_rubrics
    ))


if __name__ == "__main__":
    main()
