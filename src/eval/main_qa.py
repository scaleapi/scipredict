import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv
import time
import logging
import re
import litellm
import traceback

from src.api_client.api_conf import APIConfiguration
from src.api_client.client import LLMClient

from src.utils.prompts_utils import read_prompts
from src.utils.llm_utils import get_response_content
from src.utils.conf_utils import load_config_with_subset

import asyncio
import aiofiles

FIELDS_OF_INTEREST = [x.lower() for x in
                      ['TASK', 'DOMAIN', 'PQ_FORMAT', 'FIELD', 'EXPERIMENTAL_SETUP', 'MEASUREMENT_TAKEN',
                       'OUTCOME_PREDICTION_QUESTION', 'REQUIRED_BACKGROUND_KNOWLEDGE', 'TITLE', 'URL','PDF_FILE','PUBLISHING_DATE']]


def is_valid_mcq_response(answer: str):
    if re.match(r'^\s*[\w\d][\.\)]?(\s*,\s*[\w\d][\.\)]?)*\s*$', answer.strip()):
        return True
    return False

def is_valid_numerical_response(answer: str):
    try:
        answer = re.match(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[.\n]*', answer.strip()).group(1)
        float(answer)
        return True
    except ValueError:
        return False


async def process_row(row, llm_client, prompt_template):
    retries = 5
    context = row.to_dict()
    is_mcq, is_numerical = False, False
    match context['pq_format'].strip().lower():
        case 'mcq':
            context['pq_format'] = 'Multiple-Choice'
            is_mcq = True
        case 'numerical values':
            context['pq_format'] = 'Numerical'
            is_numerical = True
        case 'free-format question':
            context['pq_format'] = 'Free-Form'
    sys_prompt = read_prompts(prompt_template=prompt_template, key='SYS', context=context)
    user_prompt = read_prompts(prompt_template=prompt_template, key='USER', context=context)

    final_result = None
    while retries > 0:
        retries -= 1
        try:
            response = await asyncio.to_thread(llm_client, system=sys_prompt, query=user_prompt, row=row)

            result_json = get_response_content(response, to_json=True)
            if is_mcq:
                assert is_valid_mcq_response(result_json['answer']), \
                    f"MCQ answer is not valid: {result_json['answer']}"

            if is_numerical:
                assert is_valid_numerical_response(result_json['answer']), \
                    f"Numerical answer is not valid: {result_json['answer']}"

            full_result = row.to_dict()
            full_result.update(result_json)
            final_result = full_result
            print(f"Successfully processed row {row.name}")
            break
        except litellm.RateLimitError as e:
            wait_time = 10
            retries += 1  # Don't count rate limit towards retries
            print(f"Rate limit error for row {row.name}. Waiting for {wait_time} seconds before retrying...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            print(f"Error processing row {row.name}: {e}. Retrying...")
            traceback.print_exc()
            error_result = row.to_dict()
            error_result['error'] = str(e)
            final_result = error_result

    return final_result


async def process_rows(df, llm_client, prompt_template, output_jsonl_path):
    tasks = [process_row(row, llm_client, prompt_template) for _, row in
             tqdm(df.iterrows(), total=len(df), desc="Submitting tasks")]
    print(f"Results will be progressively saved to {output_jsonl_path}")

    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task

        # Append new result to the file
        async with aiofiles.open(output_jsonl_path, 'a') as jsonl_file:
            await jsonl_file.write(json.dumps(result) + '\n')


def main() -> None:
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../../configs/conf.yaml'))
    assert 'sub_confs' in cfg, "sub_confs key not found in the main config."
    curr_subconf_key = ''.join(os.path.basename(__file__).split('.')[:-1])
    assert curr_subconf_key in cfg.sub_confs, f"{curr_subconf_key} key not found in sub_confs."
    for key, path in cfg.sub_confs[curr_subconf_key].items():
        cfg = load_config_with_subset(cfg, key, path)

    # Merge command-line arguments into the final configuration
    # breakpoint()
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    print(f"Using config: {cfg}")

    load_dotenv()
    logging.getLogger().setLevel(logging.DEBUG)
    for logger_name in logging.root.manager.loggerDict:
        print(f'Muting logs from {logger_name}')
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(cfg.output_dir, cfg.model.model_provider, cfg.run.key,
                                f"predictions-{(lambda s:(s.split('.v')[0].split('.')[-1]+'_v'+s.split('.v')[1].split('-')[0]) if ('.v' in s and s.split('.v')[1][:1].isdigit()) else (lambda last:'_'.join(last.split('-')[:next((i+1 for i,t in enumerate(last.split('-')) if t.endswith('b') and t[:-1].isdigit()),len(last.split('-')))]))(s.split('.')[-1].replace('/','-').replace(' ','-')))(cfg.model.model_name.split(':')[0]).replace('_v3p2','_v3')}-{timestamp}.jsonl")
    if not os.path.exists(os.path.basename(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Output will be saved to: {output_path}")

    assert os.environ.get('API_KEYS'), "API_KEYS environment variable not set. Please create a .env file."

    df = pd.read_csv(cfg.data.main)
    print(f"Loaded {len(df)} rows from {cfg.data.main}")

    df.columns = [col.lower() for col in df.columns]
    assert set(FIELDS_OF_INTEREST).issubset(set(df.columns)), \
        f"Input data must contain the following fields: {FIELDS_OF_INTEREST}"
    df = df[FIELDS_OF_INTEREST]

    if cfg.run.limit:
        df = df.iloc[:cfg.run.limit]

    api_conf = APIConfiguration(
        model_name=cfg.model.model_name,
        model_provider=cfg.model.model_provider,
        api_base=cfg.model.api_base,
        use_cache=cfg.model.use_cache,
    )

    with open(cfg.run.prompts[cfg.run.key], 'r') as f:
        prompt_template = f.read()

    schema_path = cfg.run.schemas[cfg.run.key]
    with open(schema_path, 'r') as f:
        output_schema = json.load(f)

    llm_client = LLMClient(
        api_conf=api_conf,
        output_schema=output_schema,
        temperature=cfg.model.temperature
    )

    asyncio.run(process_rows(
        df=df,
        llm_client=llm_client,
        prompt_template=prompt_template,
        output_jsonl_path=output_path
    ))


if __name__ == "__main__":
    main()
