import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import time
import os
import json
from dotenv import load_dotenv
import logging
import asyncio

from src.api_client.api_conf import APIConfiguration
from src.api_client.client import LLMClient
from src.utils.prompts_utils import read_prompts
from src.utils.llm_utils import get_response_content
from src.utils.conf_utils import load_config_with_subset


def extract_required_fields(schema, parent_key=""):
    """
    Recursively extract all required fields from the schema.
    """
    required_fields = []
    if "required" in schema:
        for field in schema["required"]:
            full_key = f"{parent_key}.{field}" if parent_key else field
            required_fields.append(full_key)
            if "properties" in schema.get("properties", {}).get(field, {}):
                required_fields.extend(
                    extract_required_fields(schema["properties"][field], full_key)
                )
    return required_fields


def validate_response(schema_required_fields, response):
    missing_or_empty_fields = []
    out = True
    for field in schema_required_fields:
        keys = field.split(".")
        value = response
        for key in keys:
            if key not in value:
                missing_or_empty_fields.append(field)
                out = False
            value = value[key]
        if value == "":
            missing_or_empty_fields.append(field)
            out = False
    if missing_or_empty_fields:
        print(f"Missing or empty required fields: {missing_or_empty_fields}")

    return out


async def process_row(row, llm_client, prompt_template, schema_required_fields):
    sys_prompt = read_prompts(prompt_template=prompt_template, key='SYS', context=row)
    user_prompt = read_prompts(prompt_template=prompt_template, key='USER', context=row)

    retries = 3
    while retries > 0:
        retries -= 1
        try:
            response = await asyncio.to_thread(llm_client, system=sys_prompt, query=user_prompt)
            result_json = get_response_content(response, to_json=True)

            if not validate_response(schema_required_fields, result_json):
                raise ValueError(f"Response validation failed against the output schema.")

            output_data = row.to_dict()
            output_data['error_analysis'] = result_json

            print(f"Successfully analyzed task {row.get('task', row.name)}")
            return output_data
        except Exception as e:
            print(f"Error analyzing task {row.get('task', row.name)}: {e}. Retrying...")
            if retries == 0:
                error_data = row.to_dict()
                error_data['error'] = str(e)
                return error_data
    return None


async def process_dataframe(df, llm_client, prompt_template, output_path, schema_required_fields):
    tasks = [process_row(row, llm_client, prompt_template, schema_required_fields) for _, row in df.iterrows()]

    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing Errors"):
        with open(output_path, 'a') as f:
            result = await completed_task
            if result:
                f.write(json.dumps(result, default=str) + '\n')


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

    # This pipeline requires an input file with model predictions
    if not cfg.run.get('input_path'):
        raise ValueError(
            "Configuration error: `run.input_path` must be set to the JSONL file containing model predictions.")

    print(f"Loading predictions from: {cfg.run.input_path}")
    df_preds = pd.read_json(cfg.run.input_path, lines=True)

    # Load main data to merge ground truth and other context
    df_main = pd.read_csv(cfg.data.main)
    df_main.columns = df_main.columns.str.lower()

    # Ensure task columns are the same type for merging
    df_preds['task'] = df_preds['task'].astype(str)
    df_main['task'] = df_main['task'].astype(str)

    # Merge the necessary context into the predictions dataframe
    context_cols = ['task', 'clean_gta', 'experimental_setup', 'measurement_taken', 'required_background_knowledge']
    df = pd.merge(df_main[context_cols], df_preds, on='task', how='inner', suffixes=('', '_y'))

    if cfg.run.limit:
        df = df.head(cfg.run.limit)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(cfg.output_dir, cfg.model.model_provider, cfg.run.key,
                               f"{cfg.model.model_name.replace('-', '_').replace('/', '_').replace(' ', '_')}-{timestamp}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(cfg.run.prompts[cfg.run.key], 'r') as f:
        prompt_template = f.read()
    with open(cfg.run.schemas[cfg.run.key], 'r') as f:
        output_schema = json.load(f)

    schema_required_fields = extract_required_fields(output_schema)
    print(f"Extracted required fields from output schema: {schema_required_fields}")

    api_conf = APIConfiguration(
        model_name=cfg.model.model_name,
        model_provider=cfg.model.model_provider,
        api_base=cfg.model.api_base,
        use_cache=cfg.model.use_cache,
    )

    llm_client = LLMClient(api_conf=api_conf, output_schema=output_schema, temperature=0.0)

    print(f"Input data has {len(df)} rows. Output will be saved to {output_path}")
    asyncio.run(process_dataframe(df, llm_client, prompt_template, output_path, schema_required_fields))
    print("Error analysis pipeline finished.")


if __name__ == "__main__":
    main()
