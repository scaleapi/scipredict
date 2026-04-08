import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import time
import os
import json
from dotenv import load_dotenv
import logging
import asyncio
from copy import deepcopy

from src.api_client.api_conf import APIConfiguration
from src.api_client.client import LLMClient
from src.utils.prompts_utils import read_prompts
from src.utils.llm_utils import get_response_content
from src.utils.conf_utils import load_config_with_subset


async def process_row(row, llm_client, prompt_template):
    context = row.to_dict()
    sys_prompt = read_prompts(prompt_template=prompt_template, key='SYS', context=context)
    user_prompt = read_prompts(prompt_template=prompt_template, key='USER', context=context)

    retries = 3
    while retries > 0:
        retries -= 1
        try:
            response = await asyncio.to_thread(llm_client, system=sys_prompt, query=user_prompt)
            result_json = get_response_content(response, to_json=True)

            output_data = row.to_dict()
            output_data.update(result_json)

            print(f"Successfully converted MCQ for task {row['task']}")
            return output_data
        except Exception as e:
            print(f"Error processing task {row['task']}: {e}. Retrying...")
            if retries == 0:
                error_data = row.to_dict()
                error_data['error'] = str(e)
                return error_data
    return None


async def process_dataframe(df, llm_client, prompt_template, output_path):
    tasks = [process_row(row, llm_client, prompt_template) for _, row in df.iterrows()]

    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Converting MCQs"):
        with open(output_path, 'a') as f:
            result = await completed_task
            if result:
                f.write(json.dumps(result, default=str) + '\n')


def main()-> None:
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

    logging.getLogger("litellm").setLevel(logging.WARNING)
    load_dotenv()

    df = pd.read_csv(cfg.data.main)
    df.columns = df.columns.str.lower()

    print("Filtering for MCQ questions only...")
    df = deepcopy(df[df['pq_format'].str.strip().str.lower() == 'mcq'])
    if df.empty:
        raise ValueError("No MCQ questions found in the input file.")

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

    api_conf = APIConfiguration(
        model_name=cfg.model.model_name,
        model_provider=cfg.model.model_provider,
        api_base=cfg.model.api_base,
        use_cache=cfg.model.use_cache,
    )

    llm_client = LLMClient(api_conf=api_conf, output_schema=output_schema, temperature=cfg.model.temperature)

    print(f"Input data has {len(df)} rows. Output will be saved to {output_path}")
    asyncio.run(process_dataframe(df, llm_client, prompt_template, output_path))
    print("Pipeline finished.")


if __name__ == "__main__":
    main()