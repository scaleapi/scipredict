import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import time
import os
import json
import re
from dotenv import load_dotenv
import logging
import asyncio
import hashlib

from src.api_client.api_conf import APIConfiguration
from src.api_client.client import LLMClient
from src.utils.prompts_utils import read_prompts
from src.utils.llm_utils import get_response_content
from src.utils.conf_utils import load_config_with_subset


def parse_hashed_bkg(text: str) -> dict:
    if not isinstance(text, str): return {}
    pattern = r'-\s*([a-f0-9]{10}):\s*(.*)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return {h.strip(): t.strip() for h, t in matches}


async def process_row(row, key, llm_client, prompt_template):
    retries = 5

    context = row.to_dict()
    sys_prompt = read_prompts(prompt_template=prompt_template, key='SYS', context=context)
    user_prompt = read_prompts(prompt_template=prompt_template, key='USER', context=context)

    final_result = None
    while retries > 0:
        retries -= 1
        try:
            response = await asyncio.to_thread(llm_client, system=sys_prompt, query=user_prompt)
            result_json = get_response_content(response, to_json=True)
            full_result = row.to_dict()
            full_result.update(result_json)
            content = result_json[key]
            if key == 'generate_bkg':
                full_result['generated_bkg'] = "\n".join([f"- {item}" for item in content])
                full_result['generated_bkg_hashed'] = "\n".join(
                    [f"- {hashlib.sha3_256(item.encode('utf-8')).hexdigest()[:10]}: {item}" for item in content])
            elif key == 'bkg_to_qa':
                input_hashes = set(re.findall(r'- (\w{10}):', row['required_background_knowledge_hashed']))
                output_hashes = {item['original_hash'] for item in content}
                if input_hashes != output_hashes:
                    raise ValueError(f"Hash ID mismatch. Input:{input_hashes}, Output:{output_hashes}")
            elif key == 'answer_bkg_qa':
                question_hashes = set([x['original_hash'] for x in row['bkg_to_qa']])
                answer_hashes = set([x['original_hash'] for x in content])
                assert question_hashes == answer_hashes, f"Hash ID mismatch. Questions:{question_hashes}, Answers:{answer_hashes}"
            elif key == 'judge_bkg_qa':
                question_hashes = set([x['original_hash'] for x in row['answer_bkg_qa']])
                judged_hashes = set([x['original_hash'] for x in content])
                assert question_hashes == judged_hashes, f"Hash ID mismatch. Questions:{question_hashes}, Judged:{judged_hashes}"
            else:
                raise NotImplementedError(f"Unknown key for single-step processing: {key}")

            final_result = full_result
            print(f"Successfully processed task {row['task']}")
            break
        except Exception as e:
            print(f"Error processing task {row['task']}: {e}")
            error_data = row.to_dict()
            error_data['error'] = str(e)

    return final_result


async def process_dataframe(df, llm_client, prompt_template, cfg, output_path):
    tasks = [process_row(row, cfg.run.key, llm_client, prompt_template) for _, row in df.iterrows()]

    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing rows"):
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
    logging.getLogger("litellm").setLevel(logging.WARNING)

    if cfg.run.key == 'answer_bkg_qa' or cfg.run.key == 'judge_bkg_qa':
        df = pd.read_json(cfg.run.input_path, lines=True)
    else:
        df = pd.read_csv(cfg.data.main)
    df.columns = df.columns.str.lower()
    if cfg.run.limit: df = df.head(cfg.run.limit)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(cfg.output_dir, cfg.model.model_provider, cfg.run.key,
                               f"{(lambda s: (s.split('.v')[0].split('.')[-1] + '_v' + s.split('.v')[1].split('-')[0]) if ('.v' in s and s.split('.v')[1][:1].isdigit()) else (lambda last: '_'.join(last.split('-')[:next((i + 1 for i, t in enumerate(last.split('-')) if t.endswith('b') and t[:-1].isdigit()), len(last.split('-')))]))(s.split('.')[-1].replace('/', '-').replace(' ', '-')))(cfg.model.model_name.split(':')[0]).replace('_v3p2', '_v3')}-{timestamp}.jsonl")
    if cfg.run.key == 'bkg_to_qa':
        output_path = os.path.join("data", cfg.run.key,
                                   f"{(lambda s: (s.split('.v')[0].split('.')[-1] + '_v' + s.split('.v')[1].split('-')[0]) if ('.v' in s and s.split('.v')[1][:1].isdigit()) else (lambda last: '_'.join(last.split('-')[:next((i + 1 for i, t in enumerate(last.split('-')) if t.endswith('b') and t[:-1].isdigit()), len(last.split('-')))]))(s.split('.')[-1].replace('/', '-').replace(' ', '-')))(cfg.model.model_name.split(':')[0]).replace('_v3p2', '_v3')}-{timestamp}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input data has {len(df)} rows. Output will be saved to {output_path}")

    api_conf = APIConfiguration(
        model_name=cfg.model.model_name,
        model_provider=cfg.model.model_provider,
        api_base=cfg.model.api_base,
        use_cache=cfg.model.use_cache,
    )

    with open(cfg.run.prompts[cfg.run.key], 'r') as f:
        prompt_template = f.read()
    with open(cfg.run.schemas[cfg.run.key], 'r') as f:
        output_schema = json.load(f)

    llm_client = LLMClient(
        api_conf=api_conf,
        output_schema=output_schema,
        temperature=cfg.model.temperature
    )

    asyncio.run(process_dataframe(
        df=df,
        llm_client=llm_client,
        prompt_template=prompt_template,
        cfg=cfg,
        output_path=output_path
    ))
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
