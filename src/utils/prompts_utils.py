import json
import re

def fill_prompt(prompt: str, fields: list, context: dict):
    for field in fields:
        if field not in context:
            raise ValueError(f"Field {field} not found in context")

        prompt = prompt.replace('{' + f'{field}' + '}', str(context[field]))
    return prompt


def read_prompts(key: str, delimiter: str = r'\[{}\]', context: dict = None, file_path: str=None, prompt_template: str=None) -> str:
    assert file_path is not None or prompt_template is not None, "Either file_path or prompt_template must be provided"
    assert not (file_path is not None and prompt_template is not None), "Only one of file_path or prompt_template should be provided"

    if context is None:
        context = dict()
    if file_path is not None:
        with open(file_path, 'r') as file:
            content = file.read()
    else:
        content = prompt_template

    key = key.strip().lower()

    # Match everything from the key until the next delimiter or end of file
    assert '(' not in key and ')' not in key, "Key should not contain parentheses"
    pattern = rf"{delimiter.format(key)}\s*?fields\s*:(.*?)\n(.*?)(?={delimiter.format(r'.*?')}|\Z)"
    pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)

    if not matches:
        raise ValueError(f"Key {key} not found in prompts file {file_path}")

    assert len(matches) == 1, f"Key {key} not found in prompts file {file_path}"
    assert len(matches[0]) == 2, f'Pattern match failed: {matches[0]}'

    fields_line, text = matches[0][0], matches[0][1]
    fields = [x.strip().lower() for x in fields_line.split(',')]
    fields = [x for x in fields if x]  # Remove empty strings
    prompt = text.strip()

    filled_prompt = fill_prompt(prompt, fields, context)

    return filled_prompt


def extract_required_fields(json_schema: dict) -> list:
    required_fields = []

    def traverse(schema):
        if 'required' in schema:
            required_fields.extend(schema['required'])
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                traverse(value)
        if 'items' in schema:
            traverse(schema['items'])

    traverse(json_schema)
    return required_fields
