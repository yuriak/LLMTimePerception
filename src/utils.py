import json
import re
from pydantic import BaseModel
from typing import Dict
import yaml


# def extract_json_from_text(text: str) -> str:
#     """Extract JSON from text that might contain markdown or natural language."""
#     # Try to extract JSON from markdown code blocks
#     if "```json" in text:
#         parts = text.split("```json")
#         json_part = parts[-1]
#         json_text = json_part.split("```")[0]
#         return json_text.strip()
#     elif "```" in text:
#         parts = text.split("```")
#         if len(parts) >= 3:
#             return parts[1].strip()

#     # If no code blocks, try to find JSON by looking for { }
#     text = text.strip()
#     start_idx = text.find("{")
#     if start_idx != -1:
#         count = 0
#         in_string = False
#         escape_char = False

#         for i in range(start_idx, len(text)):
#             char = text[i]

#             if char == "\\" and not escape_char:
#                 escape_char = True
#                 continue

#             if char == '"' and not escape_char:
#                 in_string = not in_string

#             if not in_string:
#                 if char == "{":
#                     count += 1
#                 elif char == "}":
#                     count -= 1
#                     if count == 0:
#                         return text[start_idx : i + 1]

#             escape_char = False

#     return text

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain markdown, natural language, or math expressions."""
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        parts = text.split("```json")
        json_part = parts[-1]
        json_text = json_part.split("```")[0]
        return json_text.strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()

    # If no code blocks, try to find JSON by scanning from the end
    text = text.strip()
    end_idx = text.rfind("}")
    if end_idx != -1:
        count = 0
        in_string = False
        escape_char = False

        for i in range(end_idx, -1, -1):
            char = text[i]

            if char == "\\" and not escape_char:
                escape_char = True
                continue

            if char == '"' and not escape_char:
                in_string = not in_string

            if not in_string:
                if char == "}":
                    count += 1
                elif char == "{":
                    count -= 1
                    if count == 0:
                        return text[i : end_idx + 1]

            escape_char = False

    return text



def validate_and_parse_json_output(output: str, dclass: BaseModel = None, remove_escape=False) -> Dict:
    """Try to parse and validate JSON output with enhanced error handling."""
    if not output:
        return None

    # First, try to extract JSON from text
    if remove_escape:
        output = output.replace("\\", "")
    json_str = extract_json_from_text(output)

    # List of potential fixes to try
    fixes = [
        lambda x: x,  # try original string
        lambda x: x.replace('\\"', '"'),  # fix escaped quotes
        lambda x: x.replace("'", '"'),  # replace single quotes with double quotes
        lambda x: x.replace("\n", "").replace("\r", ""),  # remove newlines
        lambda x: x.strip('"`'),  # remove any remaining markdown-style quotes
    ]

    # Try each fix until one works
    for fix in fixes:
        try:
            fixed_str = fix(json_str)
            result = json.loads(fixed_str)
            # Validate against schema
            if dclass:
                validated_result = dclass.model_validate(result)
                return validated_result.model_dump()
            return result
        except Exception:
            continue

    return None


def post_process_output(output):

    special_texts = [
        "<|start_header_id|>assistent<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>",
        "<|im_end|>",
    ]

    for special_text in special_texts:
        output = output.replace(special_text, "").strip()

    return output


def dict_to_markdown_yaml(data, wrap_in_code_block: bool = True) -> str:
    yaml_str = yaml.dump(data, sort_keys=False, allow_unicode=True, width=float("inf"))
    if wrap_in_code_block:
        return f"```yaml\n{yaml_str}\n```"
    return yaml_str