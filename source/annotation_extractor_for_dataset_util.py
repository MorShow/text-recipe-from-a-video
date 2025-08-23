import json
import logging
from pathlib import Path
from dotenv import load_dotenv

import ollama
from openai import OpenAI

load_dotenv()
client = OpenAI()

project_root = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROMPT_TEMPLATE = """
    You are an expert at extracting structured triplets (action, noun, target) 
    from cooking instructions.
    
    Examples:
    Sentence: "Add oil to a pan and spread it well so as to fry the bacon"
    Output:
    [
      {{"action": "add", "noun": "oil", "target": "pan"}},
      {{"action": "spread", "noun": "oil", "target": "pan"}},
      {{"action": "fry", "noun": "bacon", "target": null}}
    ]
    
    Sentence: "Spread margarine on bread"
    Output:
    [
      {{"action": "spread", "noun": "margarine", "target": "bread"}}
    ]
    
    Please, output without any redundant characters: Only [{{...}}, {{...}}, {{...}}].
    Output a single-line JSON array, no line breaks or spaces except after commas.
    Return ONLY a JSON array of objects {{action,noun,target}}. Use null when missing. No markdown.
    
    Sentence: {sentence}
"""


def extract_triplets(sentence: str):
    response = ollama.chat(
        model="gemma3:4b-it-qat",
        messages=[
            {"role": "system",
             "content": "Return ONLY a JSON array of objects {action,noun,target}. Use null when missing. No markdown."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)}
        ],
        options={"temperature": 0}
    )
    text = response["message"]["content"].strip()

    try:
        if '```json' in text:
            text = text.split('\n')[1]
        print(text)
        triplets = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"JSON parse error for: {sentence}")
        triplets = []
    return triplets


def process_file(in_path: str | Path, out_path: str | Path):
    result = []

    with open(in_path, "r", encoding="utf-8") as file:
        text = file.read()
        text = text[text.index('[') + 1:text.rindex(']')]

        for line in text.split('}},'):
            line += '}}'
            line = line.strip()

            if not line or '{' not in line or '}' not in line:
                continue

            info = json.loads(line)
            sentence = info["sentence"]
            triplets = extract_triplets(sentence)
            # print(triplets)

            new_entry = {
                "video_id": info.get("video_id"),
                "segment_id": info.get("segment_id"),
                "sentence": sentence,
                "annotations": triplets if triplets else info["annotations"]
            }
            result.append(new_entry)

    with open(out_path, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(result, ensure_ascii=False))
        logging.info(f"Preprocessed file saved to {out_path}")


if __name__ == "__main__":
    in_path = project_root / "data" / "processed" / "youcookii_annotations_small.jsonl"
    out_path = project_root / "data" / "processed" / "youcookii_annotations_small_processed.jsonl"
    process_file(in_path, out_path)
