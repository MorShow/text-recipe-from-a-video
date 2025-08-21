import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
client = OpenAI()

project_root = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROMPT_TEMPLATE = """You are an expert at extracting structured triplets (action, noun, target) 
from cooking instructions.

Rules:
- action = the verb (main thing being done)
- noun = the object directly affected
- target = the location or indirect object (if none, return null)
- If there are multiple actions, return multiple triplets.
- Use singular nouns if possible.
- Return ONLY a JSON list of objects.

Example:
Sentence: "Add oil to a pan and spread it well so as to fry the bacon"
Output:
[
  {"action": "add", "noun": "oil", "target": "pan"},
  {"action": "spread", "noun": "oil", "target": "pan"},
  {"action": "fry", "noun": "bacon", "target": null}
]

Sentence: "{sentence}"
Output:
"""


def extract_triplets(sentence: str):
    # print(sentence)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured cooking action triplets."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)}
        ],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()

    try:
        triplets = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"JSON parse error for: {sentence}")
        triplets = []
    return triplets


def process_file(in_path: str | Path):
    out_path = project_root / "data" / "processed" / "youcookii_annotations_processed.jsonl"

    with open(in_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            print(line)
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
                "annotations": triplets,
                "duration": info.get("duration"),
                "subset": info.get("subset"),
                "recipe_type": info.get("recipe_type"),
                "video_url": info.get("video_url")
            }

            with open(out_path, "w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                logging.info(f"Preprocessed file saved to {out_path}")


if __name__ == "__main__":
    in_path = project_root / "data" / "processed" / "youcookii_annotations_extracted.jsonl"
    process_file(in_path)
