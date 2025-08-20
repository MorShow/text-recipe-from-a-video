import json

import torch
from datasets import Dataset
from transformers import (T5ForConditionalGeneration, T5TokenizerFast,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)


class LLMAnnotationExtractor:

    PREFIX = """
            You have an input sentence describing some cooking action. 
            Your task is extracting triplets in JSON format like {action: [ACTION], noun: [NOUN], target: [TARGET]}.
            Input sentence:
    """

    def __init__(self) -> None:
        # self.model_name = "google/flan-t5-large"
        self.model_name = "google/flan-t5-base"
        self._train_data = self.load_jsonl("train.jsonl")
        self._val_data = self.load_jsonl("val.jsonl")
        self._tokenizer = T5TokenizerFast.from_pretrained(self.model_name)

    def load_jsonl(self, path: str) -> Dataset:
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        return Dataset.from_list(rows)

    def preprocess(self, batch: Dataset) -> Dataset:
        input = [self.PREFIX + x for x in batch["input"]]
        model_tokens = self._tokenizer(input, max_length=128, truncation=True)
        target_tokens = self._tokenizer(batch["target"], max_length=64, truncation=True)
        model_tokens["target"] = target_tokens["input_ids"]
        return model_tokens

    def train(self) -> None:
        train_tokens = self._train_data.map(self.preprocess,
                                            batched=True,
                                            remove_columns=self._train_data.column_names)

        val_tokens = self._val_data.map(self.preprocess,
                                        batched=True,
                                        remove_columns=self._val_data.column_names)

        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        collator = DataCollatorForSeq2Seq(self._tokenizer, model=model)

        args = TrainingArguments(
            output_dir="data/processed",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=3e-4,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            predict_with_generate=True,
            generation_max_length=64,
            bf16=True if torch.cuda.is_available() else False  # set True if your GPU supports BF16
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tokens,
            eval_dataset=val_tokens,
            tokenizer=self._tokenizer,
            data_collator=collator,
        )

        trainer.train()
        trainer.save_model("models/triplet_extractor/final")
        self._tokenizer.save_pretrained("models/triplet_extractor/final")
