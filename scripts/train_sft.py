#!/usr/bin/env python3
"""LoRA/QLoRA SFT training for 2048 teacher chat JSONL data."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no} missing messages")
            rows.append(row)
    return rows


def encode_example(row: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("each SFT row must contain exactly user and assistant messages")
    prompt_text = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)) :]
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]
    attention_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorForCausalSFT:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_args(args: argparse.Namespace):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "seed": args.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    }
    parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def main() -> int:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("choose at most one of --bf16 and --fp16")
    if args.use_qlora:
        args.use_lora = True
    random.seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if args.use_qlora else None,
    )
    model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    train_dataset = Dataset.from_list([encode_example(row, tokenizer, args.max_seq_length) for row in train_rows])
    val_dataset = Dataset.from_list([encode_example(row, tokenizer, args.max_seq_length) for row in val_rows])

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCausalSFT(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = dict(train_result.metrics)
    metrics["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
