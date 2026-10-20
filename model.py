"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit = True,
    )
    return (model, tokenizer)

# Step 2 - count_total_parameters
import torch.nn as nn
def count_total_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return int(total)

# Step 3 - is_model_4bit_quantized
import torch.nn as nn
import bitsandbytes as bnb
def is_model_4bit_quantized(model):
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '</s>'
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return target_modules

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    if target_modules is None:
        target_modules = get_lora_target_modules()
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias='none',
        target_modules=target_modules
    )
    return model

# Step 7 - count_trainable_parameters
import torch.nn as nn
def count_trainable_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return float(trainable_count/total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    return [
        {
            "instruction": "What is a budget, and why is it important?",
            "response": "A budget is a plan for how you earn, spend, and save money. It helps you control expenses, avoid unnecessary debt, and work toward financial goals."
        },
        {
            "instruction": "How can I start saving money on a low income?",
            "response": "Track your expenses, prioritize essential needs, and set aside a small amount whenever you receive income. Automating savings and reducing recurring unnecessary expenses can help you build an emergency fund over time."
        },
        {
            "instruction": "Explain compound interest in simple terms.",
            "response": "Compound interest is interest earned on both your original investment and the interest accumulated over time. For example, if you invest ₹10,000 at 10% annual interest, compounded yearly, it grows to ₹12,100 after two years, assuming the rate stays constant and no money is withdrawn."
        },
        {
            "instruction": "What is the difference between stocks and bonds?",
            "response": "A stock represents ownership in a company, while a bond generally represents a loan to a company or government. Stocks can offer higher growth potential but often have greater volatility. Bonds typically pay interest and have different risks depending on the issuer and terms."
        },
        {
            "instruction": "What is a sensible way to manage credit card debt?",
            "response": "Pay at least the minimum due on every card to avoid late-payment penalties, and direct extra payments toward the card with the highest interest rate. Avoid adding new debt where possible and consider contacting your lender if payments become difficult."
        },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    formatted_egs = []
    for eg in examples:
        formatted_egs.append(format_instruction_example(eg))
    return formatted_egs

# Step 12 - build_text_dataset
from datasets import Dataset
def build_text_dataset(texts):
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    tokens = tokenizer(text)
    return tokens['input_ids']

# Step 14 - count_tokens
def count_tokens(input_ids):
    return len(input_ids)

# Step 15 - build_training_arguments
import unsloth
import torch
import transformers
def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    use_bf16 = torch.cuda.is_bf16_supported()
    args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
         gradient_accumulation_steps=1,
         max_steps=max_steps,
         learning_rate=learning_rate,
         output_dir=output_dir,
         bf16=use_bf16,
         fp16=not use_bf16,
         logging_steps=1,
         optim='adamw_8bit',
    )
    return args

# Step 16 - build_sft_trainer
from trl import SFTTrainer
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args = training_args
    )
    return trainer

# Step 17 - run_sft_training
from transformers.trainer_utils import IntervalStrategy

def run_sft_training(trainer):
    trainer.args.save_strategy = IntervalStrategy.NO
    trainer_stats = trainer.train()
    return float(trainer_stats.training_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    return FastLanguageModel.for_inference(model)

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):
    messages = [
        {"role": "user", "content": instruction}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# Step 20 - generate_reply (not yet solved)
# TODO: implement

