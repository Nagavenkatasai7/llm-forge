#!/usr/bin/env python3
"""Debug OOM issue on Hopper A100 with TRL 0.29."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model_name = "unsloth/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Patch chat template with {% generation %} markers (like our finetuner)
tokenizer.chat_template = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set header = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' %}"
    "{% if loop.first %}{{ bos_token + header }}{% else %}{{ header }}{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] | trim }}<|eot_id|>{% endgeneration %}"
    "{% else %}{{ message['content'] | trim }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
print(f"[1] After model+LoRA: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Create dataset similar to real finance data
msgs = []
for i in range(200):
    msgs.append({"messages": [
        {"role": "system", "content": "You are a finance specialist AI assistant."},
        {"role": "user", "content": f"Question {i}: Explain the implications of quantitative easing on bond yields and inflation expectations in emerging markets. " * 3},
        {"role": "assistant", "content": f"Answer {i}: Quantitative easing impacts bond yields through several channels. First, direct purchases compress term premia. Second, portfolio rebalancing effects push investors into riskier assets. " * 5},
    ]})
train_dataset = Dataset.from_list(msgs)

sft_config = SFTConfig(
    output_dir="/tmp/test_v7",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    bf16=True,
    max_length=2048,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    assistant_only_loss=True,
    logging_steps=1,
    max_steps=3,
    report_to="none",
)

print("Creating SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
print(f"[2] After SFTTrainer: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"[2] Peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("Training 3 steps...")
result = trainer.train()
print(f"[3] After training: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"[3] Peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"Loss: {result.training_loss:.4f}")
print("SUCCESS")
