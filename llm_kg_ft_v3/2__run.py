import os
import os
# 完全禁用wandb
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
from accelerate import Accelerator

model_path = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/Qwen3-1.7B/"  # 根据实际路径改
dataset = load_dataset(
    "json", 
    data_files="/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/df_sample_full.jsonl", 
    split="train"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    texts = [
        f"{inst}\n{inp}\n{out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=False,        # 不 pad，Trainer 会动态 pad
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

import torch
# 配置量化参数，适合16GB显存
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 使用GPU和量化加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配到可用的GPU
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4, lora_alpha=8, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=64, lora_alpha=128, lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj"]
# )
# model = get_peft_model(model, lora_config)

# 配置训练参数，优化显存使用
training_args = TrainingArguments(
    output_dir="./lora-ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # 增加梯度累积步数减少显存占用
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,      # 开启fp16加速训练
    bf16=False,     # 不使用bf16
    optim="paged_adamw_32bit",  # 使用分页优化器减少内存占用
    gradient_checkpointing=True,  # 启用梯度检查点节省显存
    save_steps=500,
    save_total_limit=2,  # 限制保存的checkpoint数量
    report_to="none"  # 禁用wandb
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/lora-ckpt/final")