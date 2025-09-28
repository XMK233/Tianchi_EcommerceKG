from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

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
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.bfloat16,
    # device_map=None # 自动选 mps/cpu
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

training_args = TrainingArguments(
    output_dir="./lora-ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,      # 必须关
    bf16=False,      # 必须关
    # dataloader_pin_memory=False,   # 避免 MPS 自动回退
    # # 强制使用CPU
    # no_cuda=True    # 禁用CUDA以强制使用CPU
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/lora-ckpt/final")