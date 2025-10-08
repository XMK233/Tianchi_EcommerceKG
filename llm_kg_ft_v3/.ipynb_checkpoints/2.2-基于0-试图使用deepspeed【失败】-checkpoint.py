import os
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

# 增加内存碎片整理配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 尝试自动检测CUDA路径，如果失败则使用CPU
import torch
try:
    # 尝试导入CUDA相关模块
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA可用，将使用GPU进行训练")
        # 打印GPU信息
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")
    else:
        print("CUDA不可用，将使用CPU进行训练")
except:
    has_cuda = False
    print("CUDA不可用，将使用CPU进行训练")

import pandas as pd, openai, random, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_path = "/mnt/d/HuggingFaceModels"
save_dir = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/lora-ckpt/final"

# 训练

memory_optim_args = {
    # 根据GPU显存调整参数
    "max_seq_length": 256,            # 减少序列长度节省内存
    "gradient_accumulation_steps": 8,  # 适当的梯度累积步数，平衡GPU利用率和内存
    "per_device_train_batch_size": 2,  # 增加批量大小提高GPU利用率
    "gradient_checkpointing": True,    # 启用梯度检查点节省显存
    "optim": "paged_adamw_32bit",     # 使用分页优化器
    "fp16": has_cuda,                 # 启用混合精度训练
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
}

dataset = load_dataset("json", data_files="/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/df_sample_full.jsonl", split="train")

# 确保数据集不为空
print(f"数据集大小: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("数据集为空，请检查数据文件路径是否正确")

# 查看数据集的前几个样本
sample = dataset[0]
print(f"数据集样本示例: {sample}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", cache_dir=model_path)
tokenizer.pad_token = tokenizer.eos_token

# 修改数据处理函数，确保正确设置标签
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
        max_length=memory_optim_args["max_seq_length"],  # 使用优化的序列长度节省内存
        padding="max_length",  # 使用max_length填充而不是动态填充
        return_tensors="pt"  # 提前返回PyTorch张量，减少内存转换开销
    )
    # 正确设置labels，确保梯度能够流动
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# 检查tokenized后的数据集格式
print(f"Tokenized后的数据格式: {list(dataset.features.keys())}")
print(f"Input IDs示例: {dataset[0]['input_ids'][:10]}...")
print(f"Labels示例: {dataset[0]['labels'][:10]}...")

# 配置量化参数，适合16GB显存
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # 使用float16而不是bfloat16，对某些显卡更友好
)
use_quantization = True

# 使用GPU和量化加载模型
model_kwargs = {
    "cache_dir": model_path,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    "device_map": "auto" if torch.cuda.is_available() else None,
    "trust_remote_code": True
}

if use_quantization:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    **model_kwargs
)

# 配置LoRA参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4, lora_alpha=8, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"  # 确保偏置不参与LoRA
)

model = get_peft_model(model, lora_config)

# 配置训练参数，平衡显存使用和GPU利用率
training_args = TrainingArguments(
    output_dir="./lora-ckpt",
    per_device_train_batch_size=memory_optim_args["per_device_train_batch_size"],
    gradient_accumulation_steps=memory_optim_args["gradient_accumulation_steps"],
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=memory_optim_args["fp16"],
    bf16=False,
    optim=memory_optim_args["optim"],
    gradient_checkpointing=memory_optim_args["gradient_checkpointing"],
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    # 添加以下参数解决梯度问题
    gradient_checkpointing_kwargs=memory_optim_args["gradient_checkpointing_kwargs"],
    # 添加性能监控
    logging_steps=10,
    # 启用梯度裁剪防止梯度爆炸
    max_grad_norm=1.0,
    # 禁用Hugging Face默认的分布式训练设置，由DeepSpeed接管
    ddp_find_unused_parameters=False,
    local_rank=-1  # 让DeepSpeed处理local_rank
)

# 自定义Trainer类以确保梯度正确传播并处理额外参数
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 忽略DeepSpeed可能传递的额外参数，解决'num_items_in_batch'参数错误
        # 确保inputs包含labels，并且labels是可微分的
        if "labels" not in inputs:
            raise ValueError("输入中缺少labels")
        
        # 确保labels是tensor并且在正确的设备上
        if not isinstance(inputs["labels"], torch.Tensor):
            inputs["labels"] = torch.tensor(inputs["labels"]).to(model.device)
        
        # 清理未使用的变量以释放内存
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 确保loss是标量并且可以微分
        if loss is None:
            # 如果模型没有自动计算loss，手动计算
            logits = outputs.logits
            labels = inputs["labels"]
            # 使用交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss()
            # 调整logits和labels的形状以匹配CrossEntropyLoss的要求
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


import deepspeed
import argparse

# 添加命令行参数解析，处理DeepSpeed传递的参数
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed training")
    # 添加DeepSpeed需要的参数
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # 添加其他可能的参数
    return parser.parse_args()

args = parse_args()

# 修改DeepSpeed配置以确保与transformers兼容
deepspeed_config = {
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": True,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 3,                  # ZeRO阶段（建议2或3）
    "offload_optimizer": {
      "device": "cpu",           # 优化器卸载到CPU
      "pin_memory": True
    },
    "allgather_partitions": True,
    "contiguous_gradients": True
  },
  "steps_per_print": 50,
  # 添加分布式训练配置
  "dist_backend": "nccl",
  "distributed": True
}
# deepspeed_config = {  
#   "train_batch_size": 16,  
#   "gradient_accumulation_steps": 4,  
#     "gradient_checkpointing": True,
#   "zero_optimization": {  
#     "stage": 3,  
#     "offload_optimizer": {  
#       "device": "cpu"  
#     }  
#   }  
# }

# 初始化DeepSpeed
# 注意：不需要手动启用gradient_checkpointing，DeepSpeed会处理
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    model=model,
    config=deepspeed_config,
    model_parameters=model.parameters(),
    training_data=dataset if dataset is not None else None,
    args=args  # 传递命令行参数
)

# 在GPU上使用自定义Trainer，注意这里使用model_engine
trainer = CustomTrainer(
    model=model_engine,
    args=training_args,
    train_dataset=dataset,
    optimizers=(optimizer, None),  # 使用DeepSpeed初始化的优化器
)

trainer.train()


# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 与DeepSpeed兼容的模型保存方式
# 检查是否在主进程上（避免多进程重复保存）
if not hasattr(model_engine, 'local_rank') or model_engine.local_rank == 0:
    print(f"正在保存模型到: {save_dir}")
    
    # 使用DeepSpeed的save_checkpoint方法保存完整模型
    if hasattr(model_engine, 'save_checkpoint'):
        # 保存DeepSpeed格式的检查点
        deepspeed_checkpoint_dir = os.path.join(save_dir, "deepspeed_checkpoint")
        os.makedirs(deepspeed_checkpoint_dir, exist_ok=True)
        model_engine.save_checkpoint(deepspeed_checkpoint_dir)
        print(f"DeepSpeed检查点已保存到: {deepspeed_checkpoint_dir}")
    
    # 保存适配Hugging Face格式的模型，便于后续加载和推理
    # try:
    # 对于PeftModel，我们需要保存基础模型和适配器
    if hasattr(model, 'save_pretrained'):
        # 尝试以普通方式保存模型
        trainer.save_model(save_dir)
        print(f"模型已保存到: {save_dir}")
    else:
        # 如果是DeepSpeed模型，尝试提取基础模型并保存
        if hasattr(model_engine, 'module'):
            base_model = model_engine.module
            # 如果是PeftModel，使用save_pretrained
            if hasattr(base_model, 'save_pretrained'):
                base_model.save_pretrained(save_dir)
                print(f"基础模型已保存到: {save_dir}")
    # except Exception as e:
    #     print(f"保存模型时出错: {e}")
    #     print("尝试使用替代方法保存模型...")
        
    #     # 替代保存方法：提取模型权重并手动保存
    #     try:
    #         # 获取模型状态字典
    #         if hasattr(model_engine, 'module'):
    #             state_dict = model_engine.module.state_dict()
    #         else:
    #             state_dict = model_engine.state_dict()
            
    #         # 保存模型权重
    #         torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
            
    #         # 保存分词器
    #         if 'tokenizer' in locals():
    #             tokenizer.save_pretrained(save_dir)
    #             print(f"分词器已保存到: {save_dir}")
                
    #         print(f"模型权重已保存到: {save_dir}")
    #     except Exception as inner_e:
    #         print(f"替代保存方法也失败: {inner_e}")