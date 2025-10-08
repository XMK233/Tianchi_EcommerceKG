import os
os.environ["WANDB_DISABLED"] = "true"

# 增加内存碎片整理配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 尝试自动检测CUDA路径，如果失败则使用CPU
import torch
import numpy as np
  # 添加numpy导入
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

model_path = "/mnt/d/HuggingFaceModels"
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
    max_grad_norm=1.0
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
        
        # # 定期打印GPU内存使用情况
        # if self.state.global_step % 50 == 0 and has_cuda:
        #     allocated = torch.cuda.memory_allocated() / (1024**3)
        #     reserved = torch.cuda.memory_reserved() / (1024**3)
        #     print(f"GPU内存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        return (loss, outputs) if return_outputs else loss

# 在GPU上使用自定义Trainer
trainer = CustomTrainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
# 确保保存目录存在
save_dir = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/lora-ckpt/final"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
print(f"模型已保存到: {save_dir}")

# ===============================================================
# 以下是获取embedding的代码
# ===============================================================

def get_text_embedding(text, model, tokenizer, device, max_length=256):
    """
    获取文本的embedding向量
    
    参数:
        text (str): 要获取embedding的文本
        model: 加载的模型
        tokenizer: 分词器
        device: 运行设备
        max_length: 最大序列长度
    
    返回:
        numpy.ndarray: 文本的embedding向量
    """
    # 分词处理
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 将输入移至指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 设置模型为评估模式
    model.eval()
    
    # 禁用梯度计算以提高效率
    with torch.no_grad():
        # 获取模型输出
        outputs = model(**inputs, output_hidden_states=True)
        
        # 获取最后一层隐藏状态
        # 通常使用最后一层的平均池化作为文本的embedding
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_size]
        
        # 计算所有token的平均值作为句子embedding
        # 可以根据需求选择不同的池化策略
        sentence_embedding = torch.mean(last_hidden_state, dim=1)  # [batch_size, hidden_size]
        
        # 如果需要，也可以使用[CLS]标记的输出作为embedding
        # cls_embedding = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 处理BFloat16类型，先转换为float32再转为numpy
        if sentence_embedding.dtype == torch.bfloat16:
            sentence_embedding = sentence_embedding.to(torch.float32)
        
        # 转换为numpy数组并移至CPU
        sentence_embedding = sentence_embedding.cpu().numpy()
        
    return sentence_embedding[0]  # 返回单个样本的embedding

def load_model_for_embedding(model_path, use_quantization=True):
    """
    加载训练好的模型用于获取embedding
    
    参数:
        model_path (str): 模型路径
        use_quantization (bool): 是否使用量化
    
    返回:
        tuple: (model, tokenizer, device)
    """
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化参数
    model_kwargs = {
        "cache_dir": model_path,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True
    }
    
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs["quantization_config"] = quantization_config
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        **model_kwargs
    )
    
    # 加载lora权重
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, save_dir)
    
    # 移动模型到设备
    model.to(device)
    
    return model, tokenizer, device

# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_text = "这是一个测试文本，用于获取embedding。"
    
    try:
        # 加载模型
        print("正在加载模型...")
        model, tokenizer, device = load_model_for_embedding(model_path)
        
        # 获取embedding
        print("正在获取embedding...")
        embedding = get_text_embedding(sample_text, model, tokenizer, device)
        
        # 打印结果
        print(f"获取的embedding形状: {embedding.shape}")
        print(f"embedding示例值: {embedding[:10]}")
        
        # 释放内存
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    except Exception as e:
        print(f"获取embedding时出错: {e}")