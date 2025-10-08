import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
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

# 配置内存优化参数
def get_memory_optimization_args():
    """为16GB显存和32GB内存配置的内存优化参数"""
    return {
        # 根据GPU显存调整参数
        "max_seq_length": 256,            # 减少序列长度节省内存
        "gradient_accumulation_steps": 4,  # 减少梯度累积步数，提高GPU利用率
        "per_device_train_batch_size": 4,  # 增加批量大小提高GPU利用率
        "gradient_checkpointing": False,   # 禁用梯度检查点，提高训练速度
        "optim": "adamw_torch_fused",     # 使用融合优化器加速训练
        "fp16": has_cuda,                 # 启用混合精度训练
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": False,                    # 禁用bf16（除非GPU支持且有明确收益）
        "use_cache": False                # 禁用KV缓存以节省内存
    }

# 获取内存优化参数
memory_optim_args = get_memory_optimization_args()

model_path = "/mnt/d/HuggingFaceModels"
# dataset = load_dataset(
#     "json", 
#     data_files="/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/df_sample_full.jsonl", 
#     split="train"
# )
# 正确加载JSON Lines文件
# 方法1：使用load_dataset的jsonl加载功能
dataset = load_dataset(
    "json", 
    data_files="/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/df_sample_full.jsonl", 
    split="train"
)

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
    # 确保examples是正确的格式
    if not isinstance(examples, dict):
        raise ValueError(f"期望examples是字典类型，但得到了{type(examples)}")
    
    # 确保所有必要的键都存在
    required_keys = ["instruction", "input", "output"]
    for key in required_keys:
        if key not in examples:
            raise ValueError(f"examples中缺少键: {key}")
    
    # 检查数据类型和长度
    for key in required_keys:
        if not isinstance(examples[key], (list, np.ndarray, torch.Tensor)):
            print(f"警告: {key}不是列表或数组类型，而是{type(examples[key])}")
    
    # 确保所有字段长度相同
    lengths = [len(examples[key]) for key in required_keys]
    if len(set(lengths)) > 1:
        print(f"警告: 字段长度不一致: {dict(zip(required_keys, lengths))}")
    
    try:
        texts = [
            f"{inst}\n{inp}\n{out}"
            for inst, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
    except Exception as e:
        print(f"构建文本时出错: {e}")
        # 提供一个安全的回退方案
        batch_size = len(examples["instruction"]) if "instruction" in examples else 1
        texts = ["安全回退文本"] * batch_size
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=memory_optim_args["max_seq_length"],  # 使用优化的序列长度节省内存
        padding="max_length",  # 使用max_length填充而不是动态填充
        return_tensors="pt"  # 提前返回PyTorch张量，减少内存转换开销
    )
    
    # 正确设置labels，确保梯度能够流动
    try:
        # 确保input_ids是正确的格式
        if not isinstance(tokenized["input_ids"], torch.Tensor):
            tokenized["input_ids"] = torch.tensor(tokenized["input_ids"])
        
        # 克隆input_ids作为labels，并确保是整数类型
        tokenized["labels"] = tokenized["input_ids"].clone().long()
        
        # 检查labels的形状和类型
        print(f"分词后: labels形状={tokenized['labels'].shape}, 类型={tokenized['labels'].dtype}")
    except Exception as e:
        print(f"设置labels时出错: {e}")
        # 创建一个安全的占位符标签
        batch_size = tokenized["input_ids"].shape[0] if isinstance(tokenized["input_ids"], torch.Tensor) else 1
        seq_length = tokenized["input_ids"].shape[1] if isinstance(tokenized["input_ids"], torch.Tensor) and len(tokenized["input_ids"].shape) > 1 else memory_optim_args["max_seq_length"]
        tokenized["labels"] = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# 检查tokenized后的数据集格式
print(f"Tokenized后的数据格式: {list(dataset.features.keys())}")
# 确保在打印前将CUDA张量移至CPU
input_ids_sample = dataset[0]['input_ids'][:10]
labels_sample = dataset[0]['labels'][:10]
if isinstance(input_ids_sample, torch.Tensor) and input_ids_sample.device.type == 'cuda':
    input_ids_sample = input_ids_sample.cpu()
labels_sample = dataset[0]['labels'][:10]
if isinstance(labels_sample, torch.Tensor) and labels_sample.device.type == 'cuda':
    labels_sample = labels_sample.cpu()
print(f"Input IDs示例: {input_ids_sample}...")
print(f"Labels示例: {labels_sample}...")

# 根据实际环境选择是否使用量化
try:
    # 配置量化参数，适合16GB显存
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # 使用float16而不是bfloat16，对某些显卡更友好
    )
    use_quantization = True
except Exception as e:
    print(f"量化配置失败: {e}")
    print("将不使用量化")
    use_quantization = False

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
    num_train_epochs=10,
    learning_rate=2e-4,
    fp16=memory_optim_args["fp16"],
    bf16=memory_optim_args["bf16"],
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
    # 添加数据加载优化
    dataloader_num_workers=4,  # 使用多个进程加载数据
    dataloader_pin_memory=True,  # 固定内存以加速数据传输
    # 启用自动批处理大小查找（如果支持）
    auto_find_batch_size=False,
    
)

# 优化数据加载和预处理
def optimize_data_loader(dataset, batch_size):
    """优化数据加载器以提高GPU利用率"""
    from torch.utils.data import DataLoader
    import multiprocessing
    
    # 确定最佳的工作进程数
    num_workers = min(4, multiprocessing.cpu_count() - 1)
    
    # 创建优化的数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2  # 预加载更多批次
    )
    
    return data_loader

# 自定义Trainer类以确保梯度正确传播并处理额外参数
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 忽略DeepSpeed可能传递的额外参数，解决'num_items_in_batch'参数错误
        # 确保inputs包含labels，并且labels是可微分的
        if "labels" not in inputs:
            raise ValueError("输入中缺少labels")
        
        # 深度复制inputs以避免修改原始数据
        inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 确保labels是tensor并且在正确的设备上
        if not isinstance(inputs["labels"], torch.Tensor):
            try:
                # 更健壮的标签转换逻辑
                import numpy as np
                
                # 打印输入标签的类型和部分内容，以便调试
                print(f"尝试转换标签: 类型={type(inputs['labels'])}")
                
                # 根据不同的输入类型采用不同的转换策略
                if isinstance(inputs["labels"], list):
                    # 对于列表，先转换为numpy数组，再转为tensor
                    try:
                        np_labels = np.array(inputs["labels"])
                        # 确保数据是整数类型
                        if np_labels.dtype.kind not in {'i', 'u'}:
                            np_labels = np_labels.astype(np.int64)
                        inputs["labels"] = torch.tensor(np_labels).to(model.device)
                    except Exception as list_err:
                        print(f"列表转换失败: {list_err}")
                        # 作为最后的尝试，确保列表中的每个元素都是整数
                        clean_labels = [int(item) if not isinstance(item, int) else item for item in inputs["labels"]]
                        inputs["labels"] = torch.tensor(clean_labels).to(model.device)
                elif hasattr(inputs["labels"], '__array__'):
                    # 对于支持数组协议的对象
                    np_labels = np.asarray(inputs["labels"])
                    if np_labels.dtype.kind not in {'i', 'u'}:
                        np_labels = np_labels.astype(np.int64)
                    inputs["labels"] = torch.tensor(np_labels).to(model.device)
                else:
                    # 尝试直接转换
                    inputs["labels"] = torch.tensor(inputs["labels"], dtype=torch.long).to(model.device)
            except Exception as e:
                print(f"Error converting labels to tensor: {e}")
                print(f"标签数据类型: {type(inputs['labels'])}")
                # 作为最后的安全措施，创建一个占位符标签（实际应用中应避免）
                batch_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 1
                seq_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs and len(inputs['input_ids'].shape) > 1 else 1
                inputs["labels"] = torch.zeros((batch_size, seq_length), dtype=torch.long).to(model.device)
        else:
            # 确保labels在正确的设备上
            inputs["labels"] = inputs["labels"].to(model.device)
        
        # 检查是否有其他CUDA张量需要处理（避免numpy转换错误）
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                # 只处理需要转换为numpy的情况，但在当前上下文中不进行转换
                # 保留在CUDA设备上以加速计算
                pass
        
        # 确保labels是整数类型
        if inputs["labels"].dtype != torch.long:
            inputs["labels"] = inputs["labels"].long()
        
        # 检查并修复可能的索引问题
        if inputs["labels"].dim() > 1:
            # 如果labels是多维的，确保其形状正确
            inputs["labels"] = inputs["labels"].squeeze()
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 确保loss是标量并且可以微分
        if loss is None:
            try:
                # 如果模型没有自动计算loss，手动计算
                logits = outputs.logits
                labels = inputs["labels"]
                # 使用交叉熵损失
                loss_fct = torch.nn.CrossEntropyLoss()
                # 调整logits和labels的形状以匹配CrossEntropyLoss的要求
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            except Exception as e:
                print(f"Error computing loss: {e}")
                # 确保在打印前处理CUDA张量
                logits_shape = logits.shape
                labels_shape = labels.shape
                print(f"Logits shape: {logits_shape}")
                print(f"Labels shape: {labels_shape}")
                raise
        
        # 定期打印GPU内存使用情况和利用率
        if self.state.global_step % 20 == 0 and has_cuda:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            # 尝试获取GPU利用率（需要pynvml库）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                pynvml.nvmlShutdown()
                print(f"GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB | GPU利用率: {gpu_util}%")
            except Exception as e:
                print(f"GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        return (loss, outputs) if return_outputs else loss
        
    def training_step(self, model, inputs, *args, **kwargs):
        # 确保所有输入都在正确的设备上
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # 将额外的参数传递给父类方法
        return super().training_step(model, inputs, *args, **kwargs)
        
    def get_train_dataloader(self, *args, **kwargs):
        # 覆盖默认的数据加载器创建逻辑
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # 使用我们优化的数据加载器
        return optimize_data_loader(self.train_dataset, self.args.per_device_train_batch_size)

# 优化模型和数据的设备放置
def prepare_model_for_training(model):
    """优化模型以进行高效训练"""
    # 启用CUDA图优化（如果支持）
    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
        torch.backends.cudnn.benchmark = True  # 启用cudnn自动调优
    
    # 确保模型在正确的设备上
    model = model.to(model.device)
    
    # 设置模型为训练模式
    model.train()
    
    return model

# 准备模型进行训练
model = prepare_model_for_training(model)

# 使用自定义Trainer
if torch.cuda.is_available():
    # 在GPU上使用自定义Trainer
    trainer = CustomTrainer(model=model, args=training_args, train_dataset=dataset)
else:
    # 在CPU上使用标准Trainer（可能需要调整参数以减少内存使用）
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

# 添加定期的内存清理和性能监控
from transformers import TrainerCallback
def cleanup_memory():
    """定期清理内存以避免内存泄漏"""
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

# 添加定期清理回调
class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 每100步清理一次内存
        if state.global_step % 100 == 0:
            cleanup_memory()

# 添加回调到trainer
cleanup_callback = MemoryCleanupCallback()
trainer.add_callback(cleanup_callback)

# 尝试训练模型
print("开始训练...")
trainer.train()

# 确保保存目录存在
save_dir = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/lora-ckpt/final"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)

print("训练完成，模型已保存到:", save_dir)