import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import random
import time
import jieba
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset, concatenate_datasets
import warnings
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 参数设置
    class Args:
        def __init__(self):
            # 数据路径
            self.train_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
            self.test_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
            self.dev_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv"
            self.entity_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_entity2text.tsv"
            self.relation_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_relation2text.tsv"
            self.output_dir = "./results"
            self.model_dir = "./model"
            
            # 模型设置
            self.model_name = "mistralai/Mistral-7B-v0.1"  # 可以替换为其他中文模型
            self.use_quantization = True
            self.quantization_bits = 4  # 4位量化
            
            # LoRA设置
            self.use_lora = True
            self.lora_rank = 8
            self.lora_alpha = 16
            self.lora_dropout = 0.1
            self.target_modules = ["q_proj", "v_proj"]  # 根据模型架构调整
            
            # 训练设置
            self.use_deepspeed = False  # 默认禁用DeepSpeed，需要CUDA环境手动启用
            self.per_device_train_batch_size = 4
            self.per_device_eval_batch_size = 4
            self.gradient_accumulation_steps = 8
            self.learning_rate = 2e-5
            self.num_train_epochs = 3
            self.warmup_ratio = 0.1
            self.weight_decay = 0.01
            self.save_steps = 500
            self.logging_steps = 100
            self.max_seq_length = 200
            
            # 负采样设置
            self.num_negative_samples = 1  # 每个正样本对应一个负样本
            
            # 预测设置
            self.top_k_candidates = 10  # 预测时考虑的候选实体数量
    
    args = Args()
    set_seed()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 1. 加载实体和关系的中文文本映射
    print("加载实体和关系的中文文本映射...")
    entity_text_map = load_text_mappings(args.entity_text_file)
    relation_text_map = load_text_mappings(args.relation_text_file)
    
    # 2. 加载训练数据
    print("加载知识图谱数据...")
    train_triples = load_kg_data(args.train_file)
    dev_triples = load_kg_data(args.dev_file)
    test_triples = load_kg_data(args.test_file, is_test=True)
    
    # 收集所有实体，用于确保预测时不会生成未见过的实体
    all_entities = collect_all_entities(train_triples, dev_triples, test_triples)
    entity_to_text = {entity: entity_text_map.get(entity, entity) for entity in all_entities}
    
    # 3. 准备训练数据，包括正样本和负样本
    print("准备训练数据...")
    train_dataset = prepare_training_data(
        train_triples,
        entity_to_text,
        relation_text_map,
        all_entities,
        args.num_negative_samples
    )
    
    dev_dataset = prepare_training_data(
        dev_triples,
        entity_to_text,
        relation_text_map,
        all_entities,
        args.num_negative_samples,
        is_dev=True
    )
    
    # 4. 加载模型和分词器
    print(f"加载模型: {args.model_name}...")
    tokenizer, model = load_model_and_tokenizer(args)
    
    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args(args)
    
    # 6. 创建Trainer
    print("创建Trainer...")
    trainer = create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, args)
    
    # 7. 开始训练
    print("开始训练...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"训练完成，耗时: {(end_time - start_time) / 3600:.2f} 小时")
    
    # 8. 保存模型
    print("保存模型...")
    save_model(trainer, tokenizer, model, args)
    
    # 9. 在测试集上进行预测
    print("在测试集上进行预测...")
    test_predictions = predict_test_set(
        test_triples,
        entity_to_text,
        relation_text_map,
        tokenizer,
        model,
        all_entities,
        args
    )
    
    # 10. 保存预测结果
    print("保存预测结果...")
    save_predictions(test_predictions, args.output_dir)
    
    print("所有任务完成！")

def load_text_mappings(file_path):
    """加载实体或关系的中文文本映射"""
    text_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    entity_text = parts[1]
                    text_map[entity_id] = entity_text
        print(f"已加载 {len(text_map)} 个条目")
    except Exception as e:
        print(f"加载文本映射时出错: {e}")
    return text_map

def load_kg_data(file_path, is_test=False):
    """加载知识图谱数据"""
    triples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if is_test:
                    if len(parts) >= 2:
                        h, r = parts[0], parts[1]
                        triples.append((h, r, None))
                else:
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        triples.append((h, r, t))
        print(f"已加载 {len(triples)} 个三元组")
    except Exception as e:
        print(f"加载知识图谱数据时出错: {e}")
    return triples

def collect_all_entities(train_triples, dev_triples, test_triples):
    """收集所有实体"""
    entities = set()
    
    for h, r, t in train_triples:
        entities.add(h)
        if t is not None:
            entities.add(t)
    
    for h, r, t in dev_triples:
        entities.add(h)
        if t is not None:
            entities.add(t)
    
    for h, r, _ in test_triples:
        entities.add(h)
    
    print(f"总共收集到 {len(entities)} 个唯一实体")
    return list(entities)

def prepare_training_data(triples, entity_to_text, relation_to_text, all_entities, num_negative_samples, is_dev=False):
    """准备用于LLM训练的数据"""
    data = []
    entity_set = set(all_entities)
    
    for h, r, t in tqdm(triples, desc="准备训练数据"):
        if t is None:  # 测试数据
            continue
        
        # 获取中文文本
        h_text = entity_to_text.get(h, h)
        r_text = relation_to_text.get(r, r)
        t_text = entity_to_text.get(t, t)
        
        # 生成正样本prompt
        pos_prompt = generate_prompt(h_text, r_text, t_text, is_positive=True)
        data.append({
            "text": pos_prompt,
            "label": 1  # 正样本标记
        })
        
        # 生成负样本
        for _ in range(num_negative_samples):
            # 负采样：随机选择一个不同于t的实体作为负样本尾实体
            neg_t = random.choice(all_entities)
            while neg_t == t:
                neg_t = random.choice(all_entities)
            
            neg_t_text = entity_to_text.get(neg_t, neg_t)
            neg_prompt = generate_prompt(h_text, r_text, neg_t_text, is_positive=False)
            data.append({
                "text": neg_prompt,
                "label": 0  # 负样本标记
            })
    
    # 转换为Dataset对象
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    
    # 如果是开发集，随机选择一部分作为评估样本
    if is_dev:
        dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    
    return dataset

def generate_prompt(h_text, r_text, t_text, is_positive=True):
    """生成用于LLM微调的prompt"""
    if is_positive:
        prompt = f"""input是`头实体是{h_text}，关系是{r_text}`，output是`正确的尾实体是{t_text}`"""
    else:
        prompt = f"""input是`头实体是{h_text}，关系是{r_text}`，output是`错误的尾实体是{t_text}`"""
    return prompt

def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    # 设置量化配置
    quantization_config = None
    if args.use_quantization:
        if args.quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif args.quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # 关键：启用FP32参数的CPU卸载
            )
        print(f"启用量化: {args.quantization_bits}位，已启用CPU卸载")
    else:
        # 即使不使用量化，也可以使用这些优化
        print("未启用量化，但仍会应用内存优化")
    
    # 配置Hugging Face国内镜像源和下载优化
    import os
    # 设置Hugging Face镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 启用并行下载和分片下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用hf_transfer加速
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条减少开销
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 增加下载超时时间
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # 增加ETAG超时时间
    # 内存优化配置 - 增强版
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"  # 启用可扩展内存段并设置最大分割大小
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 有助于更准确地定位CUDA错误
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"  # 启用CUDA内存缓存以提高效率
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
        mirror="tuna",
        cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
    )
    
    # 添加结束标记（如果模型没有的话）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 优化设备映射 - 更精细地控制内存使用
    device_map = "auto"
    if torch.cuda.is_available():
        # 当显存不足时，使用更激进的设备映射策略
        # 先让Hugging Face自动分配，如果仍有问题则会回退到手动映射
        print(f"使用自动设备映射 + CPU卸载策略")
    
    # 设置设备内存限制 - 更激进的内存限制以防止OOM
    max_memory = {
        0: "6GiB",  # 进一步降低GPU内存限制至6GB，强制更多层卸载到CPU
        "cpu": "32GiB"  # 限制CPU内存使用32GB
    }
    
    # 添加额外的内存优化参数 - 超激进的配置
    model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": device_map,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "mirror": "tuna",
            "cache_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "local_files_only": False,
            "offload_folder": "/tmp/offload",
            "offload_state_dict": True,
            "max_memory": max_memory
        }
    
    # 加载模型 - 采取最保守的内存优化策略
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # 打印当前内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"模型加载后 - GPU内存已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 准备模型进行量化训练
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return tokenizer, model

def setup_training_args(args):
    """配置训练参数"""
    # 配置DeepSpeed参数
    deepspeed_config = None
    deepspeed_config_path = None
    if args.use_deepspeed:
        try:
            import deepspeed
            deepspeed_config = {
                "fp16": {
                    "enabled": True
                },
                "bf16": {
                    "enabled": False
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": args.learning_rate,
                        "weight_decay": args.weight_decay
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": args.learning_rate,
                        "warmup_num_steps": args.warmup_ratio
                    }
                },
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                }
            }
            
            # 保存DeepSpeed配置
            deepspeed_config_path = "deepspeed_config.json"
            with open(deepspeed_config_path, "w") as f:
                json.dump(deepspeed_config, f, indent=2)
        except ImportError:
            print("警告: DeepSpeed未安装。将使用Accelerate库作为替代方案。")
            args.use_deepspeed = False
            deepspeed_config_path = None
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        deepspeed=deepspeed_config_path,
        gradient_checkpointing=True  # 节省显存
    )
    
    # 如果没有DeepSpeed，使用Accelerate的优化策略
    if not args.use_deepspeed:
        print("使用Accelerate优化策略来代替DeepSpeed：")
        print("1. 梯度累积以模拟更大批次")
        print("2. 混合精度训练(fp16)")
        print("3. 梯度检查点节省内存")
        print("4. 可通过环境变量配置分布式训练")
    
    return training_args

def create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, args):
    """创建Trainer实例"""
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不使用掩码语言建模
    )
    
    # 预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length"
        )
    
    # 处理数据集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4
    )
    
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    return trainer

def save_model(trainer, tokenizer, model, args):
    """保存训练好的模型"""
    # 如果使用了LoRA，只保存LoRA权重
    if args.use_lora:
        trainer.model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    else:
        # 保存完整模型
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    
    print(f"模型已保存到: {args.model_dir}")

def predict_test_set(test_triples, entity_to_text, relation_to_text, tokenizer, model, all_entities, args):
    """在测试集上进行预测"""
    predictions = []
    model.eval()
    device = next(model.parameters()).device
    
    for h, r, _ in tqdm(test_triples, desc="预测测试集"):
        # 获取中文文本
        h_text = entity_to_text.get(h, h)
        r_text = relation_to_text.get(r, r)
        
        # 为每个实体生成一个候选prompt
        candidate_prompts = []
        candidate_entities = []
        
        # 选择top_k个候选实体（为了效率，实际应用中可以先通过其他方法过滤）
        # 这里简单随机选择，实际应用中可以使用知识库嵌入或其他方法进行预筛选
        selected_entities = random.sample(all_entities, min(args.top_k_candidates, len(all_entities)))
        
        for candidate_entity in selected_entities:
            candidate_text = entity_to_text.get(candidate_entity, candidate_entity)
            # 生成预测prompt
            prompt = f"""input是`头实体是{h_text}，关系是{r_text}`，output是`正确的尾实体是"""
            candidate_prompts.append(prompt)
            candidate_entities.append(candidate_entity)
        
        # 批量处理候选实体
        inputs = tokenizer(
            candidate_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length
        ).to(device)
        
        # 生成并计算困惑度或概率
        with torch.no_grad():
            outputs = model(**inputs)
            # 取最后一个token的logits
            last_token_logits = outputs.logits[:, -1, :]
            # 计算每个候选实体对应的token的概率
            # 这里简化处理，实际应用中应该更精确地计算生成完整实体名的概率
            eos_token_id = tokenizer.eos_token_id
            probs = torch.softmax(last_token_logits, dim=-1)[:, eos_token_id].cpu().numpy()
        
        # 选择概率最高的实体作为预测结果
        if len(probs) > 0:
            best_idx = np.argmax(probs)
            best_entity = candidate_entities[best_idx]
        else:
            best_entity = ""  # 默认值
        
        predictions.append((h, r, best_entity))
    
    return predictions

def save_predictions(predictions, output_dir):
    """保存预测结果"""
    output_file = os.path.join(output_dir, "test_predictions.tsv")
    with open(output_file, 'w', encoding='utf-8') as f:
        for h, r, t_pred in predictions:
            f.write(f"{h}\t{r}\t{t_pred}\n")
    
    print(f"预测结果已保存到: {output_file}")

if __name__ == "__main__":
    main()