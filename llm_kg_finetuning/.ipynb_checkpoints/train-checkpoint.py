import os
import sys
import json
import argparse
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    logging as transformers_logging
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import deepspeed
import pandas as pd

# 设置日志级别
transformers_logging.set_verbosity_info()

# 导入自定义模块
from config import get_config, print_config_summary, validate_config, save_config, load_config
from data_processor import (
    EntityRelationMapper,
    KnowledgeGraphDataLoader,
    PromptGenerator,
    NegativeSampler,
    create_data_statistics,
    filter_unknown_entities
)

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM知识图谱预测器训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default="./config.json", 
                        help="配置文件路径")
    parser.add_argument("--save-config", action="store_true", 
                        help="保存当前配置到文件")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"],
                        help="运行模式: train(训练), eval(评估), predict(预测)")
    
    # 数据路径覆盖参数
    parser.add_argument("--train-file", type=str, default=None, 
                        help="训练集文件路径")
    parser.add_argument("--test-file", type=str, default=None, 
                        help="测试集文件路径")
    parser.add_argument("--dev-file", type=str, default=None, 
                        help="开发集文件路径")
    parser.add_argument("--entity-text-file", type=str, default=None, 
                        help="实体文本映射文件路径")
    parser.add_argument("--relation-text-file", type=str, default=None, 
                        help="关系文本映射文件路径")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="输出目录")
    parser.add_argument("--model-dir", type=str, default=None, 
                        help="模型保存目录")
    
    # 模型参数覆盖
    parser.add_argument("--model-name", type=str, default=None, 
                        help="预训练模型名称或路径")
    parser.add_argument("--use-quantization", action="store_true", 
                        help="启用模型量化")
    parser.add_argument("--no-use-quantization", action="store_true", 
                        help="禁用模型量化")
    parser.add_argument("--quantization-bits", type=int, default=None, 
                        help="量化位数(4或8)")
    parser.add_argument("--use-lora", action="store_true", 
                        help="启用LoRA参数高效微调")
    parser.add_argument("--no-use-lora", action="store_true", 
                        help="禁用LoRA参数高效微调")
    parser.add_argument("--lora-rank", type=int, default=None, 
                        help="LoRA秩")
    
    # 训练参数覆盖
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="训练批次大小")
    parser.add_argument("--learning-rate", type=float, default=None, 
                        help="学习率")
    parser.add_argument("--num-epochs", type=int, default=None, 
                        help="训练轮数")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, 
                        help="梯度累积步数")
    
    # 预测参数
    parser.add_argument("--top-k", type=int, default=None, 
                        help="预测时考虑的候选实体数量")
    
    # 其他选项
    parser.add_argument("--debug", action="store_true", 
                        help="启用调试模式")
    parser.add_argument("--use-deepspeed", action="store_true", 
                        help="启用DeepSpeed加速")
    parser.add_argument("--no-use-deepspeed", action="store_true", 
                        help="禁用DeepSpeed加速")
    
    args = parser.parse_args()
    return args

def update_config_with_args(config, args):
    """使用命令行参数更新配置"""
    # 更新数据路径
    if args.train_file is not None:
        config.data_paths["train_file"] = args.train_file
    if args.test_file is not None:
        config.data_paths["test_file"] = args.test_file
    if args.dev_file is not None:
        config.data_paths["dev_file"] = args.dev_file
    if args.entity_text_file is not None:
        config.data_paths["entity_text_file"] = args.entity_text_file
    if args.relation_text_file is not None:
        config.data_paths["relation_text_file"] = args.relation_text_file
    if args.output_dir is not None:
        config.data_paths["output_dir"] = args.output_dir
    if args.model_dir is not None:
        config.data_paths["model_dir"] = args.model_dir
    
    # 更新模型配置
    if args.model_name is not None:
        config.model_config["model_name"] = args.model_name
    if args.use_quantization:
        config.model_config["use_quantization"] = True
    if args.no_use_quantization:
        config.model_config["use_quantization"] = False
    if args.quantization_bits is not None:
        config.model_config["quantization_bits"] = args.quantization_bits
    if args.use_lora:
        config.model_config["use_lora"] = True
    if args.no_use_lora:
        config.model_config["use_lora"] = False
    if args.lora_rank is not None:
        config.model_config["lora_rank"] = args.lora_rank
    if args.use_deepspeed:
        config.model_config["use_deepspeed"] = True
    if args.no_use_deepspeed:
        config.model_config["use_deepspeed"] = False
    
    # 更新训练配置
    if args.batch_size is not None:
        config.training_config["per_device_train_batch_size"] = args.batch_size
        config.training_config["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config.training_config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config.training_config["num_train_epochs"] = args.num_epochs
    if args.gradient_accumulation_steps is not None:
        config.training_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    
    # 更新预测配置
    if args.top_k is not None:
        config.prediction_config["top_k_candidates"] = args.top_k
    
    return config

def load_model_and_tokenizer(config):
    """加载模型和分词器"""
    print(f"\n加载模型: {config.model_config['model_name']}...")
    
    # 设置量化配置
    quantization_config = None
    if config.model_config["use_quantization"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.model_config["quantization_bits"] == 4,
            load_in_8bit=config.model_config["quantization_bits"] == 8,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print(f"启用量化: {config.model_config['quantization_bits']}位")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config["model_name"],
        use_fast=True,
        padding_side="right"
    )
    
    # 添加结束标记（如果模型没有的话）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("已添加pad_token")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 准备模型进行量化训练
    if config.model_config["use_quantization"]:
        model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA
    if config.model_config["use_lora"]:
        lora_config = LoraConfig(
            r=config.model_config["lora_rank"],
            lora_alpha=config.model_config["lora_alpha"],
            target_modules=config.model_config["target_modules"],
            lora_dropout=config.model_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("\n可训练参数:")
        model.print_trainable_parameters()
    
    print("模型加载完成")
    return tokenizer, model

def prepare_training_data(config, data_loader, mapper):
    """准备训练数据"""
    print("\n准备训练数据...")
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 创建Prompt生成器和负采样器
    prompt_generator = PromptGenerator()
    negative_sampler = NegativeSampler(all_entities)
    
    # 准备训练数据
    train_data = []
    for h, r, t in tqdm(data_loader.train_triples, desc="处理训练数据"):
        # 获取中文文本
        h_text = mapper.get_entity_text(h)
        r_text = mapper.get_relation_text(r)
        t_text = mapper.get_entity_text(t)
        
        # 生成正样本
        pos_prompt = prompt_generator.generate_positive_prompt(h_text, r_text, t_text)
        train_data.append({
            "text": pos_prompt,
            "label": 1
        })
        
        # 生成负样本
        num_neg_samples = config.data_processing_config["num_negative_samples"]
        neg_entities = negative_sampler.sample_negative(t, num_neg_samples)
        for neg_t in neg_entities:
            neg_t_text = mapper.get_entity_text(neg_t)
            neg_prompt = prompt_generator.generate_negative_prompt(h_text, r_text, neg_t_text)
            train_data.append({
                "text": neg_prompt,
                "label": 0
            })
    
    # 转换为Dataset对象
    train_dataset = Dataset.from_pandas(
        pd.DataFrame(train_data).sample(frac=1, random_state=config.data_processing_config["seed"])
    )
    
    # 准备开发集数据
    dev_data = []
    for h, r, t in tqdm(data_loader.dev_triples[:1000], desc="处理开发数据"):  # 限制开发集大小以提高效率
        h_text = mapper.get_entity_text(h)
        r_text = mapper.get_relation_text(r)
        t_text = mapper.get_entity_text(t)
        
        # 只生成正样本用于评估
        pos_prompt = prompt_generator.generate_positive_prompt(h_text, r_text, t_text)
        dev_data.append({
            "text": pos_prompt,
            "label": 1
        })
    
    dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"开发数据大小: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset, all_entities

def setup_training_args(config):
    """设置训练参数"""
    print("\n设置训练参数...")
    
    # 确保输出目录存在
    os.makedirs(config.data_paths["output_dir"], exist_ok=True)
    os.makedirs(config.data_paths["model_dir"], exist_ok=True)
    
    # 配置DeepSpeed参数
    deepspeed_config_path = None
    if config.model_config["use_deepspeed"]:
        deepspeed_config = {
            "fp16": {
                "enabled": config.training_config["fp16"]
            },
            "bf16": {
                "enabled": False
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.training_config["learning_rate"],
                    "weight_decay": config.training_config["weight_decay"]
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config.training_config["learning_rate"],
                    "warmup_num_steps": config.training_config["warmup_ratio"]
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
        deepspeed_config_path = os.path.join(config.data_paths["output_dir"], "deepspeed_config.json")
        with open(deepspeed_config_path, "w") as f:
            json.dump(deepspeed_config, f, indent=2)
        print(f"DeepSpeed配置已保存到: {deepspeed_config_path}")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.data_paths["output_dir"],
        per_device_train_batch_size=config.training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config.training_config["gradient_accumulation_steps"],
        learning_rate=config.training_config["learning_rate"],
        num_train_epochs=config.training_config["num_train_epochs"],
        warmup_ratio=config.training_config["warmup_ratio"],
        weight_decay=config.training_config["weight_decay"],
        save_steps=config.training_config["save_steps"],
        logging_steps=config.training_config["logging_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=config.training_config["fp16"],
        deepspeed=deepspeed_config_path,
        gradient_checkpointing=config.model_config["gradient_checkpointing"]
    )
    
    return training_args

def create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, config):
    """创建Trainer实例"""
    print("\n创建Trainer...")
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不使用掩码语言建模
    )
    
    # 预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            max_length=config.training_config["max_seq_length"],
            truncation=True,
            padding="max_length"
        )
    
    # 处理数据集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4 if not config.debug else 1
    )
    
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4 if not config.debug else 1
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

def save_model(trainer, tokenizer, model, config):
    """保存训练好的模型"""
    print(f"\n保存模型到: {config.data_paths['model_dir']}...")
    
    # 如果使用了LoRA，只保存LoRA权重
    if config.model_config["use_lora"]:
        trainer.model.save_pretrained(config.data_paths["model_dir"])
        tokenizer.save_pretrained(config.data_paths["model_dir"])
    else:
        # 保存完整模型
        model.save_pretrained(config.data_paths["model_dir"])
        tokenizer.save_pretrained(config.data_paths["model_dir"])
    
    print("模型保存完成")

def predict_test_set(test_triples, mapper, tokenizer, model, all_entities, config):
    """在测试集上进行预测"""
    print("\n在测试集上进行预测...")
    
    predictions = []
    model.eval()
    device = next(model.parameters()).device
    prompt_generator = PromptGenerator()
    
    # 批次处理预测
    batch_size = config.prediction_config["batch_size"]
    top_k = config.prediction_config["top_k_candidates"]
    
    for i in tqdm(range(0, len(test_triples), batch_size), desc="处理测试批次"):
        batch_triples = test_triples[i:i+batch_size]
        batch_predictions = []
        
        for h, r, _ in batch_triples:
            # 获取中文文本
            h_text = mapper.get_entity_text(h)
            r_text = mapper.get_relation_text(r)
            
            # 生成预测prompt
            prompt = prompt_generator.generate_prediction_prompt(h_text, r_text)
            
            # 为每个测试样本生成候选实体
            # 注意：这里为了简化，我们直接从所有实体中随机选择top_k个作为候选
            # 实际应用中，应该使用更智能的方法（如知识图谱嵌入）来预筛选候选实体
            selected_entities = random.sample(all_entities, min(top_k, len(all_entities)))
            
            # 为每个候选实体生成完整的预测prompt
            candidate_prompts = []
            candidate_entity_ids = []
            
            for candidate_entity in selected_entities:
                candidate_text = mapper.get_entity_text(candidate_entity)
                full_prompt = prompt + candidate_text
                candidate_prompts.append(full_prompt)
                candidate_entity_ids.append(candidate_entity)
            
            # 批量处理候选实体
            inputs = tokenizer(
                candidate_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.training_config["max_seq_length"]
            ).to(device)
            
            # 计算每个候选实体的概率
            with torch.no_grad():
                outputs = model(**inputs)
                # 取最后一个token的logits
                logits = outputs.logits[:, -1, :]
                # 计算概率
                probs = torch.softmax(logits, dim=-1)
                # 取结束token的概率作为候选实体的得分
                eos_token_id = tokenizer.eos_token_id
                eos_probs = probs[:, eos_token_id].cpu().numpy()
            
            # 选择得分最高的实体作为预测结果
            best_idx = np.argmax(eos_probs)
            best_entity = candidate_entity_ids[best_idx]
            
            batch_predictions.append((h, r, best_entity))
        
        # 过滤掉不在已知实体集合中的预测结果
        filtered_predictions = filter_unknown_entities(batch_predictions, set(all_entities))
        predictions.extend(filtered_predictions)
    
    print(f"预测完成，共处理 {len(predictions)} 个测试样本")
    return predictions

def save_predictions(predictions, output_dir):
    """保存预测结果"""
    output_file = os.path.join(output_dir, "test_predictions.tsv")
    with open(output_file, 'w', encoding='utf-8') as f:
        for h, r, t_pred in predictions:
            f.write(f"{h}\t{r}\t{t_pred}\n")
    
    print(f"预测结果已保存到: {output_file}")

def run_training(config):
    """执行训练流程"""
    print("\n===== 开始训练流程 ====")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 2. 加载实体和关系映射
    print("\n2. 加载实体和关系映射...")
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据...")
    train_dataset, dev_dataset, all_entities = prepare_training_data(config, data_loader, mapper)
    
    # 4. 加载模型和分词器
    print("\n4. 加载模型和分词器...")
    tokenizer, model = load_model_and_tokenizer(config)
    
    # 5. 设置训练参数
    print("\n5. 设置训练参数...")
    training_args = setup_training_args(config)
    
    # 6. 创建Trainer
    print("\n6. 创建Trainer...")
    trainer = create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, config)
    
    # 7. 开始训练
    print("\n7. 开始训练...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"训练完成，耗时: {(end_time - start_time) / 3600:.2f} 小时")
    
    # 8. 保存模型
    print("\n8. 保存模型...")
    save_model(trainer, tokenizer, model, config)
    
    # 9. 在测试集上进行预测
    print("\n9. 在测试集上进行预测...")
    predictions = predict_test_set(data_loader.test_triples, mapper, tokenizer, model, all_entities, config)
    
    # 10. 保存预测结果
    print("\n10. 保存预测结果...")
    save_predictions(predictions, config.data_paths["output_dir"])
    
    print("\n===== 训练流程完成 ====")

def run_evaluation(config):
    """执行评估流程"""
    print("\n===== 开始评估流程 ====")
    
    # 加载数据
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 加载实体和关系映射
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 加载模型和分词器
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.data_paths["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(
        config.data_paths["model_dir"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 在开发集上进行评估
    print("\n在开发集上进行评估...")
    # 注意：这里简化了评估过程，实际应用中应该实现更详细的评估指标计算
    dev_triples_with_t = [(h, r, t) for h, r, t in data_loader.dev_triples if t is not None]
    predictions = predict_test_set(dev_triples_with_t, mapper, tokenizer, model, all_entities, config)
    
    # 计算简单的准确率（仅供参考）
    correct = 0
    for (h, r, t_pred), (_, _, t_true) in zip(predictions, dev_triples_with_t):
        if t_pred == t_true:
            correct += 1
    
    accuracy = correct / len(predictions) if predictions else 0
    print(f"评估准确率: {accuracy:.4f}")
    
    # 保存评估结果
    eval_result = {
        "accuracy": accuracy,
        "total_samples": len(predictions),
        "correct_predictions": correct
    }
    
    with open(os.path.join(config.data_paths["output_dir"], "evaluation_result.json"), 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    print("\n===== 评估流程完成 ====")

def run_prediction(config):
    """执行预测流程"""
    print("\n===== 开始预测流程 ====")
    
    # 加载数据
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 加载实体和关系映射
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 加载模型和分词器
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.data_paths["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(
        config.data_paths["model_dir"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 在测试集上进行预测
    print("\n在测试集上进行预测...")
    predictions = predict_test_set(data_loader.test_triples, mapper, tokenizer, model, all_entities, config)
    
    # 保存预测结果
    save_predictions(predictions, config.data_paths["output_dir"])
    
    print("\n===== 预测流程完成 ====")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"配置文件不存在: {args.config}，使用默认配置")
        config = get_config()
    
    # 使用命令行参数更新配置
    config = update_config_with_args(config, args)
    
    # 设置调试模式
    if args.debug:
        print("\n启用调试模式")
        config.debug = True
    else:
        config.debug = False
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 验证配置
    valid, errors = validate_config(config)
    if not valid:
        print("\n配置验证失败:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    
    # 保存配置（如果需要）
    if args.save_config:
        save_config(config, args.config)
        print("\n配置已保存")
        return
    
    # 设置随机种子
    set_seed(config.data_processing_config["seed"])
    
    # 根据模式执行相应的流程
    if args.mode == "train":
        run_training(config)
    elif args.mode == "eval":
        run_evaluation(config)
    elif args.mode == "predict":
        run_prediction(config)
    else:
        print(f"不支持的模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# 导入pandas用于数据处理
import pandas as pd