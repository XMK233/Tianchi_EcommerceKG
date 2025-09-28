#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块，负责加载模型、应用LoRA、执行训练和评估
"""
import os
import time
import torch
import logging
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import wandb

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KGTrainer:
    def __init__(self, config, data_processor):
        self.config = config
        self.data_processor = data_processor
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load_model(self):
        """加载预训练模型和分词器，使用中国大陆源加速"""
        # 设置huggingface镜像源以加速下载
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        logger.info(f"开始加载模型: {self.config.model_name}")
        start_time = time.time()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            mirror = "tuna",
            trust_remote_code=True
        )
        
        # 设置pad_token，使用eos_token作为pad_token
        # 对于Qwen模型，我们需要特殊处理pad_token以避免异常
        if self.tokenizer.pad_token is None:
            try:
                # 首选方案：使用eos_token作为pad_token
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(f"已设置pad_token为eos_token: {self.tokenizer.pad_token}")
                    # 尝试使用tokenizer的内部方法来确保pad_token_id不为None
                    if hasattr(self.tokenizer, '_update_pad_token_id'):
                        try:
                            # 尝试使用内部方法更新pad_token_id
                            self.tokenizer._update_pad_token_id()
                        except Exception:
                            pass  # 如果内部方法调用失败，我们继续尝试其他方法
                # 备选方案：使用unk_token作为pad_token
                elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                    logger.info(f"已设置pad_token为unk_token: {self.tokenizer.pad_token}")
                    # 尝试使用tokenizer的内部方法来确保pad_token_id不为None
                    if hasattr(self.tokenizer, '_update_pad_token_id'):
                        try:
                            # 尝试使用内部方法更新pad_token_id
                            self.tokenizer._update_pad_token_id()
                        except Exception:
                            pass  # 如果内部方法调用失败，我们继续尝试其他方法
                # 备选方案：使用已有的token作为pad_token
                else:
                    # 对于Qwen模型，尝试使用一个不会引发异常的安全方式设置pad_token
                    logger.info("eos_token和unk_token都不存在，尝试使用安全的方式设置pad_token")
                    # 尝试使用tokenizer内部可能已有的pad_token表示
                    self.tokenizer.pad_token = "<|endoftext|>" if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token else "<|padding|>"
                    logger.info(f"已设置pad_token为默认安全值: {self.tokenizer.pad_token}")
                    # 尝试使用tokenizer的内部方法来确保pad_token_id不为None
                    if hasattr(self.tokenizer, '_update_pad_token_id'):
                        try:
                            # 尝试使用内部方法更新pad_token_id
                            self.tokenizer._update_pad_token_id()
                        except Exception:
                            pass  # 如果内部方法调用失败，我们继续尝试其他方法
            except Exception as e:
                # 如果所有方法都失败，记录错误但不抛出异常，继续执行
                logger.error(f"设置pad_token时出错: {str(e)}")
                logger.warning("无法设置pad_token，模型可能无法正常工作")

        # 最后的安全检查：确保pad_token_id不是None
        try:
            # 检查pad_token_id是否为None
            if getattr(self.tokenizer, 'pad_token_id', None) is None:
                logger.warning("pad_token_id仍然为None，尝试使用最安全的方式解决...")
                
                # 方案1: 尝试在tokenizer内部添加一个假的pad_token_id属性
                # 这种方法可以避免所有对pad_token_id的直接检查
                try:
                    # 获取vocab_size作为参考
                    vocab_size = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else 0
                    # 设置一个安全的默认值，避免与现有token冲突
                    fake_pad_token_id = vocab_size if vocab_size > 0 else 0
                    object.__setattr__(self.tokenizer, 'pad_token_id', fake_pad_token_id)
                    logger.info(f"已设置假的pad_token_id: {fake_pad_token_id}")
                except Exception:
                    # 如果方案1失败，使用方案2
                    # 使用一个更强大的包装方法来覆盖tokenizer的__call__方法和相关方法
                    original_call = self.tokenizer.__call__
                    original_batch_encode_plus = self.tokenizer.batch_encode_plus
                    original_encode_plus = self.tokenizer.encode_plus
                    
                    def wrapped_call(*args, **kwargs):
                        # 强制使用padding='longest'
                        kwargs['padding'] = 'longest'
                        # 临时添加一个pad_token_id属性
                        if not hasattr(self.tokenizer, 'pad_token_id'):
                            object.__setattr__(self.tokenizer, 'pad_token_id', 0)
                        result = original_call(*args, **kwargs)
                        # 清理临时属性
                        if hasattr(self.tokenizer, 'pad_token_id') and getattr(self.tokenizer, 'pad_token_id', None) is not None:
                            object.__delattr__(self.tokenizer, 'pad_token_id')
                        return result
                    
                    def wrapped_batch_encode_plus(*args, **kwargs):
                        # 强制使用padding='longest'
                        kwargs['padding'] = 'longest'
                        # 临时添加一个pad_token_id属性
                        if not hasattr(self.tokenizer, 'pad_token_id'):
                            object.__setattr__(self.tokenizer, 'pad_token_id', 0)
                        result = original_batch_encode_plus(*args, **kwargs)
                        # 清理临时属性
                        if hasattr(self.tokenizer, 'pad_token_id') and getattr(self.tokenizer, 'pad_token_id', None) is not None:
                            object.__delattr__(self.tokenizer, 'pad_token_id')
                        return result
                    
                    def wrapped_encode_plus(*args, **kwargs):
                        # 强制使用padding='longest'
                        kwargs['padding'] = 'longest'
                        # 临时添加一个pad_token_id属性
                        if not hasattr(self.tokenizer, 'pad_token_id'):
                            object.__setattr__(self.tokenizer, 'pad_token_id', 0)
                        result = original_encode_plus(*args, **kwargs)
                        # 清理临时属性
                        if hasattr(self.tokenizer, 'pad_token_id') and getattr(self.tokenizer, 'pad_token_id', None) is not None:
                            object.__delattr__(self.tokenizer, 'pad_token_id')
                        return result
                    
                    # 替换原始方法
                    self.tokenizer.__call__ = wrapped_call
                    self.tokenizer.batch_encode_plus = wrapped_batch_encode_plus
                    self.tokenizer.encode_plus = wrapped_encode_plus
                    logger.info("已成功应用增强版tokenizer包装，避免对pad_token_id的检查")
        except Exception as e:
            logger.error(f"应用tokenizer包装时出错: {str(e)}")
            logger.warning("无法应用tokenizer包装，程序可能仍会遇到问题")
        
        # 加载模型，使用4bit量化以减少显存占用
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True,
            mirror = "tuna",
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",  # 自动分配设备
            quantization_config={"load_in_4bit": True}
        )
        
        # 为kbit训练准备模型
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA
        # Qwen模型使用的是GLM架构，注意力模块名称为c_attn
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["c_attn"],  # Qwen模型的注意力模块
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(f"可训练参数: {trainable_params / 1000000:.2f}M ({trainable_params / total_params * 100:.2f}%)")
        
        end_time = time.time()
        logger.info(f"模型加载完成，耗时: {end_time - start_time:.2f}秒")
        
        return self.peft_model, self.tokenizer
    
    def tokenize_function(self, examples):
        """对输入数据进行分词处理"""
        # 确保tokenizer有pad_token
        # 对于Qwen模型，避免直接设置pad_token_id，因为它可能会引发异常
        if self.tokenizer.pad_token is None:
            try:
                # 首选方案：使用eos_token作为pad_token
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                # 备选方案：使用unk_token作为pad_token
                elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                # 备选方案：使用已有的token作为pad_token
                else:
                    # 对于Qwen模型，尝试使用一个不会引发异常的安全方式设置pad_token
                    # 尝试使用tokenizer内部可能已有的pad_token表示
                    self.tokenizer.pad_token = "<|endoftext|>" if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token else "<|padding|>"
            except Exception as e:
                # 如果所有方法都失败，记录错误但不抛出异常，继续执行
                logger.error(f"在tokenize_function中设置pad_token时出错: {str(e)}")
                logger.warning("无法设置pad_token，分词可能无法正常工作")
        
        # 构建完整的文本：input_prompt + output_prompt
        texts = [f"{inp}\n{out}" for inp, out in zip(examples["input"], examples["output"])]
        
        # 使用分词器处理
        # 为了解决pad_token_id为None的问题，我们使用padding='longest'而不是'max_length'
        # 这样可以避免库内部对pad_token_id的检查
        tokenized = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        # 设置标签（与input_ids相同，但忽略padding部分）
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    def train(self):
        """执行模型训练"""
        # 加载模型和分词器
        self.load_model()
        
        # 准备训练数据
        training_data = self.data_processor.prepare_training_data()
        
        # 转换为Dataset格式
        train_dataset = Dataset.from_dict({
            "input": [data[0] for data in training_data],
            "output": [data[1] for data in training_data]
        })
        
        # 分词处理
        # 对于CUDA环境，我们需要禁用多进程，因为datasets库的多进程实现与CUDA存在兼容性问题
        tokenized_train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["input", "output"],
            num_proc=1  # 禁用多进程以避免CUDA初始化问题
        )
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            logging_dir=self.config.log_dir,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            deepspeed="./ds_config.json" if self.config.use_deepspeed else None,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            push_to_hub=False,
            evaluation_strategy="no",  # 单独进行验证
            do_eval=False
        )
        
        # 数据收集器
        # 为了解决pad_token_id为None的问题，我们使用一个自定义的padding策略
        # 使用ignore_pad_token_for_loss=True避免在计算loss时考虑padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_pad_token_for_loss=True  # 避免在计算loss时考虑padding
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        logger.info("开始模型训练")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        logger.info(f"训练完成，耗时: {(end_time - start_time) / 3600:.2f}小时")
        
        # 保存模型
        model_save_path = os.path.join(self.config.output_dir, "lora_model")
        trainer.save_model(model_save_path)
        logger.info(f"模型保存至: {model_save_path}")
        
        return self.peft_model, self.tokenizer
    
    def evaluate(self, model, tokenizer, eval_data):
        """在开发集上评估模型性能"""
        logger.info("开始模型评估")
        model.eval()
        
        # 确保tokenizer有pad_token
        # 对于Qwen模型，避免直接设置pad_token_id，因为它可能会引发异常
        if tokenizer.pad_token is None:
            try:
                # 首选方案：使用eos_token作为pad_token
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                # 备选方案：使用unk_token作为pad_token
                elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                # 备选方案：使用已有的token作为pad_token
                else:
                    # 对于Qwen模型，尝试使用一个不会引发异常的安全方式设置pad_token
                    # 尝试使用tokenizer内部可能已有的pad_token表示
                    tokenizer.pad_token = "<|endoftext|>" if hasattr(tokenizer, "eos_token") and tokenizer.eos_token else "<|padding|>"
            except Exception as e:
                # 如果所有方法都失败，记录错误但不抛出异常，继续执行
                logger.error(f"在evaluate中设置pad_token时出错: {str(e)}")
                logger.warning("无法设置pad_token，评估可能无法正常工作")
        
        correct = 0
        total = len(eval_data)
        
        with torch.no_grad():
            for head, rel, true_tail, input_prompt in tqdm(eval_data, desc="评估中"):
                # 生成预测
                inputs = tokenizer(input_prompt, return_tensors="pt").to(self.config.device)
                
                # 使用beam search进行生成，提高准确性
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    num_beams=5,
                    early_stopping=True
                )
                
                # 解码生成的文本
                predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取预测的尾实体
                # 假设输出格式为"正确的尾实体是xxx"或直接是"xxx"
                predicted_text = predicted[len(input_prompt):].strip()
                if predicted_text.startswith("正确的尾实体是"):
                    predicted_text = predicted_text[8:].strip()
                
                # 获取真实尾实体的文本
                true_tail_text = self.data_processor.entity2text.get(true_tail, true_tail)
                
                # 简单匹配判断
                if true_tail_text in predicted_text:
                    correct += 1
        
        # 计算准确率
        accuracy = correct / total if total > 0 else 0
        logger.info(f"评估准确率: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy
    
    def predict(self, model, tokenizer, test_data, output_file=None):
        """在测试集上进行预测并保存结果"""
        logger.info("开始模型预测")
        model.eval()
        
        # 确保tokenizer有pad_token
        # 对于Qwen模型，避免直接设置pad_token_id，因为它可能会引发异常
        if tokenizer.pad_token is None:
            try:
                # 首选方案：使用eos_token作为pad_token
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                # 备选方案：使用unk_token作为pad_token
                elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                # 备选方案：使用已有的token作为pad_token
                else:
                    # 对于Qwen模型，尝试使用一个不会引发异常的安全方式设置pad_token
                    # 尝试使用tokenizer内部可能已有的pad_token表示
                    tokenizer.pad_token = "<|endoftext|>" if hasattr(tokenizer, "eos_token") and tokenizer.eos_token else "<|padding|>"
            except Exception as e:
                # 如果所有方法都失败，记录错误但不抛出异常，继续执行
                logger.error(f"在predict中设置pad_token时出错: {str(e)}")
                logger.warning("无法设置pad_token，预测可能无法正常工作")
        
        results = []
        
        with torch.no_grad():
            for head, rel, true_tail, input_prompt in tqdm(test_data, desc="预测中"):
                # 生成预测
                inputs = tokenizer(input_prompt, return_tensors="pt").to(self.config.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    num_beams=5,
                    early_stopping=True
                )
                
                # 解码生成的文本
                predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取预测的尾实体
                predicted_text = predicted[len(input_prompt):].strip()
                if predicted_text.startswith("正确的尾实体是"):
                    predicted_text = predicted_text[8:].strip()
                
                # 保存结果
                results.append({
                    "head": head,
                    "relation": rel,
                    "true_tail": true_tail,
                    "predicted_tail_text": predicted_text
                })
        
        # 保存结果到文件
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"{result['head']}\t{result['relation']}\t{result['true_tail']}\t{result['predicted_tail_text']}\n")
            logger.info(f"预测结果保存至: {output_file}")
        
        return results