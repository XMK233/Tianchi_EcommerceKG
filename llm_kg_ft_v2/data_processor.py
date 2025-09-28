#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块，负责加载数据集、构建映射表、负采样和数据格式化
"""
import os
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

class KGDataProcessor:
    def __init__(self, config):
        self.config = config
        # 初始化映射表
        self.entity2text = {}
        self.relation2text = {}
        # 初始化所有实体集合，用于负采样
        self.all_entities = set()
        self.all_entities_list = []  # 预先转换为列表以加速随机选择
        # 加载映射表
        self._load_mappings()
        
    def _load_mappings(self):
        """加载实体和关系的中文文本映射"""
        # 加载实体文本映射
        print(f"加载实体文本映射: {self.config.entity_text_file}")
        if os.path.exists(self.config.entity_text_file):
            with open(self.config.entity_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id, entity_text = parts[0], parts[1]
                        self.entity2text[entity_id] = entity_text
                        self.all_entities.add(entity_id)
            # 预先转换为列表以加速随机选择
            self.all_entities_list = list(self.all_entities)
        else:
            raise FileNotFoundError(f"实体文本映射文件不存在: {self.config.entity_text_file}")
        
        # 加载关系文本映射
        print(f"加载关系文本映射: {self.config.relation_text_file}")
        if os.path.exists(self.config.relation_text_file):
            with open(self.config.relation_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        relation_id, relation_text = parts[0], parts[1]
                        self.relation2text[relation_id] = relation_text
        else:
            raise FileNotFoundError(f"关系文本映射文件不存在: {self.config.relation_text_file}")
        
        print(f"加载完成: {len(self.entity2text)}个实体, {len(self.relation2text)}个关系")
    
    def load_dataset(self, file_path: str) -> List[Tuple[str, str, str]]:
        """加载数据集（三元组）"""
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head, rel, tail = parts[0], parts[1], parts[2]
                        data.append((head, rel, tail))
        else:
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        print(f"加载数据集{file_path}完成，共{len(data)}个三元组")
        return data
    
    def _generate_negative_tails(self, head: str, rel: str, positive_triples: Set[Tuple[str, str, str]], 
                                num_negatives: int, max_retries: int = 100) -> List[str]:
        """为给定的(head, rel)生成指定数量的负样本尾实体"""
        neg_tails = []
        attempts = 0
        max_attempts = num_negatives * max_retries  # 最大尝试次数
        
        while len(neg_tails) < num_negatives and attempts < max_attempts:
            # 使用预先转换的列表加速随机选择
            neg_tail = random.choice(self.all_entities_list)
            # 检查是否为有效的负样本
            if (head, rel, neg_tail) not in positive_triples:
                neg_tails.append(neg_tail)
            attempts += 1
        
        # 如果无法生成足够的负样本（很少发生），直接返回已生成的
        return neg_tails
    
    def _process_triple_batch(self, triple_batch: List[Tuple[str, str, str]], num_negatives: int, 
                             positive_triples: Set[Tuple[str, str, str]]) -> List[Tuple[str, str, str, str]]:
        """处理一批三元组，生成正负样本"""
        batch_result = []
        for head, rel, tail in triple_batch:
            # 添加正样本
            batch_result.append((head, rel, tail, "positive"))
            
            # 批量生成负样本尾实体
            neg_tails = self._generate_negative_tails(head, rel, positive_triples, num_negatives)
            # 添加负样本
            for neg_tail in neg_tails:
                batch_result.append((head, rel, neg_tail, "negative"))
        return batch_result
    
    def negative_sampling(self, triples: List[Tuple[str, str, str]], num_negatives: int) -> List[Tuple[str, str, str, str]]:
        """负采样：为每个正样本生成负样本
        在CUDA环境中，默认使用串行处理以避免多进程与CUDA的冲突
        """
        return self.negative_sampling_serial(triples, num_negatives)
    
    def negative_sampling_parallel(self, triples: List[Tuple[str, str, str]], num_negatives: int) -> List[Tuple[str, str, str, str]]:
        """并行化负采样：使用多进程加速负采样过程"""
        result = []
        
        # 构建所有正样本的集合，用于检查负样本是否存在
        positive_triples = set(triples)
        
        # 根据CPU核心数确定并行度
        num_workers = min(os.cpu_count() or 4, 8)  # 最多使用8个进程
        batch_size = max(1, len(triples) // (num_workers * 4))  # 每个进程处理的批次大小
        
        # 如果数据量很小，使用串行处理
        if len(triples) < 1000 or num_workers <= 1:
            return self.negative_sampling_serial(triples, num_negatives, positive_triples)
        
        print(f"使用并行负采样，进程数: {num_workers}，批大小: {batch_size}")
        
        # 分批次处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有批次的任务
            futures = []
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i+batch_size]
                # 使用闭包传递参数
                futures.append(executor.submit(
                    self._process_triple_batch, 
                    triple_batch=batch, 
                    num_negatives=num_negatives, 
                    positive_triples=positive_triples
                ))
            
            # 收集所有结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行负采样"):
                batch_result = future.result()
                result.extend(batch_result)
        
        return result
    
    def negative_sampling_serial(self, triples: List[Tuple[str, str, str]], num_negatives: int, 
                                positive_triples: Set[Tuple[str, str, str]] = None) -> List[Tuple[str, str, str, str]]:
        """串行负采样：用于数据量小或调试时"""
        result = []
        
        # 如果没有提供正样本集合，自己构建
        if positive_triples is None:
            positive_triples = set(triples)
        
        for head, rel, tail in tqdm(triples, desc="串行负采样"):
            # 添加正样本
            result.append((head, rel, tail, "positive"))
            
            # 批量生成负样本尾实体
            neg_tails = self._generate_negative_tails(head, rel, positive_triples, num_negatives)
            # 添加负样本
            for neg_tail in neg_tails:
                result.append((head, rel, neg_tail, "negative"))
        
        return result
    
    def build_prompt(self, head: str, rel: str, tail: str, label: str) -> Tuple[str, str]:
        """构建prompt，格式为(input, output)"""
        # 获取实体和关系的中文文本
        head_text = self.entity2text.get(head, head)
        rel_text = self.relation2text.get(rel, rel)
        tail_text = self.entity2text.get(tail, tail)
        
        # 构建输入prompt
        input_prompt = f"头实体是{head_text}，关系是{rel_text}"
        
        # 构建输出prompt
        if label == "positive":
            output_prompt = f"正确的尾实体是{tail_text}"
        else:
            output_prompt = f"错误的尾实体是{tail_text}"
        
        return input_prompt, output_prompt
    
    def prepare_training_data(self) -> List[Tuple[str, str]]:
        """准备训练数据，返回格式为(input_prompt, output_prompt)的列表"""
        # 加载训练集
        train_triples = self.load_dataset(self.config.train_file)
        
        # 负采样
        train_with_negatives = self.negative_sampling(train_triples, self.config.num_negatives)
        
        # 构建训练数据
        training_data = []
        for head, rel, tail, label in tqdm(train_with_negatives, desc="构建训练数据"):
            input_prompt, output_prompt = self.build_prompt(head, rel, tail, label)
            training_data.append((input_prompt, output_prompt))
        
        # 打乱数据顺序
        random.shuffle(training_data)
        
        print(f"训练数据准备完成，共{len(training_data)}条样本")
        return training_data
    
    def prepare_evaluation_data(self, file_path: str) -> List[Tuple[str, str, str, str]]:
        """准备评估数据，返回格式为(head, rel, tail, input_prompt)的列表"""
        # 加载数据集
        triples = self.load_dataset(file_path)
        
        # 构建评估数据
        eval_data = []
        for head, rel, tail in tqdm(triples, desc="构建评估数据"):
            head_text = self.entity2text.get(head, head)
            rel_text = self.relation2text.get(rel, rel)
            input_prompt = f"头实体是{head_text}，关系是{rel_text}"
            eval_data.append((head, rel, tail, input_prompt))
        
        print(f"评估数据准备完成，共{len(eval_data)}条样本")
        return eval_data