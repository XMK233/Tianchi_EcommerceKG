# -*- coding: utf-8 -*-
"""
数据处理脚本：加载数据集、进行负采样、生成训练所需的表格数据
"""
import os
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

class KGDataProcessor:
    def __init__(self):
        # 数据集路径配置
        self.train_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
        self.test_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
        self.dev_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv"
        self.entity_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_entity2text.tsv"
        self.relation_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_relation2text.tsv"
        self.output_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/df_sample_full.jsonl"
        
        # 负采样参数
        self.num_negatives = 3  # 每个正样本对应的负样本数量
        self.max_retries = 100  # 生成负样本的最大尝试次数
        
        # 初始化映射表
        self.entity2text = {}
        self.relation2text = {}
        self.all_entities = set()
        self.all_entities_list = []  # 预先转换为列表以加速随机选择
        
        # 加载映射表
        self._load_mappings()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
    def _load_mappings(self):
        """加载实体和关系的中文文本映射"""
        # 加载实体文本映射
        print(f"加载实体文本映射: {self.entity_text_file}")
        if os.path.exists(self.entity_text_file):
            with open(self.entity_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id, entity_text = parts[0], parts[1]
                        self.entity2text[entity_id] = entity_text
                        self.all_entities.add(entity_id)
            # 预先转换为列表以加速随机选择
            self.all_entities_list = list(self.all_entities)
        else:
            raise FileNotFoundError(f"实体文本映射文件不存在: {self.entity_text_file}")
        
        # 加载关系文本映射
        print(f"加载关系文本映射: {self.relation_text_file}")
        if os.path.exists(self.relation_text_file):
            with open(self.relation_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        relation_id, relation_text = parts[0], parts[1]
                        self.relation2text[relation_id] = relation_text
        else:
            raise FileNotFoundError(f"关系文本映射文件不存在: {self.relation_text_file}")
        
        print(f"加载完成: {len(self.entity2text)}个实体, {len(self.relation2text)}个关系")
    
    def load_dataset(self, file_path: str) -> list:
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
    
    def _generate_negative_tails_batch(self, triple_batch: list, positive_triples: set) -> list:
        """批量为给定的(head, rel)生成负样本尾实体"""
        results = []
        for head, rel, tail in triple_batch:
            # 处理正样本
            head_text = self.entity2text.get(head, head)
            rel_text = self.relation2text.get(rel, rel)
            tail_text = self.entity2text.get(tail, tail)
            
            input_prompt = f"头实体是：{head_text}，关系是：{rel_text}"
            output_prompt = f"正确的尾实体为：{tail_text}"
            results.append(("我们要做一些三元组的推理，这是正例", input_prompt, output_prompt))
            
            # 生成负样本
            neg_tails = []
            attempts = 0
            max_attempts = self.num_negatives * self.max_retries
            
            while len(neg_tails) < self.num_negatives and attempts < max_attempts:
                neg_tail = random.choice(self.all_entities_list)
                if (head, rel, neg_tail) not in positive_triples:
                    neg_tails.append(neg_tail)
                attempts += 1
            
            # 处理负样本
            for neg_tail in neg_tails:
                neg_tail_text = self.entity2text.get(neg_tail, neg_tail)
                neg_output = f"错误的尾实体为：{neg_tail_text}"
                results.append(("我们要做一些三元组的推理，这是负例", input_prompt, neg_output))
        
        return results
    
    def generate_sample_data(self):
        """生成采样数据并保存为pandas表格"""
        # 只使用训练数据来生成样本
        train_triples = self.load_dataset(self.train_file)
        
        # 构建所有正样本的集合，用于检查负样本是否存在
        positive_triples = set(train_triples)
        
        # 根据数据量和CPU核心数确定并行度
        num_workers = min(os.cpu_count() or 4, 8)  # 最多使用8个进程
        batch_size = max(1, len(train_triples) // (num_workers * 4))  # 每个进程处理的批次大小
        
        print(f"使用并行处理生成样本数据，进程数: {num_workers}，批大小: {batch_size}")
        
        # 存储所有结果
        all_results = []
        
        # 分批次处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有批次的任务
            futures = []
            for i in range(0, len(train_triples), batch_size):
                batch = train_triples[i:i+batch_size]
                futures.append(executor.submit(
                    self._generate_negative_tails_batch, 
                    triple_batch=batch, 
                    positive_triples=positive_triples
                ))
            
            # 收集所有结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成样本数据"):
                batch_result = future.result()
                all_results.extend(batch_result)
        
        # 创建DataFrame
        df = pd.DataFrame(all_results, columns=["instruction", "input", "output"])
        
        # 保存为tsv文件
        # df.to_csv(self.output_file, sep='\t', index=False, encoding='utf-8')
        df.sample(100).to_json(self.output_file)
        print(f"样本数据已保存至: {self.output_file}")
        print(f"生成的样本数据规模: {len(df)}条")
        
        # 显示一些样本
        print("\n生成的样本数据示例:")
        print(df.head())
    
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 创建数据处理器实例
    processor = KGDataProcessor()
    
    # 生成样本数据
    processor.generate_sample_data()