import os
import json
import numpy as np
import hashlib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import gc
from typing import List, Dict, Any, Optional
from collections import OrderedDict


import json
from torch.utils.data import Sampler

import json
import os
from torch.utils.data import Sampler

import yaml

def load_config(config_path="config.yaml"):
    """
    加载YAML配置文件。
    Args:
        config_path (str): 配置文件的路径。
    Returns:
        dict: 包含所有配置参数的字典。
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


config = load_config() # 默认读取当前目录下的 config.yaml
hash_code = config.get("hash_code") 

class LengthOrderedBatchSampler(Sampler):
    """
    顺序采样形成batch，每个batch内的样本按长度排序
    自动发现指定目录下的索引JSON文件
    """
    def __init__(self, dataset, batch_size, index_json_dir, sort_ascending=True,skip=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_ascending = sort_ascending  # True=长度升序，False=长度降序

        # 1. 自动发现JSON文件
        json_files = []
        for file in os.listdir(index_json_dir):
            if file.endswith("_index.json"):
                json_files.append(file)
        
        if not json_files:
            raise FileNotFoundError(f"在目录 {index_json_dir} 中未找到 *_index.json 文件")
        
        # 使用找到的第一个文件
        json_path = os.path.join(index_json_dir, json_files[0])
        if len(json_files) > 1:
            print(f"警告：找到多个索引文件，使用 {json_files[0]}")
        
        # 2. 从JSON文件加载长度信息
        with open(json_path, 'r', encoding='utf-8') as f:
            length_data = json.load(f)  # 加载为列表
        
        # 3. 提取seq_len列表，与数据集顺序对应
        self.seq_lens = [item["seq_len"] for item in length_data]
        
        

        # # 4. 验证长度匹配
        # if len(self.seq_lens) != len(dataset):
        #     raise ValueError(f"JSON长度列表大小({len(self.seq_lens)})与数据集大小({len(dataset)})不匹配")
        
        # 5. 生成所有样本索引（顺序采样基础）
        self.all_indices = list(range(len(dataset)   ))
        #训练集中的断点续训
        if len(dataset)>3000 and skip>0:
            self.all_indices=self.all_indices[skip:]
        #处理验证集的情况
        else:
            self.seq_lens=self.seq_lens[skip:]
            
        
        # 6. 生成批次
        self._generate_batches()

    def _generate_batches(self):
        """顺序切分batch，每个batch内部按长度排序"""
        self.batches = []
        
        # 顺序遍历所有样本索引
        for i in range(0, len(self.all_indices), self.batch_size):
            # 获取当前batch的原始索引
            batch_indices = self.all_indices[i:i + self.batch_size]
            
            # 为当前batch创建(索引, 长度)对
            batch_with_len = [(idx, self.seq_lens[idx]) for idx in batch_indices]
            
            # 按长度排序
            sorted_batch = sorted(
                batch_with_len,
                key=lambda x: x[1],
                reverse=not self.sort_ascending
            )
            
            # 提取排序后的索引
            sorted_indices = [item[0] for item in sorted_batch]
            self.batches.append(sorted_indices)

    def __iter__(self):
        for batch in self.batches:
            #print(batch)
            yield batch

    def __len__(self):
        return len(self.batches)

class ChunkedCompactCachedDataset(Dataset):
    """
    分块加载的紧凑缓存数据集
    针对小vocab_size优化，支持分块加载以限制内存使用
    真正压缩存储：只存储实际token，不存储padding
    """
    def __init__(self, data_path, tokenizer, max_length=2048, max_length2=1024, vocab_size=6400, 
                 cache_dir="./compact_cache", force_rebuild=False, max_samples=None,
                 use_mmap=False,  # 分块模式默认不使用mmap
                 chunk_size=10000,  # 每个分块的样本数
                 max_cached_chunks=5,  # 最大缓存的块数
                 **kwargs):
        """
        参数:
            data_path: 原始数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            vocab_size: 词汇表大小
            cache_dir: 缓存目录
            force_rebuild: 是否强制重建缓存
            max_samples: 最大样本数限制
            use_mmap: 是否使用内存映射（分块模式下默认False）
            chunk_size: 每个分块包含的样本数
            max_cached_chunks: 内存中最大缓存的块数
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length2 = max_length2
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.data_path = data_path
        self.max_samples = int(max_samples)
        
        # 分块参数
        self.use_mmap = use_mmap
        self.chunk_size = chunk_size
        self.max_cached_chunks = max_cached_chunks
        
        # 验证vocab_size是否可以用int16存储
        assert vocab_size <= 32767, f"vocab_size {vocab_size} 超出int16范围"
        
        # 获取特殊token id
        self.bos_token_id = tokenizer.bos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id or 2
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存标识
        if hash_code=='none':
            cache_key = self._generate_cache_key()
        else:
            cache_key=hash_code
        self.cache_base = os.path.join(cache_dir, cache_key)
        
        # 文件路径
        self.meta_path = f"{self.cache_base}_meta.json"
        self.data_path_bin = f"{self.cache_base}_data.bin"
        self.index_path = f"{self.cache_base}_index.json"
        
        # 分块相关初始化
        self.chunk_cache = {}  # chunk_id -> 数据字典
        self.lru_order = []  # LRU队列，记录最近使用的chunk
        self.chunk_memory_usage = {}  # chunk_id -> 内存使用量(字节)
        self.total_cache_memory = 0  # 总缓存内存(字节)
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        if not force_rebuild and self._cache_valid():
            print(f"加载现有缓存: {self.cache_base}")
            self._load_cached_data()
        else:
            print(f"构建新缓存: {self.cache_base}")
            self._build_cache()
            self._load_cached_data()
        
        # 计算分块信息
        self.num_chunks = (self.total_samples + self.chunk_size - 1) // self.chunk_size
        self.chunk_start_indices = np.zeros(self.num_chunks, dtype=np.int64)
        
        for i in range(self.num_chunks):
            self.chunk_start_indices[i] = i * self.chunk_size
        
        print(f"数据集分块: {self.total_samples}个样本, {self.num_chunks}个分块, 每块{self.chunk_size}样本")
    
    def _generate_cache_key(self):
        """生成缓存key，基于文件内容和参数"""
        file_hash = ""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        key_str = f"{file_hash}_{self.tokenizer.__class__.__name__}_{self.max_length}_{self.vocab_size}"
        if self.max_samples:
            key_str += f"_{self.max_samples}"
        
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _cache_valid(self):
        """检查缓存是否有效"""
        required_files = [self.meta_path, self.data_path_bin, self.index_path]
        if not all(os.path.exists(f) for f in required_files):
            return False
        
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('max_length') != self.max_length:
                return False
            if meta.get('vocab_size') != self.vocab_size:
                return False
        except:
            return False
        
        return True
    
    def _build_cache(self):
        """构建缓存，只存储实际token，不存储padding"""
        print("加载原始数据...")
        dataset = load_dataset('json', data_files=self.data_path, split='train')
        
        # 限制样本数
        if self.max_samples:
            texts = dataset[0:min(self.max_samples, len(dataset))]['text']
        else:
            texts = dataset['text']
        
        self.total_samples = len(texts)
        print(f"处理 {self.total_samples} 个样本...")
        
        # 存储结构
        index_data = []
        
        # 打开二进制文件写入
        with open(self.data_path_bin, 'wb') as f_data:
            offset = 0
            
            pbar = tqdm(total=self.total_samples, desc="构建缓存")
            
            batch_size = 10000
            for i in range(0, self.total_samples, batch_size):
                batch_end = min(i + batch_size, self.total_samples)
                batch_texts = texts[i:batch_end]
                
                # 批量编码
                batch_encoding = self.tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    max_length=self.max_length - 2,  # 留出BOS和EOS的位置
                    truncation=True,
                    return_attention_mask=False,
                    return_tensors=None
                )
                
                # 处理批次
                for j, tokens in enumerate(batch_encoding.input_ids):
                    sample_idx = i + j
                    
                    # 处理单个样本
                    tokens_with_special = [self.bos_token_id] + tokens + [self.eos_token_id]
                    seq_len = len(tokens_with_special)
                    
                    # 确保长度不超过最大长度
                    if seq_len > self.max_length:
                        tokens_with_special = tokens_with_special[:self.max_length]
                        seq_len = self.max_length
                    
                    # 转换为int16数组
                    tokens_np = np.array(tokens_with_special, dtype=np.int16)
                    
                    # 记录索引信息
                    index_entry = {
                        'offset': offset,
                        'tokens_len': tokens_np.nbytes,  # 实际token的字节数
                        'seq_len': seq_len
                    }
                    index_data.append(index_entry)
                    
                    # 写入二进制文件
                    tokens_np.tofile(f_data)
                    
                    offset += tokens_np.nbytes
                
                pbar.update(batch_end - i)
            
            pbar.close()
        
        # 保存索引文件
        with open(self.index_path, 'w') as f:
            json.dump(index_data, f)
        
        # 保存元数据
        meta = {
            'total_samples': self.total_samples,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'tokenizer': self.tokenizer.__class__.__name__,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'data_size': os.path.getsize(self.data_path_bin),
            'index_size': os.path.getsize(self.index_path)
        }
        
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f)
        
        # 计算压缩率
        data_mb = meta['data_size'] / 1024 / 1024
        index_mb = meta['index_size'] / 1024 / 1024
        
        # 估计未压缩的大小
        uncompressed_size = self.total_samples * self.max_length * 4  # 假设float32
        
        print(f"缓存构建完成:")
        print(f"  - 数据文件: {self.data_path_bin} ({data_mb:.2f} MB)")
        print(f"  - 索引文件: {self.index_path} ({index_mb:.2f} KB)")
        print(f"  - 每个样本平均大小: {meta['data_size'] / self.total_samples:.1f} 字节")
        print(f"  - 压缩比: {uncompressed_size / meta['data_size']:.2f}x")
        
        # 内存优化：清理原始数据
        del texts
        gc.collect()
    
    def _load_cached_data(self):
        """加载缓存数据"""
        # 分块模式下强制使用文件模式
        if self.chunk_size > 0:
            self.use_mmap = False
        
        if self.use_mmap:
            self._load_memory_mapped()
        else:
            self._load_file_based()
    
    def _load_file_based(self):
        """基于文件加载缓存"""
        # 加载元数据
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
            self.total_samples = meta['total_samples']
            self.pad_token_id = meta.get('pad_token_id', 0)
            self.bos_token_id = meta.get('bos_token_id', 0)
            self.eos_token_id = meta.get('eos_token_id', 2)
        
        # 加载索引文件
        with open(self.index_path, 'r') as f:
            index_data = json.load(f)
        
        # 预加载索引到内存数组，提高访问速度
        self.offsets = np.zeros(self.total_samples, dtype=np.int64)
        self.tokens_lens = np.zeros(self.total_samples, dtype=np.int32)  # 实际token的字节数
        self.seq_lens = np.zeros(self.total_samples, dtype=np.int16)
        
        for i, item in enumerate(index_data):
            self.offsets[i] = item['offset']
            self.tokens_lens[i] = item['tokens_len']
            self.seq_lens[i] = item['seq_len']
        
        # 打开数据文件
        self.data_file = open(self.data_path_bin, 'rb')
        
        print(f"文件加载完成: {self.total_samples} 个样本")
        print(f"索引内存占用: {self.offsets.nbytes + self.tokens_lens.nbytes + self.seq_lens.nbytes:,} 字节")
    
    def _load_memory_mapped(self):
        """使用内存映射加载数据"""
        # 加载元数据
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
            self.total_samples = meta['total_samples']
            self.pad_token_id = meta.get('pad_token_id', 0)
            self.bos_token_id = meta.get('bos_token_id', 0)
            self.eos_token_id = meta.get('eos_token_id', 2)
        
        # 加载索引
        with open(self.index_path, 'r') as f:
            index_data = json.load(f)
        
        # 预加载索引
        self.offsets = np.zeros(self.total_samples, dtype=np.int64)
        self.seq_lens = np.zeros(self.total_samples, dtype=np.int16)
        
        for i, item in enumerate(index_data):
            # 转换为元素偏移量（int16元素，2字节）
            self.offsets[i] = item['offset'] // 2
            self.seq_lens[i] = item['seq_len']
        
        print(f"内存映射加载完成: {self.total_samples} 个样本")
        print(f"索引内存占用: {self.offsets.nbytes + self.seq_lens.nbytes:,} 字节")
    
    def _get_chunk_id(self, index):
        """获取样本所属的分块ID"""
        return index // self.chunk_size
    
    def _load_chunk(self, chunk_id):
        """加载指定分块到内存"""
        if chunk_id in self.chunk_cache:
            # 更新LRU顺序
            if chunk_id in self.lru_order:
                self.lru_order.remove(chunk_id)
            self.lru_order.append(chunk_id)
            return
        
        # 如果缓存已满，移除最久未使用的分块
        if len(self.chunk_cache) >= self.max_cached_chunks:
            self._evict_oldest_chunk()
        
        # 计算分块范围
        start_idx = chunk_id * self.chunk_size
        end_idx = min((chunk_id + 1) * self.chunk_size, self.total_samples)
        chunk_samples = end_idx - start_idx
        
        # 预分配内存
        chunk_data = []
        chunk_seq_lens = []
        
        # 估算内存占用
        estimated_memory = 0
        
        # 读取分块数据
        for idx in range(start_idx, end_idx):
            # 文件读取模式
            offset = int(self.offsets[idx])
            seq_len = int(self.seq_lens[idx])
            
            # 从文件读取
            self.data_file.seek(offset)
            tokens_len = seq_len * 2  # 每个int16占2字节
            tokens_bytes = self.data_file.read(tokens_len)
            tokens = np.frombuffer(tokens_bytes, dtype=np.int16)
            
            # 存储数据
            chunk_data.append(tokens)
            chunk_seq_lens.append(seq_len)
            
            # 估算内存（tokens + numpy数组开销 + python列表开销）
            estimated_memory += tokens.nbytes + 64
        
        # 将分块数据存储在缓存中
        self.chunk_cache[chunk_id] = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'data': chunk_data,  # 存储实际token数据
            'seq_lens': chunk_seq_lens,
            'memory_used': estimated_memory
        }
        
        # 更新LRU队列
        self.lru_order.append(chunk_id)
        self.total_cache_memory += estimated_memory
        self.chunk_memory_usage[chunk_id] = estimated_memory
        
        # 输出调试信息
        if chunk_id % 10 == 0:  # 每10个分块输出一次
            print(f"加载分块 {chunk_id}: 样本 {start_idx}-{end_idx}, "
                  f"内存 {estimated_memory/1024:.1f}KB")
    
    def _evict_oldest_chunk(self):
        """移除最久未使用的分块"""
        if not self.lru_order:
            return
        
        # 移除LRU队列中最老的chunk
        oldest_chunk_id = self.lru_order.pop(0)
        
        if oldest_chunk_id in self.chunk_cache:
            # 释放内存
            memory_freed = self.chunk_cache[oldest_chunk_id]['memory_used']
            del self.chunk_cache[oldest_chunk_id]
            
            if oldest_chunk_id in self.chunk_memory_usage:
                self.total_cache_memory -= self.chunk_memory_usage[oldest_chunk_id]
                del self.chunk_memory_usage[oldest_chunk_id]
            
            # 调用垃圾回收
            gc.collect()
            
            # 输出调试信息
            if oldest_chunk_id % 10 == 0:
                print(f"释放分块 {oldest_chunk_id}, 释放内存 {memory_freed/1024:.1f}KB, "
                      f"当前缓存内存: {self.total_cache_memory/1024:.1f}KB")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        """获取单个样本，支持分块缓存"""
        if self.chunk_size > 0:
            # 分块模式
            return self._get_item_chunked(index)
        elif self.use_mmap:
            # 内存映射模式
            return self._get_item_mmap(index)
        else:
            # 文件读取模式
            return self._get_item_file(index)
    
    def _get_item_chunked(self, index):
        """分块模式下获取样本"""
        # 获取所属分块
        chunk_id = self._get_chunk_id(index)
        
        # 确保分块已加载
        if chunk_id not in self.chunk_cache:
            self.cache_misses += 1
            self._load_chunk(chunk_id)
        else:
            self.cache_hits += 1
            # 更新LRU顺序
            if chunk_id in self.lru_order:
                self.lru_order.remove(chunk_id)
            self.lru_order.append(chunk_id)
        
        # 从缓存中获取数据
        chunk = self.chunk_cache[chunk_id]
        local_idx = index - chunk['start_idx']
        
        tokens = chunk['data'][local_idx]
        seq_len = chunk['seq_lens'][local_idx]
        
        return {
            'input_ids': torch.from_numpy(tokens.astype(np.int32)),
            'seq_len': seq_len
        }
    
    def _get_item_file(self, index):
        """从文件读取单个样本（非分块模式）"""
        # 计算读取位置
        offset = self.offsets[index]
        tokens_len = self.tokens_lens[index]
        seq_len = self.seq_lens[index]
        
        # 读取实际token
        self.data_file.seek(offset)
        tokens_bytes = self.data_file.read(tokens_len)
        tokens = np.frombuffer(tokens_bytes, dtype=np.int16).copy()
        
        return {
            'input_ids': torch.from_numpy(tokens.astype(np.int32)),
            'seq_len': seq_len
        }
    
    def _get_item_mmap(self, index):
        """从内存映射读取单个样本（非分块模式）"""
        offset = self.offsets[index]
        seq_len = self.seq_lens[index]
        
        # 从内存映射直接读取实际token
        tokens = self.mmap[offset:offset + seq_len].copy()
        
        return {
            'input_ids': torch.from_numpy(tokens.astype(np.int32)),
            'seq_len': seq_len
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]):
        #print(batch)
        """collate函数，用于DataLoader"""
        # 获取batch中所有序列的实际长度
        seq_lens = [item['seq_len'] for item in batch]
        
        # 找到batch内的最大长度（不超过设定的max_length）
        max_len = min(max(seq_lens), self.max_length2)
        
        # 初始化tensors
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # 填充数据
        for i, item in enumerate(batch):
            input_ids_tensor = item['input_ids']
            actual_len = min(item['seq_len'], max_len)
            
            # 填充input_ids
            input_ids[i, :actual_len] = input_ids_tensor[:actual_len]
            
            # attention_mask
            attention_mask[i, :actual_len] = 1
            
            # labels: 对于语言模型，labels就是input_ids（padding部分为-100）
            labels[i, :actual_len] = input_ids_tensor[:actual_len]

            #labels = [x if x != self.pad_token_id else -100 for x in input_ids]

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(labels.shape)
        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'labels': labels
        # }
        return input_ids,labels,torch.tensor(seq_lens)
    
    def get_statistics(self):
        """获取数据集统计信息"""
        if not hasattr(self, 'seq_lens'):
            return {}
        
        seq_lens = self.seq_lens.astype(np.int32)
        stats = {
            'total_samples': self.total_samples,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'avg_seq_len': float(np.mean(seq_lens)),
            'min_seq_len': int(np.min(seq_lens)),
            'max_seq_len': int(np.max(seq_lens)),
            'median_seq_len': int(np.median(seq_lens)),
            'std_seq_len': float(np.std(seq_lens)),
        }
        
        # 统计不同长度的分布
        hist, bins = np.histogram(seq_lens, bins=10)
        stats['length_distribution'] = {
            'bins': bins.tolist(),
            'counts': hist.tolist()
        }
        
        return stats
    
    def get_cache_info(self):
        """获取缓存统计信息"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_cache_memory_kb': self.total_cache_memory / 1024,
            'cached_chunks': len(self.chunk_cache),
            'max_cached_chunks': self.max_cached_chunks,
            'chunk_size': self.chunk_size,
            'num_chunks': self.num_chunks
        }
    
    def clear_cache(self):
        """清空所有缓存"""
        self.chunk_cache.clear()
        self.lru_order.clear()
        self.chunk_memory_usage.clear()
        self.total_cache_memory = 0

        try:
            gc.collect()
        except Exception:
            pass

        print(f"缓存已清空，当前内存使用: {self.total_cache_memory/1024:.1f}KB")
    
    def close(self):
        """关闭文件"""
        if hasattr(self, 'data_file'):
            self.data_file.close()
        
        # 清空缓存
        self.clear_cache()
    
    def __del__(self):
        """析构时关闭文件"""
        self.close()


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn
    )


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用紧凑缓存
    print("构建数据集...")
    start_time = time.time()
    
    dataset = ChunkedCompactCachedDataset(
        data_path="/home/wlgc/weilx/dataset/data4.jsonl",  # 替换为你的数据文件
        tokenizer=tokenizer,
        max_length=2048,
        vocab_size=6400,
        cache_dir="/data/weilx/compact_cache",
        force_rebuild=False,  # 第一次构建
        max_samples=2000e4,  # 测试用，限制样本数
        use_mmap=False,  # 分块模式下不使用mmap
        chunk_size=10000,  # 每块2000个样本
        max_cached_chunks=3  # 最大缓存3个块
    )
    
    print(f"数据集构建耗时: {time.time() - start_time:.2f}秒")
    
    # 获取统计信息
    stats = dataset.get_statistics()
    print(f"\n数据集统计信息:")
    for key, value in stats.items():
        if key != 'length_distribution':
            print(f"  {key}: {value}")
    
    # 测试读取
    print("\n测试单个样本读取...")
    for i in range(10):
        item = dataset[i]
        print(f"样本 {i}: 序列长度={item['seq_len']}, input_ids形状={item['input_ids'].shape}")
    
    # 测试连续读取（模拟顺序访问）
    print("\n测试连续读取（模拟顺序访问）...")
    start_time = time.time()
    for i in range(0, 100, 1):
        item = dataset[i]
    print(f"连续读取100个样本耗时: {time.time() - start_time:.3f}秒")
    
    # 获取缓存信息
    cache_info = dataset.get_cache_info()
    print(f"\n缓存统计信息:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # 测试DataLoader
    print("\n测试DataLoader...")
    data_loader = create_data_loader(
        dataset,
        batch_size=4,
        shuffle=False,  # 顺序访问
        num_workers=0,
        pin_memory=False
    )
    
    batch = next(iter(data_loader))
    print(f"批次数据:")
    print(f"  input_ids形状: {batch['input_ids'].shape}")
    print(f"  attention_mask形状: {batch['attention_mask'].shape}")
    print(f"  labels形状: {batch['labels'].shape}")
    print(f"  padding比例: {(batch['attention_mask'] == 0).sum().item() / batch['attention_mask'].numel():.2%}")
    
    # 关闭数据集
    dataset.close()