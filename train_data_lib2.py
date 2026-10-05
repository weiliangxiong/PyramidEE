import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import yaml
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 配置加载 ======================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ====================== 2. 桶ID判断函数 ======================
def get_real_bucket_id(seq_len, buckets):
    num_buckets = len(buckets)
    for i, (s, e) in enumerate(buckets):
        if i == num_buckets - 1:
            if seq_len >= s:
                return i
        else:
            if s <= seq_len < e:
                return i
    return num_buckets - 1

# ====================== 3. 缓存数据集 ======================
class BucketCachedDataset(Dataset):
    def __init__(self, cache_dir: str, hash_code: str):
        self.cache_base = os.path.join(cache_dir, hash_code)
        self.meta_path = f"{self.cache_base}_meta.json"
        self.index_path = f"{self.cache_base}_index.json"
        self.data_path = f"{self.cache_base}_data.bin"

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.total_samples = self.meta["total_samples"]
        self.pad_token_id = self.meta["pad_token_id"]

        with open(self.index_path, 'r', encoding='utf-8') as f:
            self.index_data = json.load(f)

        self.seq_lens = np.array([item["seq_len"] for item in self.index_data], dtype=np.int32)
        self.offsets = np.array([item["offset"] for item in self.index_data], dtype=np.int64)
        self.tokens_lens = np.array([item["tokens_len"] for item in self.index_data], dtype=np.int32)
        self.data_file = None

    def _get_file(self):
        if self.data_file is None:
            self.data_file = open(self.data_path, 'rb')
        return self.data_file

    def __len__(self):
        return self.total_samples

    # def __getitem__(self, idx):
    #     f = self._get_file()
    #     f.seek(self.offsets[idx])
    #     tokens = np.frombuffer(f.read(self.tokens_lens[idx]), dtype=np.int16)
    #     tokens_copy = np.array(tokens.astype(np.int32), copy=True)
    #     #return {"input_ids": torch.from_numpy(tokens.astype(np.int32)), "seq_len": int(self.seq_lens[idx])}
    #     return {"input_ids": torch.from_numpy(tokens_copy), "seq_len": int(self.seq_lens[idx])}

    def __getitem__(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        raw_bytes = f.read(self.tokens_lens[idx])
        # 直接bytes → tensor，绕过numpy
        input_ids = torch.frombuffer(raw_bytes, dtype=torch.uint16).to(torch.int32).clone()
        #print(input_ids)
        # from transformers import AutoTokenizer
        # tok = AutoTokenizer.from_pretrained(
        #         "/root/autodl-tmp/chatglm3_token",
        #         trust_remote_code=True,          # ChatGLM3 必须
        #         local_files_only=True,           # 纯离线，不连 HF
        #     )
            
        # # input_ids 可以是 list / numpy / torch 一维
        # text = tok.decode(input_ids, skip_special_tokens=True)
        # print(text)
        
        return {"input_ids": input_ids, "seq_len": int(self.seq_lens[idx])}

    # def collate_fn(self, batch):
    #     seq_lens = [x["seq_len"] for x in batch]
    #     max_len = max(seq_lens)
    #     #if max_len>1024:
    #     #    max_len=1024

        
        
    #     bs = len(batch)
    #     input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
    #     labels = torch.full((bs, max_len), -100, dtype=torch.long)
    #     for i, x in enumerate(batch):
    #         #print(x["input_ids"].min())
    #         #print(x["input_ids"].max())
    #         sl = x["seq_len"]
    #         input_ids[i, :sl] = x["input_ids"][:sl]
    #         labels[i, :sl] = x["input_ids"][:sl]
    #     # print(input_ids)
    #     # print(input_ids.min())
    #     # print(input_ids.max())
    #     # print(self.pad_token_id)
    #     return input_ids, labels, torch.tensor(seq_lens)
    def collate_fn(self, batch):
        seq_lens = [x["seq_len"] for x in batch]
        max_len = max(seq_lens)
        
        # 限制最大长度为 1024
        MAX_LEN = 1024
        max_len = min(max_len, MAX_LEN)
        
        bs = len(batch)
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)
        
        for i, x in enumerate(batch):
            sl = x["seq_len"]
            # 实际拷贝的长度：取 seq_len 和 max_len 的较小值
            effective_len = min(sl, max_len)
            
            input_ids[i, :effective_len] = x["input_ids"][:effective_len]
            labels[i, :effective_len] = x["input_ids"][:effective_len]
        
        # seq_lens 也要反映截断后的真实长度
        truncated_seq_lens = [min(sl, max_len) for sl in seq_lens]
        
        return input_ids, labels, torch.tensor(truncated_seq_lens)

# ====================== 4. 最终版：桶样本数为batch_size整数倍 ======================
class DebugRoundRobinSampler(Sampler):
    def __init__(self, dataset, batch_size, buckets):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = buckets
        self.num_buckets = len(buckets)

        # 兼容 torch.utils.data.Subset
        self.original_ds = dataset
        self.subset_indices = None
        if isinstance(dataset, torch.utils.data.Subset):
            self.original_ds = dataset.dataset
            self.subset_indices = dataset.indices
            self.current_seq_lens = self.original_ds.seq_lens[self.subset_indices]
            self.total_samples = len(self.subset_indices)
        else:
            self.current_seq_lens = self.original_ds.seq_lens
            self.subset_indices = list(range(len(self.original_ds)))
            self.total_samples = len(self.original_ds)

        # 分桶（最后一个桶：≥最小值）
        self.bucket_indices = [[] for _ in buckets]
        num_buckets = len(buckets)
        for local_idx, sl in enumerate(self.current_seq_lens):
            global_idx = self.subset_indices[local_idx]
            for i, (s, e) in enumerate(buckets):
                if i == num_buckets - 1:
                    if sl >= s:
                        self.bucket_indices[i].append(global_idx)
                        break
                else:
                    if s <= sl < e:
                        self.bucket_indices[i].append(global_idx)
                        break

        # ==============================================
        # 核心新增：裁剪每个桶，使样本数为batch_size的整数倍
        # ==============================================
        self.original_bucket_counts = [len(idx_list) for idx_list in self.bucket_indices]
        for i in range(len(self.bucket_indices)):
            idx_list = self.bucket_indices[i]
            # 计算需要保留的样本数（向下取整到batch_size的整数倍）
            keep_num = (len(idx_list) // batch_size) * batch_size
            # 裁剪索引列表
            self.bucket_indices[i] = idx_list[:keep_num]

        # 分batch
        self.batches = []
        for idx_list in self.bucket_indices:
            self.batches.append([idx_list[i:i+batch_size] for i in range(0, len(idx_list), batch_size)])

        self.counter = [0]*self.num_buckets

        # 计算一个epoch的总batch数
        self.total_batches = sum(len(batch_list) for batch_list in self.batches)

        # 调试打印：新增裁剪前后对比
        print("\n===== 调试：样本统计信息 =====")
        print(f"当前数据集总样本数: {self.total_samples}")
        print(f"一个epoch总batch数: {self.total_batches}")
        total_bucket_samples = 0
        for i, (s,e) in enumerate(buckets):
            original_num = self.original_bucket_counts[i]
            final_num = len(self.bucket_indices[i])
            dropped_num = original_num - final_num
            total_bucket_samples += final_num
            print(f"桶{i} 区间:[{s},{e}) | 原样本数:{original_num} | 裁剪后:{final_num} | 丢弃:{dropped_num} | batch数:{len(self.batches[i])}")
        print(f"分桶后所有桶样本数之和: {total_bucket_samples}")
        print("=============================\n")

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        while True:
            for bucket_id in range(self.num_buckets):
                batch_list = self.batches[bucket_id]
                if not batch_list:
                    continue
                current_batch = batch_list[self.counter[bucket_id] % len(batch_list)]
                self.counter[bucket_id] += 1
                yield current_batch

# 在你的代码里添加这个类，或者修改原有的 DebugRoundRobinSampler
class DistributedDebugRoundRobinSampler:
    def __init__(self, dataset, batch_size, buckets, num_replicas=None, rank=None, **kwargs):
        # 只保存参数，不在这里生成任何列表
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = buckets
        self.kwargs = kwargs
        
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = num_replicas
        self.rank = rank
        
        # 初始化 base_sampler（它内部可能做了分桶统计，这是轻量的）
        self.base_sampler = DebugRoundRobinSampler(dataset, batch_size, buckets, **kwargs)
    
    def __iter__(self):
        # ✅ 关键：直接迭代 base_sampler，不调用 list()
        # 用 enumerate 获取全局 batch 序号，只 yield 属于当前 rank 的 batch
        for global_batch_idx, batch_indices in enumerate(self.base_sampler):
            if global_batch_idx % self.num_replicas == self.rank:
                yield batch_indices
    
    def __len__(self):
        # 估算当前 rank 的 batch 数量（向上取整）
        # 如果 base_sampler 有 __len__，可以用它来算
        if hasattr(self.base_sampler, '__len__'):
            total_batches = len(self.base_sampler)
            return (total_batches + self.num_replicas - 1 - self.rank) // self.num_replicas
        return 0
        

# ====================== 5. 测试代码 ======================
if __name__ == "__main__":
    config = load_config()
    BUCKETS = config["length_buckets"]
    args = type('Args', (), {'num_workers': 0})()

    # 加载完整数据集
    full_train_ds = BucketCachedDataset("/data/weilx/compact_cache2", config["hash_code"])
    
    # 划分训练/验证集
    val_size = 1000
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, len(full_train_ds)))

    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

    # 创建采样器
    val_sampler = DebugRoundRobinSampler(val_ds, config["batch_size"], BUCKETS)
    train_sampler = DebugRoundRobinSampler(train_ds, config["batch_size"], BUCKETS)

    # 创建DataLoader
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=full_train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=full_train_ds.collate_fn)

    # 测试
    print(f"训练集dataloader长度: {len(train_loader)}")
    print(f"验证集dataloader长度: {len(val_loader)}")

    for step, (input_ids, labels,seq_lengths) in enumerate(val_loader):
        print(seq_lengths)
        if step > 10:
            break  # 🔥 手动跳出