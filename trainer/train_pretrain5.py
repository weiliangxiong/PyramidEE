from train_other_lib import *
from train_data_lib2 import *
#from model.model_minimind import MiniMindConfig
# import random
# import numpy as np

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# def setup_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# setup_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out_data4_model2_2048", help="模型保存目录")
    parser.add_argument("--checkpoints_dir", type=str, default="data4_model2_2048", help="模型保存目录")
    parser.add_argument("--data_cache_dir", type=str, default="/data/weilx/data_cache", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--learning_rate", type=float, default= 5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    #float32  bfloat16
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=320, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=4e4, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    #parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    #parser.add_argument("--data_path", type=str, default="/home/wlgc/weilx/dataset/data4.jsonl", help="预训练数据路径")
    parser.add_argument("--data_path", type=str, default="/home/wlgc/weilx/dataset/data_preprocess/data4_replace1.jsonl", help="预训练数据路径")
    #parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

     # 1. 加载配置
    config = load_config() # 默认读取当前目录下的 config.yaml
    args.save_dir = config.get("save_dir") 
    args.checkpoints_dir = config.get("checkpoints_dir") 
    args.data_cache_dir = config.get("data_cache_dir") 
    args.batch_size = config.get("batch_size") 
    args.accumulation_steps = config.get("accumulation_steps") 
    args.log_interval = config.get("log_interval") 
    args.hidden_size = config.get("hidden_size") 
    args.num_hidden_layers = config.get("num_hidden_layers") 
    args.data_path = config.get("data_path") 
    args.from_weight = config.get("from_weight") 
    args.from_resume = config.get("from_resume") 
    args.max_seq_len=config.get("max_seq_len") 
    args.save_interval=int(config.get("save_interval") )

    load_model_step=config.get("load_model_step") 


    args.model_class_path=config.get("model_class_path") 

    print('-'*30+'args'+'-'*30)
    print(args)
    print('-'*60)

    print('-'*30+'config'+'-'*30)
    print(config)
    print('-'*60)



    model_class_path = config.get("model_class_path") 

    #from model.model_minimind import MiniMindForCausalLM
    import importlib
    module_path = model_class_path  # 例如: "numpy", "os.path", "my_package.utils"
    attribute_name = "MiniMindConfig"  # 要导入的类/函数名 (如果需要)

    # 1. 导入模块
    imported_module = importlib.import_module(module_path)
    print(f"成功导入模块: {imported_module}")
    # 2. 如果需要从模块中获取特定的类、函数或变量
    if attribute_name:
        config_class = getattr(imported_module, attribute_name)
        print(f"成功获取属性 '{attribute_name}': {config_class}")


    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    #print(dist.get_rank())
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    

    if args.model_class_path=='model.model_minimind_attention_0313_3' or args.model_class_path=='model.model_minimind_attention_0313_4' \
        or args.model_class_path=='model.model_minimind_attention_0315_1':
        lm_config = config_class(
            hidden_size=1024,
            num_hidden_layers=20,
            layer_wise_hidden_size=[1024]*8+[1024]*12,
            layer_wise_num_attention_heads=[16]*20,
            layer_wise_num_key_value_heads=[2]*20,

            # layer_wise_hidden_size=[1536]*4 + [1280]*4 + [1024]*4 + [768]*4,
            # layer_wise_num_attention_heads=[32]*4 + [16]*4 + [8]*4 + [8]*4,
            # layer_wise_num_key_value_heads=[4]*4 + [4]*4 + [2]*4 + [2]*4,

            # layer_wise_hidden_size=[1024]*4 + [1024]*4 + [720]*4 + [720]*4,
            # layer_wise_num_attention_heads=[16]*4 + [16]*4 + [12]*4 + [12]*4,
            # layer_wise_num_key_value_heads=[4]*4 + [4]*4 + [2]*4 + [2]*4,

            # layer_wise_hidden_size=[720]*4 + [720]*4 + [1024]*4 + [1024]*4,
            # layer_wise_num_attention_heads=[12]*4 + [12]*4 + [16]*4 + [16]*4,
            # layer_wise_num_key_value_heads=[4]*4 + [4]*4 + [8]*4 + [8]*4,

            vocab_size=6400,
            flash_attn=True,
            use_moe=False,

        )
    elif args.model_class_path.find('attention')>0:
        lm_config = config_class(
            hidden_size=1024,          # 基础维度（词嵌入/输出）
            num_hidden_layers=16,      # 12层
            # ★ 分层hidden_size（核心修改）
            layer_wise_hidden_size=[1024]*4 + [1024]*4 + [1200]*4+[1200]*4,
            # ★ 分层注意力头数
            layer_wise_num_attention_heads=[16]*4 + [16]*4 + [24]*4+[24]*4,
            # ★ 分层KV头数
            layer_wise_num_key_value_heads=[2]*4 + [2]*4 + [2]*4+[2]*4,
            #flash_attn=True,
            inference_rope_scaling=True,
            use_moe=False
        )
    else:
        #
        #config_class
        lm_config = config_class(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    

    
    

    #step=12e4
    step=load_model_step
    checkpoint_path=f'{args.checkpoints_dir}/{step}'

    my_mkdir(f'{args.checkpoints_dir}')
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=checkpoint_path) if args.from_resume==1 else None
    #ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device, step=step)

    # # 验证所有buffer都在正确设备上
    # for name, buf in model.named_buffers():
    #     if "freqs_" in name:
    #         assert buf.device == args.device, f"Buffer {name} is on {buf.device}, expected {args.device}"
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # if dist.is_initialized():
    #     model = DistributedDataParallel(
    #         model, 
    #         device_ids=[local_rank],
    #         find_unused_parameters=True  # ← 加上这个
    #     )
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     static_graph=False,
    #     find_unused_parameters=True  # ← 必须加
    # )
    
        # 使用紧凑缓存
    print("构建数据集...")
    start_time = time.time()
    
    # dataset = ChunkedCompactCachedDataset(
    #     data_path=args.data_path,  # 替换为你的数据文件
    #     tokenizer=tokenizer,
    #     max_length=2048,
    #     max_length2=args.max_seq_len,
    #     vocab_size=6400,
    #     cache_dir=args.data_cache_dir,
    #     force_rebuild=False,  # 第一次构建
    #     max_samples=2000e4,  # 测试用，限制样本数
    #     use_mmap=False,  # 分块模式下不使用mmap
    #     chunk_size=500000,  # 每块2000个样本
    #     max_cached_chunks=3  # 最大缓存3个块
    # )
    
    # print(f"数据集构建耗时: {time.time() - start_time:.2f}秒")
    
    # # 获取统计信息
    # stats = dataset.get_statistics()
    # print(f"\n数据集统计信息:")
    # for key, value in stats.items():
    #     if key != 'length_distribution':
    #         print(f"  {key}: {value}")
    BUCKETS = config["length_buckets"]
    print(f"===== 调试：配置桶区间 {BUCKETS} =====")

    # 加载数据
    dataset = BucketCachedDataset(args.data_cache_dir, config["hash_code"])

    full_train_ds = dataset

            # 随机分割为训练集和验证集（例如 9:1 比例）
    #val_size = int(0.1 * len(full_train_ds))  # 验证集大小为总数据的10%
    # val_size=500
    # train_size = len(full_train_ds) - val_size
    base_seed = 1337
    #torch.manual_seed(base_seed)
    #torch.cuda.manual_seed(base_seed)

    # 固定随机种子，确保分割结果可复现
    # generator = torch.Generator().manual_seed(base_seed)  # 与全局种子保持一致

    # train_ds, val_ds = torch.utils.data.random_split(
    #     full_train_ds,
    #     [train_size, val_size],
    #     generator=generator
    # )


    # # 生成固定的验证集索引，这里取前 val_size 个样本
    # val_indices = list(range(val_size))
    # train_indices = list(range(val_size, len(full_train_ds)))


    max_len = len(full_train_ds)
    if hasattr(full_train_ds, 'seq_lens'):
        max_len = min(max_len, len(full_train_ds.seq_lens))
    elif hasattr(full_train_ds, 'dataset') and hasattr(full_train_ds.dataset, 'seq_lens'):
        max_len = min(max_len, len(full_train_ds.dataset.seq_lens))
    
    # 验证集大小，确保不超过 max_len
    val_size = 500
    if val_size >= max_len:
        val_size = int(max_len * 0.1)  # 如果数据集太小，按比例分
    
    # 生成不越界的索引
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, max_len))  # 最大到 max_len-1，不会越界


    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

    # val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=args.batch_size,
    #     shuffle=False,  # 验证集不打乱
    #     sampler=val_sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )

    #args.short_threshold = args.short_threshold or (args.max_seq_len // 2)
    #Logger(f"✅ 长短样本分组配置：短样本≤{args.short_threshold}，长样本>{args.short_threshold}")
    
    # 分布式参数
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # # # 创建长度分组Sampler（核心：先短后长）
    # train_sampler = LengthGroupedSampler(
    #     dataset=train_ds,
    #     batch_size=args.batch_size,
    #     short_threshold=args.short_threshold,
    #     rank=rank,
    #     world_size=world_size,
    #     shuffle=False
    # )
    

    # Logger(f"✅ 训练集分组完成：总样本数={len(train_ds)}，短样本数={len(train_sampler.short_indices)}，长样本数={len(train_sampler.long_indices)}")

    # 初始化新的Sampler：顺序采样，batch内按长度升序排序
    
    # val_sampler = LengthOrderedBatchSampler(
    #     dataset=val_ds,
    #     batch_size=args.batch_size,
    #     index_json_dir=args.data_cache_dir,
    #     sort_ascending=True,  # batch内短样本在前、长样本在后
    #     skip=train_size
    # )
   
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        #train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); 
        #indices = torch.randperm(len(train_ds)).tolist()
        indices = list(range(len(train_ds)))
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # train_skip=0
        # if 
        # train_sampler = LengthOrderedBatchSampler(
        #     dataset=train_ds,
        #     batch_size=args.batch_size,
        #     index_json_dir=args.data_cache_dir,
        #     sort_ascending=True,  # batch内短样本在前、长样本在后
        #     skip=skip*args.batch_size
           
        # )

        # val_sampler = DebugRoundRobinSampler(val_ds, config["batch_size"], BUCKETS)
        # train_sampler = DebugRoundRobinSampler(train_ds, config["batch_size"], BUCKETS)

        train_sampler = DistributedDebugRoundRobinSampler(
            dataset=train_ds,
            batch_size=config["batch_size"],
            buckets=BUCKETS,
            num_replicas=world_size,
            rank=rank
        )
        
        val_sampler = DistributedDebugRoundRobinSampler(
            dataset=val_ds,
            batch_size=config["batch_size"],
            buckets=BUCKETS,
            num_replicas=world_size,
            rank=rank
        )
        

        #dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn, num_workers=0)
    

        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,collate_fn=dataset.collate_fn)
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(optimizer,scaler,lm_config,model,autocast_ctx,epoch, train_loader, len(train_loader)+skip, args,val_loader,start_step, wandb)
        else:
            train_epoch(optimizer,scaler,lm_config,model,autocast_ctx,epoch, train_loader, len(train_loader),args,val_loader,0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

