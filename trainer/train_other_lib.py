import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

#from model.model_minimind import MiniMindConfig


#from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
from datasets import load_dataset
warnings.filterwarnings('ignore')


from torch.utils.data import Dataset


#import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import math

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#import torch
import time
from torch.utils.data import Sampler, Subset
import os
import yaml


def my_mkdir(folder_name):
    #folder_name = "python"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"文件夹 '{folder_name}' 创建成功。")
    else:
        print(f"文件夹 '{folder_name}' 已存在。")

def val_epoch(val_loader, model,autocast_ctx,args,wandb=None):
    model.eval()
    total_loss = 0.0
    max_val_steps=60
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():

        for step, (input_ids, labels,seq_lengths) in enumerate(val_loader):
            #print(f'step:{step}')
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            seq_lengths = seq_lengths.to(args.device)
            with autocast_ctx:
                res = model(input_ids, labels=labels,seq_lengths=seq_lengths)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps
                current_loss = loss.item() * args.accumulation_steps
            total_loss+=current_loss
            if step >= max_val_steps-1:
                break  # 🔥 手动跳出

    
    #avg_loss = total_loss / len(val_loader)
    avg_loss = total_loss / max_val_steps
    Logger(f'Validation Loss: {avg_loss:.6f}')
    #print('--------------')
    #print(f'Validation Loss: {avg_loss:.6f}')
    if wandb:
        wandb.log({"val_loss": avg_loss})
    model.train()  # 回到训练模式
    return avg_loss


def train_epoch(optimizer,scaler,lm_config,model,autocast_ctx,epoch, loader, iters, args,val_loader,start_step=0,wandb=None):
     #train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    

    start_time = time.time()
    print(start_time)
    start_time2=time.time()
    for step, (input_ids, labels,seq_lengths) in enumerate(loader, start=start_step + 1):
        #print(seq_len)
        # print('input_ids')
        # print(input_ids.shape)
        # print('labels')
        # print(labels.shape)
        input_ids = input_ids.to(args.device)
        # try:
        #     print(input_ids.shape)
        #     x1, x2, _ = input_ids.shape
        # except Exception as e:
        #     print(e)
        #     continue
        # if input_ids.shape[0] !=8 :
        #     print(input_ids.shape)
        #     continue


        labels = labels.to(args.device)
        seq_lengths=seq_lengths.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels,seq_lengths=seq_lengths)
            loss = res.loss + res.aux_loss
            #loss = res.loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            train_time=time.time()-start_time2
            start_time2=time.time()
            
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            Logger(f'train_time:{train_time}')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            t1=time.time()
            val_epoch(val_loader, model,autocast_ctx,args,wandb)
            t2=time.time()
            Logger(f'val_time:{t2-t1}')

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_{step}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            #checkpoint_path=f'../checkpoints/{step}'
            checkpoint_path=f'{args.checkpoints_dir}/{step}'
            my_mkdir(checkpoint_path)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=checkpoint_path)
            model.train()
            del state_dict
            #time.sleep(60)

        del input_ids, labels, res, loss




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

  