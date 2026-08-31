# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config (保持不变)
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 16,
            num_hidden_layers: int = 8, # 物理总层数
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            # --- 早退机制相关配置 ---
            use_early_exit_loss: bool = True,
            use_gradual_early_exit: bool = True,
            total_training_steps: int = 200000,
            early_exit_escale: float = 0.5,
            # --- MOE 配置 ---
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        
        self.use_early_exit_loss = use_early_exit_loss
        self.use_gradual_early_exit = use_gradual_early_exit
        self.total_training_steps = total_training_steps
        self.early_exit_escale = early_exit_escale

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model (Core Fix)
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# ... (RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv, Attention, FeedForward 定义保持不变，为节省篇幅此处省略，直接使用) ...
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads, self.n_local_kv_heads = args.num_attention_heads, self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout, self.resid_dropout = nn.Dropout(args.dropout), nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq, xk, xv = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim), xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim), xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk, xv = torch.cat([past_key_value[0], xk], dim=1), torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2)
        
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                scores += extended_attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = scores @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, layer_id):
        super().__init__()
        intermediate_size = config.intermediate_size or int(config.hidden_size * 8 / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 保持原有的动态中间层大小逻辑
        # intermediate_size = 10240 - 512 * layer_id + 2048 + 2048
        # if layer_id > 15: intermediate_size += 256
        #intermediate_size=10240-640*layer_id+2048+2048+2048-1024+64
        intermediate_size=10240-512*layer_id+2048+2048

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config, layer_id) # 简化：暂不考虑MOE
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.layers_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 8),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 8, 1)
        )

    
    # def _compute_layers_to_use(self, seq_length: int) -> int:
    #     if seq_length < 150: return 10
    #     elif seq_length < 300: return 11
    #     elif seq_length < 500: return 12
    #     elif seq_length < 750: return 13
    #     elif seq_length < 990: return 14
    #     else: return 15   
    def _compute_layers_to_use(self, seq_length: int) -> int:
        """根据序列长度计算需要使用的层数，使用MLP实现"""
        # 将序列长度转换为张量
        seq_tensor = torch.tensor([[seq_length]], dtype=torch.float32, device=next(self.parameters()).device)
        
        # 归一化处理，除以最大序列长度
        seq_tensor = seq_tensor / self.config.max_position_embeddings
        
        # 通过MLP计算
        output = self.layers_mlp(seq_tensor)
        
        # 将输出映射到合理的层数范围
        # 原始逻辑中，层数在8到15之间
        # 使用sigmoid将输出限制在0-1之间，然后映射到8-15
        layers = torch.sigmoid(output) *7  + 8  # 7是范围(15-8)，8是起始值
        
        # 四舍五入到最接近的整数
        layers = torch.round(layers).int().item()
        
        # 确保层数在有效范围内
        layers = max(8, min(15, layers))
        
        return layers
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                seq_lengths: Optional[torch.Tensor] = None,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        if seq_lengths is None:
            pad_token_id = getattr(self.config, 'pad_token_id', 100)
            seq_lengths = torch.sum(input_ids != pad_token_id, dim=1)
        
        # 【修复点1】：计算 layers_per_sample 并确定本批次的最大深度
        # 即使批次内等长，这也是必须的，它决定了我们要算多少层
        layers_per_sample = torch.tensor([self._compute_layers_to_use(length.item()) 
                                          for length in seq_lengths], 
                                         device=input_ids.device)
        
        # 确定当前批次实际需要运行的最大层数 L_max
        # 因为用户说批次内等长，所以 max = min = 任意一个元素
        L_max = layers_per_sample.max().item()
        # 安全检查：不能超过物理层数
        L_max = min(L_max, self.config.num_hidden_layers)

        presents = []
        # 【修复点2】：只收集到 L_max 层的隐藏状态
        all_layer_hiddens = []

        # 核心循环：只跑到 L_max 层
        for layer_idx in range(L_max):
            # 执行层计算 (全量计算，因为批次内等长)
            updated_hidden, present = self.layers[layer_idx](
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values[layer_idx],
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            
            hidden_states = updated_hidden
            
            # 保存当前层输出
            all_layer_hiddens.append(hidden_states)
            
            presents.append(present)
        
        # 对于没跑到的层 (L_max 之后的层)，past_key_value 保持原样或填 None
        # 这主要是为了推理时的 Cache 兼容，训练时一般 use_cache=False
        for layer_idx in range(L_max, self.config.num_hidden_layers):
            presents.append(None)

        # 最后的 Norm (只对最终输出做 Norm，中间层的 Norm 在计算损失时单独做)
        final_hidden_states = self.norm(hidden_states)

        aux_loss = hidden_states.new_zeros(1).squeeze()
        
        # 【修复点3】：返回 L_max，让外层知道 all_layer_hiddens 实际有多少层
        return final_hidden_states, all_layer_hiddens, L_max, presents, aux_loss

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                        MiniMind For CausalLM (Core Fix)
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long), persistent=False)

    def get_enabled_layers(self, L_effective):
        """
        修改版：接收动态的 L_effective (即当前批次的 L_max)
        """
        if not self.config.use_gradual_early_exit:
            return [True] * L_effective

        T = self.config.total_training_steps
        current_step = self.current_step

        # 【修复点4】：基于当前批次的实际层数 L_effective 计算间隔
        step_interval = T / (2 * L_effective)
        num_enabled = min(int(current_step / step_interval) + 1, L_effective)

        enabled = [False] * L_effective
        for i in range(num_enabled):
            enabled[L_effective - 1 - i] = True
        return enabled

    def get_early_exit_weights(self, L_effective, device, enabled_layers):
        """
        修改版：接收动态的 L_effective
        """
        escale = self.config.early_exit_escale
        e = torch.zeros(L_effective, device=device)
        
        for l in range(L_effective):
            if not enabled_layers[l]:
                e[l] = 0.0
                continue

            if l < L_effective - 1:
                e[l] = escale * sum(range(l + 1))
            else:
                e[l] = (L_effective - 1) + escale * sum(range(L_effective - 1))
        
        total = e.sum()
        weights = e / total if total > 0 else torch.zeros_like(e)
        return weights

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                seq_lengths: Optional[torch.Tensor] = None,
                current_step: int = 0,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        
        self.current_step = torch.tensor(current_step, dtype=torch.long, device=self.device)
        
        # 【修复点5】：解包 L_max
        final_hidden, all_layer_hiddens, L_max, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            seq_lengths=seq_lengths,
            **args
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(final_hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            total_loss = 0.0

            if self.config.use_early_exit_loss and self.training:
                # 【修复点6】：使用动态的 L_max 作为循环上限
                enabled_layers = self.get_enabled_layers(L_max)
                weights = self.get_early_exit_weights(L_max, final_hidden.device, enabled_layers)

                # 遍历范围是 0 到 L_max-1
                for l in range(L_max):
                    if not enabled_layers[l]:
                        continue
                    
                    # 取出第 l 层的隐藏状态
                    h = self.model.norm(all_layer_hiddens[l])
                    logits_l = self.lm_head(h)
                    shift_logits_l = logits_l[..., :-1, :].contiguous()
                    
                    loss_l = F.cross_entropy(
                        shift_logits_l.view(-1, shift_logits_l.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    total_loss += weights[l] * loss_l
                
                loss = total_loss
            else:
                # 验证 / 推理：只算最后一层
                shift_logits = logits[..., :-1, :].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
        output.aux_loss = aux_loss
        return output