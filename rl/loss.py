import torch
import torch.nn.functional as F
from util.etc import topk_filter, Color

debug = False

def loss1(logits: torch.Tensor, target_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, topk, temperature):
    actions = actions.unsqueeze(-1)
    q_val = torch.gather(logits, dim=2, index=actions).squeeze(-1)

    if topk > 0:
        logits = topk_filter(logits, topk)
        target_logits = topk_filter(target_logits, topk)

    v_val = logits.logsumexp(dim=-1)
    a_val = q_val - v_val

    t_q_val = torch.zeros_like(q_val)
    t_v_val = target_logits.logsumexp(dim=-1)
    t_a_val = torch.zeros_like(a_val)

    t_q_val[:,:-1] = t_v_val[:,1:]
    t_a_val[:,:-1] = t_v_val[:,1:] - t_v_val[:,:-1]

    terminal_v = t_v_val[:,-1]
    t_q_val[:,-1] = rewards
    t_a_val[:,-1] = rewards - terminal_v

    return F.mse_loss(a_val, t_a_val, reduction='none').mean(dim=1)

def loss2(logits: torch.Tensor, target_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, topk, temperature):
    actions = actions.unsqueeze(-1)
    q_val = torch.gather(logits, dim=2, index=actions).squeeze()
    
    if topk > 0:
        logits = topk_filter(logits, topk)
        target_logits = topk_filter(target_logits, topk)

    v_val = logits.logsumexp(dim=-1)
    a_val = q_val - v_val

    t_v_val = target_logits.logsumexp(dim=-1)
    reversed_a_val = a_val.flip(dims=[-1])
    acc_a_val = reversed_a_val.cumsum(dim=-1)
    a_val = acc_a_val.flip(dims=[-1])

    return F.mse_loss(a_val, rewards.view(-1, 1) - t_v_val, reduction='none').mean(dim=1)
