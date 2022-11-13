import torch
from rl.policies.mlp_policy import MlpPolicy

def sample(policy: MlpPolicy, target_policy: MlpPolicy, env, resolved_batch, shot_num):
    acc_logits = []
    acc_target_logits = []

    target_policy.eval()

    states = env.reset(resolved_batch, 'train')
    for step in range(shot_num):
        actions, logits = policy.get_actions_logits(states)
        target_logits = target_policy.get_logits(states)
        states, rewards = env.step(actions)
        acc_logits.append(logits.unsqueeze(1))
        acc_target_logits.append(target_logits.unsqueeze(1))

    return torch.cat(acc_logits, dim=1), torch.cat(acc_target_logits, dim=1), rewards
