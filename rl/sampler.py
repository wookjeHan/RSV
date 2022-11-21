import torch
from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

def sample(policy: MlpPolicy, target_policy: MlpPolicy, env: ClassificationEnv, resolved_batch, shot_num):
    logit_stack = []
    target_logit_stack = []
    indices_stack = []

    target_policy.eval()

    states = env.reset(resolved_batch, 'train')
    for step in range(shot_num):
        (indices, replace), logits = policy.get_actions_logits(states)
        target_logits = target_policy.get_logits(states)
        states, rewards = env.step((indices, replace))
        logit_stack.append(logits.unsqueeze(1))
        target_logit_stack.append(target_logits.unsqueeze(1))
        indices_stack.append(indices.unsqueeze(1))

    total_logits = torch.cat(logit_stack, dim=1)
    total_target_logits = torch.cat(target_logit_stack, dim=1).detach()
    total_indices = torch.cat(indices_stack, dim=1)

    return total_logits, total_target_logits, total_indices, rewards

def argmax(policy: MlpPolicy, target_policy: MlpPolicy, env: ClassificationEnv, resolved_batch, shot_num):
    indices_stack = []

    target_policy.eval()

    states = env.reset(resolved_batch, 'train')
    for step in range(shot_num):
        (indices, action_mask) = policy.get_actions(states, max_action=True)
        states, rewards = env.step((indices, action_mask))
        indices_stack.append(indices.unsqueeze(1))

    total_indices = torch.cat(indices_stack, dim=1)

    return total_indices, rewards
