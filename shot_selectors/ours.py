import torch
from rl.envs.classification_env import ClassificationEnv

class OursShotSelector():
    def __init__(self, trainset, shot_num, resolver, policy, env: ClassificationEnv, **kwargs):
        self.trainset = trainset
        self.shot_num = shot_num
        self.resolver = resolver

        self.policy = policy
        self.env = env

    def __call__(self, resolved_batch):
        batch_size = len(resolved_batch['label'])
        selected_datas = [[None for _ in range(self.shot_num)] for _ in range(batch_size)]

        states = self.env.reset(resolved_batch, mode='eval')
        for step in range(self.shot_num):
            indices, action_mask = self.policy.get_actions(states, max_action=True)
            states, _ = self.env.step((indices, action_mask))

            for i in range(batch_size):
                selected_datas[i][self.shot_num - 1 - step] = self.trainset[indices[i]]

        shots = []
        for selected_data in selected_datas:
            shots.append(self.resolver(selected_data))

        return shots