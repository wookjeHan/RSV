import torch

class OursShotSelector():
    def __init__(self, trainset, shot_num, resolver, policy, env, **kwargs):
        self.trainset = trainset
        self.shot_num = shot_num
        self.resolver = resolver

        self.policy = policy
        self.env = env

    def __call__(self, resolved_batch):
        batch_size = len(resolved_batch['label'])
        selected_datas = [[] for _ in range(batch_size)]
        action_log = torch.zeros(batch_size, self.shot_num)

        states = self.env.reset(resolved_batch, mode='eval')
        for step in range(self.shot_num):
            actions = self.policy.get_actions(states, max_action=True)
            action_log[:,step] = actions
            states, _ = self.env.step(actions)

            for i in range(batch_size):
                selected_datas[i].append(self.trainset[actions[i]])

        shots = []
        # print(action_log)
        for selected_data in selected_datas:
            shots.append(self.resolver(selected_data, include_label=True)['resolved_input'])

        return shots