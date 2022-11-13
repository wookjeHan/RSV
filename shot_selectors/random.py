import random

class RandomShotSelector():
    def __init__(self, trainset, shot_num, resolver, **kwargs):
        selected_indices = random.sample(range(len(trainset)), shot_num)
        selected_data = [trainset[idx] for idx in selected_indices]
        self.resolved_shots = resolver(selected_data, include_label=True)['resolved_input']

    def __call__(self, batch):
        return [self.resolved_shots for _ in batch['label']]