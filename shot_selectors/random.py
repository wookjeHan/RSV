import random

class RandomShotSelector():
    def __init__(self, trainset, shot_num, resolver, **kwargs):
        self.resolver = resolver
        selected_indices = random.sample(range(len(trainset)), shot_num)
        self.selected_data = [trainset[idx] for idx in selected_indices]
        self.resolved_shots = self.resolver(self.selected_data)

    def __call__(self, batch):
        return [self.resolved_shots for _ in batch['label']]