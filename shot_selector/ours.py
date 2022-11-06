import random

class ours_shot_selector():
    def __init__(self, trainset, shot_num, class_num, policy, **kwargs):
        self.trainset = trainset
        self.shot_num = shot_num
        if policy is None:
            self.policy = lambda x: random.random.randint(0, 55)
        else:
            self.policy = policy

    def __call__(self, batch, resolver):
        print(batch.keys())
        if self.resolved_shots is None:
            raw_sentences = [self.trainset[idx] for idx in self.indices]
            self.resolved_shots = resolver(raw_sentences, include_label=True)['resolved_input']

        batch_len = len(batch['label'])

        return [self.resolved_shots for _ in range(batch_len)]