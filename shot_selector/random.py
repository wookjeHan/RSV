import random

class random_shot_selector():
    def __init__(self, trainset, shot_num, class_num, **kwargs):
        self.trainset = trainset
        self.shot_num = shot_num
        self.class_num = class_num
        self.resolved_shots = None
        # self.indices = [0, 23, 33]
        self.indices = random.sample(range(len(trainset)), shot_num)

    def __call__(self, batch, resolver):
        if self.resolved_shots is None:
            raw_sentences = [self.trainset[idx] for idx in self.indices]
            self.resolved_shots = resolver(raw_sentences, include_label=True)['resolved_input']

        batch_len = len(batch['label'])

        return [self.resolved_shots for _ in range(batch_len)]