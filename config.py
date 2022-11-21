from dataclasses import dataclass

# Global Configs
@dataclass
class GlobalConfig:
    language_model = 'gpt2'
    dataset = 'superglue_cb'
    batch_size = 2
    shot_num = 3
    tv_split_ratio = 0.7

# A Index of Snapshot to Test
@dataclass
class TestSnapshotIndex:
    dataset = 'superglue_cb'
    prompt = 'manual'
    tv_split_ratio = 0.0
    shot_num = 3
    topk = 8
    temperature = 1.0
    weight_decay = 0.5
    replace = True
    lr = 0.001
    seed = 0
    epoch = 490
