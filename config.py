from dataclasses import dataclass

# Global Configs
@dataclass
class GlobalConfig:
    language_model = 'gpt2'
    dataset = 'superglue_cb'
    batch_size = 2
    shot_num = 3
    tv_split_ratio = 0.7