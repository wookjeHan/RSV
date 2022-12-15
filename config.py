from dataclasses import dataclass

# Global Configs
@dataclass
class GlobalConfig:
    language_model = 'gpt2'
    dataset = 'super_glue,cb'
    batch_size = 4
    shot_num = 3
    tv_split_ratio = 0.0
