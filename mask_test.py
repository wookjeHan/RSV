import argparse
import numpy as np

from transformers import AutoTokenizer
from language_model import DummyClassificationModel

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataset import DataModule, get_splited_dataset
from util.etc import fix_seed, set_device, Color

import resolvers
from config import GlobalConfig

def main(args):
    # Random seed
    fix_seed(args.seed)

    # Use GPU
    set_device('cuda')

    # Datasets
    trainset, valset, testset = get_splited_dataset(args)

    # Resolver & Verbalizer
    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)
    sample_shot = resolver(trainset[0:1])
    verbalizers = sample_shot['verbalizers']

    # Dataloader
    test_dataloader = DataModule(testset, resolver=resolver, batch_size=args.batch_size).get_dataloader()

    # Model
    language_model = DummyClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    trainset = trainset[:6]

    for endo_sample in [True, False]:
        env = ClassificationEnv(
            trainset,
            resolver,
            args.shot_num,
            endo_sample,
            language_model,
            tokenizer,
            verbalizers,
            2.0,
            1.0,
        )
        for replace in [True, False]:
            print(f"{Color.RED}>>>endo sample: {endo_sample} replace: {replace}>>>{Color.END}")
            policy = MlpPolicy(1024, [1024], len(trainset), replace).cuda()
            for resolved_batch in test_dataloader:
                print(f"initial indices: {resolved_batch['idx']}")
                embeddings, action_mask = env.reset(resolved_batch, 'train')
                for step in range(args.shot_num):
                    indices, replace = policy.get_actions((embeddings, action_mask), max_action=True)
                    (embeddings, action_mask), rewards = env.step((indices, replace))
                    print(f"{step}th indices: {indices.view(-1)}")
                    print(f"{step}th action_mask:\n{action_mask.detach().cpu().numpy()}")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--language_model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='superglue_cb')
    parser.add_argument('--prompt', type=str, default='manual')

    parser.add_argument('--batch_size', type=int, default=GlobalConfig.batch_size)
    parser.add_argument('--shot_num', type=int, default=GlobalConfig.shot_num)
    parser.add_argument('--tv_split_ratio', type=float, default=GlobalConfig.tv_split_ratio)
    args = parser.parse_args()

    main(args)