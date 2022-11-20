import argparse
import numpy as np

from transformers import AutoTokenizer
from language_model import ClassificationModel

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataset import DataModule, get_splited_dataset
from util.etc import fix_seed

import resolvers
from config import GlobalConfig

def main(args):
    # Random seed
    fix_seed(args.seed)

    # Datasets
    trainset, valset, testset = get_splited_dataset(args)

    # Resolver & Verbalizer
    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)
    sample_shot = resolver(trainset[0:1])
    verbalizers = sample_shot['verbalizers']

    # Dataloader
    test_dataloader = DataModule(testset, resolver=resolver, batch_size=args.batch_size).get_dataloader()

    # Model
    language_model = ClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    policy = MlpPolicy(len(trainset)).cuda()
    env = ClassificationEnv(
        trainset,
        resolver,
        args.shot_num,
        language_model,
        tokenizer,
        verbalizers,
        2.0,
        1.0,
    )

    for repeat in range(20):
        correct = []
        for resolved_batch in test_dataloader:
            states = env.reset(resolved_batch, 'train')

            for step in range(args.shot_num):
                actions = policy.get_actions(states, max_action=True)
                states, rewards = env.step(actions)

            correct += (rewards > 0).long().tolist()

        print(f"[{repeat:2d}/20] Accuracy {np.array(correct).mean():.3f}")


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