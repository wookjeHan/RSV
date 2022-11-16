import argparse
import numpy as np

from transformers import AutoTokenizer
from language_model import ClassificationModel
from datasets import load_dataset

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataloader import DataModule
from util.etc import fix_seed

import resolvers
from config import GlobalConfig

def main(args):
    # Random seed
    fix_seed(args.seed)

    # Datasets
    # TODO: split superglue_cb into super_glue and cb
    dataset = load_dataset('super_glue', 'cb')
    total_trainset = list(dataset['train'])
    total_valset = list(dataset['validation'])
    trainset_size = len(total_trainset)
    split_idx = int(trainset_size * args.tv_split_ratio)

    trainset = total_trainset[:split_idx] # Train dataset -> list of data(Dictionary)
    valset = total_trainset[split_idx:] # Validation dataset -> list of data(Dictionary)
    testset = total_valset # Eval dataset -> list of data(Dictionary)

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))
    print("Testset:", len(testset))

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